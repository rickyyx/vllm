from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
from transformers import PreTrainedTokenizerBase

K = TypeVar("K")
V = TypeVar("V")


class Trie(Generic[K, V]):
    """Generic trie (prefix tree) data structure for efficient prefix matching
    and retrieval.

    Can be used to store any type of keys and values where the key can be
    iterated over.

    In the comments, value node refers to a node on this tree that has a
    non-None value (corresponds to a valid key), and leaf node refers to a node
    that has no children. All leaf nodes are value nodes, but not the other way
    around.

    Attributes:
        children: Children nodes in the trie.
        value: The value stored at this node, if None, this is not a value node
            and the path to this node is a valid key.
    """

    def __init__(self,
                 data: Optional[Dict[K, V]] = None,
                 path: Optional[Tuple[K, ...]] = None):
        """Creates a new trie from a dictionary of keys and values.

        Example:
            >>> trie = Trie({"ab": 1, "a": 2})
            >>> trie["ab"]
            Trie(value=1)
            >>> trie["a"]
            Trie(value=2)

        Args:
            data: A dictionary of keys and values to insert into the trie.
            path: A tuple representing the path of keys to this node.
        """
        self.children: Dict[K, Trie[K, V]] = {}
        self.value: Optional[V] = None

        self.path = path if path is not None else tuple()

        # _size holds the number of value nodes under the current node.
        self._size: int = 0
        # _value holds the values of all the value nodes under the current
        # node for fast lookup (including the value of the current node)
        self._values: List[V] = []
        # A numpy array representation of self._values
        self._values_mask: Optional[np.ndarray] = None
        if data:
            for key, value in data.items():
                self.__setitem__(key, value)

    def __setitem__(self, key: K, value: V) -> None:
        """Inserts a key-value pair into the trie."""
        if not key:
            # Since the key step at i + 1 is key[1:] from step i
            # when we reach to an empty key, we are a leaf node.
            if self.value is None:
                self._size += 1
                self._values.append(value)
            self.value = value
            return
        first, rest = key[0], key[1:]
        new_path = self.path + (first, )
        if first not in self.children:
            self.children[first] = Trie(path=new_path)
        size_before = len(self.children[first])
        self.children[first].__setitem__(rest, value)
        size_after = len(self.children[first])
        if size_after > size_before:
            self._size += 1
            self._values.append(value)

    def __getitem__(self, key: K) -> "Trie[K, V]":
        """Retrieves the node corresponding to the given key."""
        if not key:
            return self
        first, rest = key[0], key[1:]
        if first not in self.children:
            raise KeyError(key)
        return self.children[first].__getitem__(rest)

    def __len__(self) -> int:
        """Returns the total number of value nodes stored in the trie."""
        return self._size

    def keys(self) -> List[K]:
        """Returns a list of all keys stored in the trie at the current depth.
        """
        return list(self.children.keys())

    def values(self) -> List["Trie[K, V]"]:
        """Returns a list of all the sub-tries at the current depth."""
        return list(self.children.values())

    def items(self) -> List[Tuple[K, "Trie[K, V]"]]:
        """Returns a list of key-value pairs stored in the trie at the current
        depth."""
        return list(self.children.items())

    @property
    def num_keys(self) -> int:
        return len(self.children)

    def is_valid(self, key: Optional[K] = None) -> bool:
        """Determines if the node at the end of the specified key is a valid
        key. Valid keys are those paths that end in a value node.

        If key is None, this method determines if the current node is a value
        node.

        Example:
            >>> trie = Trie({"ab": 1, "a": 2})
            >>> trie.is_valid("ab")
            True
            >>> trie.is_valid("a")
            True
            >>> trie.is_valid() # Checks if the root node is a value node.
            False
        """
        if not key:
            return self.value is not None
        first, rest = key[0], key[1:]
        if first not in self.children:
            return False
        return self.children[first].is_valid(rest)

    def __hash__(self) -> int:
        return hash((self.path, self.value))

    def __eq__(self, other: object) -> bool:
        """Checks if another trie is equal to this one.

        Two tries are equal if all the branches of the two tries are identical.
        This is used for unittesting.
        """
        if not isinstance(other, Trie):
            return False
        if self.value != other.value:
            return False
        if len(self.children) != len(other.children):
            return False
        for key, child in self.children.items():
            if key not in other.children or child != other.children[key]:
                return False
        return True

    def get_values(self) -> List[V]:
        """Returns the value of all the value nodes under the current node.

        It also includes the value of the current node if it is a value node.
        """
        return self._values

    def get_value_mask(self) -> np.ndarray:
        """Returns a numpy array representation of the values stored in the trie
        at the current depth. The array is of shape (num_values,).
        """
        if self._values_mask is None:
            raise ValueError("Value mask is not computed. "
                             "Hint: Call `set_value_mask_recursive` first.")
        return self._values_mask

    def set_values_mask(self, vocab_size: int):
        """Returns a mask of all the value nodes under the current node.

        The mask is a boolean array of size vocab_size where the value is True
        if the index is a value node and False otherwise.
        """
        mask = np.zeros(vocab_size, dtype=np.bool_)
        mask[self._values] = True
        self._values_mask = mask

    def set_value_mask_recursive(self, vocab_size: int) -> None:
        """Calls get_values_mask recursively on all nodes in the trie."""
        self.set_values_mask(vocab_size)
        for child in self.children.values():
            child.set_value_mask_recursive(vocab_size)


class TokenizerTrie(Trie):
    """Specialized trie that works with tokens from a tokenizer.

    Args:
        tokenizer: The huggingface tokenizer to use for tokenization.

    Attributes:
        tokenizer: The huggingface tokenizer to use for tokenization.
        eos_token_id: The id of the end of sequence token.
        vocab_size: The size of the vocabulary of the tokenizer.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 vocab_size: Optional[int] = None):
        self._id_to_token_map = self._create_id_to_token_map(tokenizer)
        self._token_to_id_map = {
            v: k
            for k, v in self._id_to_token_map.items()
        }

        # TODO (Kourosh): Length of the two maps are not the same, Does this
        # cause any problem? In other words, len(tokenizer_trie) != vocab_size
        super().__init__(self._token_to_id_map)
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = vocab_size or len(self.tokenizer)

        self.set_value_mask_recursive(self.vocab_size)

    def _create_id_to_token_map(
            self, tokenizer: PreTrainedTokenizerBase) -> Dict[int, str]:
        """Helper function to create a mapping from token ids to tokens.

        Some tokenizers (e.g. Llama) have different tokens for sub-words that
        are at the beginning of the word or in the middle
        (e.g. `_foo` vs. `foo`). When we are capturing these tokens in the
        trie, we want to make sure the prefixes are correctly captured. In
        order to do this, we use the following trick:

        We prepend a single character token (e.g. `0`) to the start so that
        tokens that start with whitespaces are captured with their whitespaces.
        Then we can remove the first character after decoding so that we get
        the full string representation of the token.
        """
        id_to_token_map = {}
        token_id_for_zero = tokenizer.encode("0")[-1]
        special_tokens = set(tokenizer.all_special_ids)
        for token_idx in range(len(tokenizer)):
            # Special tokens should not be part of the tokenizer trie.
            if token_idx in special_tokens:
                continue
            decode_w_zero = tokenizer.decode([token_id_for_zero,
                                              token_idx])[1:]
            id_to_token_map[token_idx] = decode_w_zero

        return id_to_token_map

    def id2str(self, token_id: int) -> str:
        if token_id in self.tokenizer.all_special_ids:
            return self.tokenizer.decode(token_id)
        if token_id not in self._id_to_token_map:
            raise KeyError(f"Token id {token_id} not found in the trie.")
        return self._id_to_token_map[token_id]
