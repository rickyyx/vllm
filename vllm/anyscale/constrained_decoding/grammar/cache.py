from typing import TYPE_CHECKING, Any, Dict, Tuple, TypeVar, Union

from .stack import Stack

if TYPE_CHECKING:
    from .trie import Trie

PrefixKey = TypeVar("PrefixKey", bound=Tuple[Stack, "Trie"])


class PrefixCache(dict):
    """A Specialized dictionary for caching stack prefixes and tries.

    Basically this is a two level dictionary where the first level is a stack
    prefix and the second level is a trie object (both are hashable).

    When a stack is looked up we see if there are any prefix keys within the
    dictionary that are prefixes of the given stack. If so there is a cache hit
    on the prefix, and then we look up the trie object in the second level
    dictionary.
    """

    def __getitem__(
            self, key: Union[PrefixKey,
                             Stack]) -> Union[Dict["Trie", Any], Any]:
        """Key is either a (stack, trie) tuple or just a stack."""
        if isinstance(key, tuple):
            stack, trie = key
            for k in self.keys():
                if stack.startswith(k):
                    return super().__getitem__(k).__getitem__(trie)
            raise KeyError
        for k in self.keys():
            if key.startswith(k):
                return super().__getitem__(k)
        raise KeyError

    def __check_single_key_valye_types(self,
                                       key: Any,
                                       value: Any = None) -> None:
        if not isinstance(key, Stack):
            raise ValueError("Single element key must be a stack")
        if value and not isinstance(value, dict):
            raise ValueError(
                "In case of stack key, Value must be a dictionary")

    def __setitem__(self, key: Union[PrefixKey, Stack], value: Any) -> None:
        if isinstance(key, tuple):
            stack, trie = key

            if stack not in self:
                self.__setitem__(stack, {})
            self[stack].__setitem__(trie, value)
        else:
            self.__check_single_key_valye_types(key, value)
            super().__setitem__(key, value)

    def __contains__(self, key: Union[PrefixKey, Stack]) -> bool:
        if isinstance(key, tuple):
            stack, trie = key
            for k in self.keys():
                if stack.startswith(k):
                    return trie in self[k]
            return False
        self.__check_single_key_valye_types(key)
        return any(key.startswith(k) for k in self.keys())

    def get_matched_stack_prefix(self, key: Stack) -> Stack:
        """Get the matched stack key in the cache."""
        for k in self.keys():
            if key.startswith(k):
                return k
        raise KeyError
