import pytest

from vllm.anyscale.constrained_decoding.grammar.trie import TokenizerTrie, Trie


class TestTrie:

    @pytest.fixture
    def basic_trie(self):
        basic_trie: Trie["str", int] = Trie({"foo": 1, "food": 2, "bar": 3})
        return basic_trie

    def test_setitem_and_getitem(self, basic_trie):
        basic_trie["baz"] = 4
        assert basic_trie["baz"].value == 4
        with pytest.raises(KeyError):
            basic_trie["foody"]  # pylint: disable=pointless-statement

    def test_length(self, basic_trie):
        assert len(basic_trie) == 3

    def test_keys(self, basic_trie):
        keys = basic_trie.keys()
        assert "f" in keys
        assert "b" in keys
        assert len(keys) == basic_trie.num_keys

    def test_values(self, basic_trie):
        values = basic_trie.values()
        assert len(values) == 2  # f and b

    def test_items(self, basic_trie):
        items = basic_trie.items()
        assert ("f", basic_trie["f"]) in items
        assert ("b", basic_trie["b"]) in items

    def test_is_valid(self, basic_trie):
        assert basic_trie.is_valid("foo")
        assert not basic_trie.is_valid("fo")  # codespell:ignore
        assert not basic_trie.is_valid()

    def test_eq(self, basic_trie):
        another_trie = Trie({"foo": 1, "food": 2, "bar": 3})
        assert basic_trie == another_trie

        different_trie = Trie({"foo": 1})
        assert basic_trie != different_trie

    def test_get_values(self, basic_trie):
        assert set(basic_trie["f"].get_values()) == set([1, 2])

    def test_hash(self, basic_trie):
        another_trie = Trie({"foo": 1, "baz": 4, "bar": 3})
        assert hash(basic_trie) == hash(another_trie)
        assert hash(basic_trie["f"]) == hash(another_trie["f"])
        assert hash(basic_trie["fo"]) == hash(  # codespell:ignore
            another_trie["fo"])  # codespell:ignore
        assert hash(basic_trie["foo"]) == hash(another_trie["foo"])


class TestTokenizerTrie:

    def test_from_hf(self, hf_tokenizer):

        trie = TokenizerTrie(hf_tokenizer)
        assert trie.id2str(0) == hf_tokenizer.decode(0)
        assert trie.id2str(1) == hf_tokenizer.decode(1)

    def test_id2str(self, tokenizer_trie):
        assert tokenizer_trie.id2str(0) == "<unk>"
        assert tokenizer_trie.id2str(1) == "<s>"
        assert tokenizer_trie.id2str(2) == "</s>"
        assert tokenizer_trie.id2str(1000) == "ied"
