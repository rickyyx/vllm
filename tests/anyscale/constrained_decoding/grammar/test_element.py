import pytest

from vllm.anyscale.constrained_decoding.grammar.element import (Element,
                                                                ElementType)


def test_element_creation():
    element = Element(ElementType.END)
    assert element.etype == ElementType.END
    assert element.code_point is None
    assert element.upper_bound is None
    assert element.alternatives is None
    assert element.rule_name is None


def test_element_freeze():
    element = Element(ElementType.CHAR, code_point=ord("a"))
    assert not element.is_frozen()
    element.freeze()
    assert element.is_frozen()


def test_element_is_end():
    element = Element(ElementType.END)
    assert element.is_end()


def test_element_is_alt():
    element = Element(ElementType.ALT)
    assert element.is_alt()


def test_element_is_rule_ref():
    element = Element(ElementType.RULE_REF, rule_name="rule1")
    assert element.is_rule_ref()


def test_element_is_char():
    element = Element(ElementType.CHAR, code_point=ord("a"))
    assert element.is_char()


def test_element_is_char_set():
    element = Element(ElementType.CHAR_SET,
                      code_point=ord("a"),
                      upper_bound=ord("z"))
    assert element.is_char_set()


def test_element_is_char_alt():
    element = Element(
        ElementType.CHAR_SET,
        alternatives=[Element(ElementType.CHAR, code_point=ord("a"))])
    assert element.is_char_alt()


def test_element_is_match():
    element_char = Element(ElementType.CHAR, code_point=ord("a"))
    assert element_char.is_match(ord("a"))
    assert not element_char.is_match(ord("b"))

    element_range = Element(ElementType.CHAR_SET,
                            code_point=ord("a"),
                            upper_bound=ord("z"))
    assert element_range.is_match(ord("a"))
    assert element_range.is_match(ord("m"))
    assert element_range.is_match(ord("z"))
    assert not element_range.is_match(ord("A"))

    element_range_not = Element(ElementType.CHAR_SET_NOT,
                                code_point=ord("a"),
                                upper_bound=ord("z"))
    assert not element_range_not.is_match(ord("a"))
    assert not element_range_not.is_match(ord("m"))
    assert not element_range_not.is_match(ord("z"))
    assert element_range_not.is_match(ord("A"))


def test_element_equality():
    element1 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
    element2 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
    element3 = Element(ElementType.CHAR, code_point=ord("b")).freeze()
    assert element1 == element2
    assert element1 != element3


def test_element_hash():
    element1 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
    element2 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
    assert hash(element1) == hash(element2)


def test_element_hash_mutable():
    element = Element(ElementType.CHAR, code_point=ord("a"))
    with pytest.raises(ValueError):
        hash(element)
