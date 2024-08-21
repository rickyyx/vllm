import pytest
from anyguide import Element, ElementType


class TestBasics:

    def test_element_freeze(self):
        element = Element(ElementType.CHAR, code_point=ord("a"))
        assert not element.is_frozen()
        element.freeze()
        assert element.is_frozen()

    def test_element_is_end(self):
        element = Element(ElementType.END)
        assert element.is_end()

    def test_element_is_alt(self):
        element = Element(ElementType.ALT)
        assert element.is_alt()

    def test_element_is_rule_ref(self):
        element = Element(ElementType.RULE_REF, rule_name="rule1")
        assert element.is_rule_ref()

    def test_element_is_char(self):
        element = Element(ElementType.CHAR, code_point=ord("a"))
        assert element.is_char()

    def test_element_is_char_set(self):
        element = Element(ElementType.CHAR_SET,
                          code_point=ord("a"),
                          upper_bound=ord("z"))
        assert element.is_char_set()

    def test_element_is_char_alt(self):
        element = Element(
            ElementType.CHAR_SET,
            alternatives=[Element(ElementType.CHAR, code_point=ord("a"))])
        assert element.is_char_alt()

        element = Element(ElementType.CHAR_SET, upper_bound=ord("z"))
        assert not element.is_char_alt()

    def test_element_is_match(self):
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

    def test_element_equality(self):
        element1 = Element(ElementType.CHAR, code_point=ord("a"))
        element1.freeze()
        element2 = Element(ElementType.CHAR, code_point=ord("a"))
        element2.freeze()
        element3 = Element(ElementType.CHAR, code_point=ord("b"))
        element3.freeze()
        assert element1 == element2
        assert element1 != element3

    def test_element_hash(self):
        element1 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
        element2 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
        assert hash(element1) == hash(element2)

    def test_element_hash_mutable(self):
        element = Element(ElementType.CHAR, code_point=ord("a"))
        with pytest.raises(RuntimeError):
            hash(element)

    def test_elements_in_set(self):
        element1 = Element(ElementType.CHAR, code_point=ord("a"))
        element1.freeze()
        element2 = Element(ElementType.CHAR, code_point=ord("a"))
        element2.freeze()
        assert len({element1, element2}) == 1


class TestHash:

    def test_hash_char_element(self):
        element1 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
        element2 = Element(ElementType.CHAR, code_point=ord("a")).freeze()
        element3 = Element(ElementType.CHAR, code_point=ord("b")).freeze()
        element4 = Element(ElementType.CHAR, code_point=ord("c")).freeze()

        element_set = set([element1, element3])
        assert element2 in element_set
        assert element4 not in element_set

    def test_hash_char_set_element(self):
        element1 = Element(ElementType.CHAR_SET,
                           code_point=ord("a"),
                           upper_bound=ord("z")).freeze()
        element2 = Element(ElementType.CHAR_SET,
                           code_point=ord("a"),
                           upper_bound=ord("z")).freeze()
        element3 = Element(ElementType.CHAR_SET,
                           code_point=ord("b"),
                           upper_bound=ord("z")).freeze()
        element4 = Element(ElementType.CHAR_SET,
                           code_point=ord("c"),
                           upper_bound=ord("z")).freeze()

        element_set = set([element1, element3])
        assert element2 in element_set
        assert element4 not in element_set

    def test_hash_char_set_not_element(self):
        element1 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("a"),
                           upper_bound=ord("z")).freeze()
        element2 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("a"),
                           upper_bound=ord("z")).freeze()
        element3 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("b"),
                           upper_bound=ord("z")).freeze()
        element4 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("c"),
                           upper_bound=ord("z")).freeze()

        element_set = set([element1, element3])
        assert element2 in element_set
        assert element4 not in element_set

    def test_hash_char_set_alt_element(self):
        element1 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("a")),
                               Element(ElementType.CHAR, code_point=ord("b"))
                           ]).freeze()
        element2 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("b")),
                               Element(ElementType.CHAR, code_point=ord("a")),
                           ]).freeze()
        element3 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("a")),
                               Element(ElementType.CHAR, code_point=ord("c"))
                           ]).freeze()
        element4 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("d")),
                               Element(ElementType.CHAR, code_point=ord("a"))
                           ]).freeze()

        element_set = set([element1, element3])
        assert element2 in element_set
        assert element4 not in element_set

    def test_hash_rule_ref(self):
        element1 = Element(ElementType.RULE_REF, rule_name="rule1").freeze()
        element2 = Element(ElementType.RULE_REF, rule_name="rule1").freeze()
        element3 = Element(ElementType.RULE_REF, rule_name="rule2").freeze()
        element4 = Element(ElementType.RULE_REF, rule_name="rule3").freeze()

        element_set = set([element1, element3])
        assert element2 in element_set
        assert element4 not in element_set


class TestEqual:

    def test_equal_char_element(self):
        element1 = Element(ElementType.CHAR, code_point=ord("a"))
        element2 = Element(ElementType.CHAR, code_point=ord("a"))
        element3 = Element(ElementType.CHAR, code_point=ord("b"))

        assert element1 == element2
        assert element1 != element3

    def test_equal_char_set_element(self):
        element1 = Element(ElementType.CHAR_SET,
                           code_point=ord("a"),
                           upper_bound=ord("z"))
        element2 = Element(ElementType.CHAR_SET,
                           code_point=ord("a"),
                           upper_bound=ord("z"))
        element3 = Element(ElementType.CHAR_SET,
                           code_point=ord("b"),
                           upper_bound=ord("z"))

        assert element1 == element2
        assert element1 != element3

    def test_equal_char_set_not_element(self):
        element1 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("a"),
                           upper_bound=ord("z"))
        element2 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("a"),
                           upper_bound=ord("z"))
        element3 = Element(ElementType.CHAR_SET_NOT,
                           code_point=ord("b"),
                           upper_bound=ord("z"))

        assert element1 == element2
        assert element1 != element3

    def test_equal_char_set_alt_element(self):
        element1 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("a")),
                               Element(ElementType.CHAR, code_point=ord("b"))
                           ])
        element2 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("b")),
                               Element(ElementType.CHAR, code_point=ord("a")),
                           ])
        element3 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("a")),
                               Element(ElementType.CHAR, code_point=ord("c"))
                           ])
        element4 = Element(ElementType.CHAR_SET,
                           alternatives=[
                               Element(ElementType.CHAR, code_point=ord("d")),
                               Element(ElementType.CHAR, code_point=ord("a"))
                           ])

        assert element1 == element2
        assert element1 != element3
        assert element1 != element4

    def test_equal_rule_ref(self):
        element1 = Element(ElementType.RULE_REF, rule_name="rule1")
        element2 = Element(ElementType.RULE_REF, rule_name="rule1")
        element3 = Element(ElementType.RULE_REF, rule_name="rule2")

        assert element1 == element2
        assert element1 != element3
