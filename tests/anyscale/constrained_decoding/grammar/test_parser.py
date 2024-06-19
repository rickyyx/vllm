import pytest

from vllm.anyscale.constrained_decoding.grammar.element import (Element,
                                                                ElementType)
from vllm.anyscale.constrained_decoding.grammar.parser import (
    Grammar, GrammarParser, GrammarParserError)


def _validate_types(alternates, type_list):
    assert len(alternates) == len(type_list)
    for i, alt in enumerate(alternates):
        assert alt.etype == type_list[i]


def _assert_rules_equal(ref_rule, expected_rule):
    assert len(ref_rule) == len(expected_rule)

    for ref_element, expected_element in zip(ref_rule, expected_rule):
        assert ref_element.etype == expected_element.etype
        assert ref_element.code_point == expected_element.code_point
        assert ref_element.rule_name == expected_element.rule_name


def simple_grammars():
    """Simple single rule grammars."""
    return [("root ::= \"a\"",
             Grammar(
                 rules={
                     "root": [
                         Element(ElementType.CHAR, code_point=ord("a")),
                         Element(ElementType.END)
                     ]
                 },
                 start_rule_name="root",
             )),
            ("root ::= [a-z]",
             Grammar(
                 rules={
                     "root": [
                         Element(ElementType.CHAR_SET,
                                 code_point=ord("a"),
                                 upper_bound=ord("z")),
                         Element(ElementType.END)
                     ]
                 },
                 start_rule_name="root",
             )),
            ("root ::= [a-z0-9]",
             Grammar(
                 rules={
                     "root": [
                         Element(ElementType.CHAR_SET,
                                 alternatives=[
                                     Element(ElementType.CHAR_SET,
                                             code_point=ord("a"),
                                             upper_bound=ord("z")),
                                     Element(ElementType.CHAR_SET,
                                             code_point=ord("0"),
                                             upper_bound=ord("9")),
                                 ]),
                         Element(ElementType.END)
                     ]
                 },
                 start_rule_name="root",
             )),
            ("root ::= [^a-z0-9]",
             Grammar(
                 rules={
                     "root": [
                         Element(ElementType.CHAR_SET_NOT,
                                 alternatives=[
                                     Element(ElementType.CHAR_SET,
                                             code_point=ord("a"),
                                             upper_bound=ord("z")),
                                     Element(ElementType.CHAR_SET,
                                             code_point=ord("0"),
                                             upper_bound=ord("9")),
                                 ]),
                         Element(ElementType.END)
                     ]
                 },
                 start_rule_name="root",
             )),
            ("root ::= \"h\" | \"w\"",
             Grammar(
                 rules={
                     "root": [
                         Element(ElementType.CHAR, code_point=ord("h")),
                         Element(ElementType.ALT),
                         Element(ElementType.CHAR, code_point=ord("w")),
                         Element(ElementType.END)
                     ]
                 },
                 start_rule_name="root",
             )),
            ("root ::= [\n\t\r]",
             Grammar(
                 rules={
                     "root": [
                         Element(ElementType.CHAR_SET,
                                 alternatives=[
                                     Element(ElementType.CHAR,
                                             code_point=ord("\n")),
                                     Element(ElementType.CHAR,
                                             code_point=ord("\t")),
                                     Element(ElementType.CHAR,
                                             code_point=ord("\r")),
                                 ]),
                         Element(ElementType.END)
                     ]
                 },
                 start_rule_name="root",
             )),
            (
                r"""root ::= [^"\] | "\\" (["bfnrt] | "u")""",
                Grammar(
                    rules={
                        "root": [
                            Element(ElementType.CHAR_SET_NOT,
                                    alternatives=[
                                        Element(ElementType.CHAR,
                                                code_point=ord("\"")),
                                        Element(ElementType.CHAR,
                                                code_point=ord("\\")),
                                    ]),
                            Element(ElementType.ALT),
                            Element(ElementType.CHAR, code_point=ord("\\")),
                            Element(ElementType.RULE_REF, rule_name="root_1"),
                            Element(ElementType.END)
                        ],
                        "root_1": [
                            Element(ElementType.CHAR_SET,
                                    alternatives=[
                                        Element(ElementType.CHAR,
                                                code_point=ord("\"")),
                                        Element(ElementType.CHAR,
                                                code_point=ord("n")),
                                        Element(ElementType.CHAR,
                                                code_point=ord("b")),
                                        Element(ElementType.CHAR,
                                                code_point=ord("r")),
                                        Element(ElementType.CHAR,
                                                code_point=ord("t")),
                                        Element(ElementType.CHAR,
                                                code_point=ord("f")),
                                    ]),
                            Element(ElementType.ALT),
                            Element(ElementType.CHAR, code_point=ord("u")),
                            Element(ElementType.END)
                        ]
                    },
                    start_rule_name="root",
                ),
            )
            # Add more test cases here
            ]


def multi_rule_grammars():
    """Simple multi-rule grammars with references."""
    return [
        ("root ::= rule_1\nrule_1 ::= \"a\"",
         Grammar(
             rules={
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="rule_1"),
                     Element(ElementType.END)
                 ],
                 "rule_1": [
                     Element(ElementType.CHAR, code_point=ord("a")),
                     Element(ElementType.END)
                 ]
             },
             start_rule_name="root",
         )),
        ("# Non-DAG order\nrule_1 ::= \"a\"\nroot ::= rule_1",
         Grammar(
             rules={
                 "rule_1": [
                     Element(ElementType.CHAR, code_point=ord("a")),
                     Element(ElementType.END)
                 ],
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="rule_1"),
                     Element(ElementType.END)
                 ],
             },
             start_rule_name="root",
         )),
        ("root ::= rule_1 rule_2\nrule_1 ::= \"a\"\nrule_2 ::= \"b\"",
         Grammar(
             rules={
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="rule_1"),
                     Element(ElementType.RULE_REF, rule_name="rule_2"),
                     Element(ElementType.END)
                 ],
                 "rule_1": [
                     Element(ElementType.CHAR, code_point=ord("a")),
                     Element(ElementType.END)
                 ],
                 "rule_2": [
                     Element(ElementType.CHAR, code_point=ord("b")),
                     Element(ElementType.END)
                 ]
             },
             start_rule_name="root",
         )),
        ("rule_1 ::= \"a\" | \"b\"\nroot ::= rule_1 rule_2\nrule_2 ::= \"c\"",
         Grammar(
             rules={
                 "rule_1": [
                     Element(ElementType.CHAR, code_point=ord("a")),
                     Element(ElementType.ALT),
                     Element(ElementType.CHAR, code_point=ord("b")),
                     Element(ElementType.END)
                 ],
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="rule_1"),
                     Element(ElementType.RULE_REF, rule_name="rule_2"),
                     Element(ElementType.END)
                 ],
                 "rule_2": [
                     Element(ElementType.CHAR, code_point=ord("c")),
                     Element(ElementType.END)
                 ]
             },
             start_rule_name="root",
         )),
        ("root ::= (a | b)\na ::= \"a\"\nb ::= \"b\"",
         Grammar(
             rules={
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="root_1"),
                     Element(ElementType.END)
                 ],
                 "root_1": [
                     Element(ElementType.RULE_REF, rule_name="a"),
                     Element(ElementType.ALT),
                     Element(ElementType.RULE_REF, rule_name="b"),
                     Element(ElementType.END)
                 ],
                 "a": [
                     Element(ElementType.CHAR, code_point=ord("a")),
                     Element(ElementType.END)
                 ],
                 "b": [
                     Element(ElementType.CHAR, code_point=ord("b")),
                     Element(ElementType.END)
                 ]
             },
             start_rule_name="root",
         )),
        # Add more test cases here
    ]


def repetition_grammars():
    """Some complex grammars with repetitions."""
    return [
        (
            "root ::= \"a\"*\"b\" | rule_1\nrule_1 ::= \"c\"+\"d\" | rule_2\nrule_2  ::= \"e\"?\"f\"",  # noqa
            Grammar(
                rules={
                    "root": [
                        Element(ElementType.RULE_REF, rule_name="root_1"),
                        Element(ElementType.CHAR, code_point=ord("b")),
                        Element(ElementType.ALT),
                        Element(ElementType.RULE_REF, rule_name="rule_1"),
                        Element(ElementType.END)
                    ],
                    "root_1": [
                        Element(ElementType.CHAR, code_point=ord("a")),
                        Element(ElementType.RULE_REF, rule_name="root_1"),
                        Element(ElementType.ALT),
                        Element(ElementType.END),
                    ],
                    "rule_1": [
                        Element(ElementType.RULE_REF, rule_name="rule_1_1"),
                        Element(ElementType.CHAR, code_point=ord("d")),
                        Element(ElementType.ALT),
                        Element(ElementType.RULE_REF, rule_name="rule_2"),
                        Element(ElementType.END),
                    ],
                    "rule_1_1": [
                        Element(ElementType.CHAR, code_point=ord("c")),
                        Element(ElementType.RULE_REF, rule_name="rule_1_1"),
                        Element(ElementType.ALT),
                        Element(ElementType.CHAR, code_point=ord("c")),
                        Element(ElementType.END),
                    ],
                    "rule_2": [
                        Element(ElementType.RULE_REF, rule_name="rule_2_1"),
                        Element(ElementType.CHAR, code_point=ord("f")),
                        Element(ElementType.END),
                    ],
                    "rule_2_1": [
                        Element(ElementType.CHAR, code_point=ord("e")),
                        Element(ElementType.ALT),
                        Element(ElementType.END),
                    ]
                },
                start_rule_name="root",
            )),
        ("root ::= (\"a\" | \"b\")+",
         Grammar(
             rules={
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="root_2"),
                     Element(ElementType.END),
                 ],
                 "root_1": [
                     Element(ElementType.CHAR, code_point=ord("a")),
                     Element(ElementType.ALT),
                     Element(ElementType.CHAR, code_point=ord("b")),
                     Element(ElementType.END)
                 ],
                 "root_2": [
                     Element(ElementType.RULE_REF, rule_name="root_1"),
                     Element(ElementType.RULE_REF, rule_name="root_2"),
                     Element(ElementType.ALT),
                     Element(ElementType.RULE_REF, rule_name="root_1"),
                     Element(ElementType.END)
                 ],
             },
             start_rule_name="root",
         )),
        ("root ::= [a-z0-9]*",
         Grammar(
             rules={
                 "root": [
                     Element(ElementType.RULE_REF, rule_name="root_1"),
                     Element(ElementType.END),
                 ],
                 "root_1": [
                     Element(ElementType.CHAR_SET,
                             alternatives=[
                                 Element(ElementType.CHAR_SET,
                                         code_point=ord("a"),
                                         upper_bound=ord("z")),
                                 Element(ElementType.CHAR_SET,
                                         code_point=ord("0"),
                                         upper_bound=ord("9")),
                             ]),
                     Element(ElementType.RULE_REF, rule_name="root_1"),
                     Element(ElementType.ALT),
                     Element(ElementType.END),
                 ]
             },
             start_rule_name="root",
         ))
        # Add more test cases here
    ]


def special_case_grammars():
    """Some special cases."""
    return [
        # Comments, whitespaces, and newlines that should be ignored
        (
            "# Whitespaces and comments\n\nroot ::= \"a\"      rule_1\n\n\n\n\n       rule_1::= \"b\" # Comment\n",  # noqa
            Grammar(
                rules={
                    "root": [
                        Element(ElementType.CHAR, code_point=ord("a")),
                        Element(ElementType.RULE_REF, rule_name="rule_1"),
                        Element(ElementType.END),
                    ],
                    "rule_1": [
                        Element(ElementType.CHAR, code_point=ord("b")),
                        Element(ElementType.END),
                    ]
                },
                start_rule_name="root",
            )),
        # Add more test cases here
    ]


def generate_grammar_test_cases():
    test_cases = []
    test_cases.extend(simple_grammars())
    test_cases.extend(multi_rule_grammars())
    test_cases.extend(repetition_grammars())
    test_cases.extend(special_case_grammars())
    # Add more sub-case generators
    return test_cases


class TestGrammarParserValid:
    """End to end test for the GrammarParser for valid grammars."""

    @pytest.fixture
    def parser(self):

        def _parser(grammar_str):
            return GrammarParser(grammar_str)

        return _parser

    def freeze_rules(self, rules):
        for rule in rules.values():
            for element in rule:
                element.freeze()

    def assert_grammar_equality(self, actual: Grammar, expected: Grammar):
        self.freeze_rules(actual.rules)
        self.freeze_rules(expected.rules)
        assert actual.rules == expected.rules
        assert actual.start_rule_name == expected.start_rule_name

    @pytest.mark.parametrize("grammar_str, expected_grammar",
                             generate_grammar_test_cases())
    def test_grammar_parser_valid(self, parser, grammar_str, expected_grammar):
        parsed_grammar = parser(grammar_str).parse()
        self.assert_grammar_equality(parsed_grammar, expected_grammar)


def generate_parse_charset_cases():
    # src, e_type, code_point, upper_bound, alternatives
    # alternatives is also in format of (e_type, code_point, upper_bound)

    yield "[a-z]", Element(ElementType.CHAR_SET, ord("a"), ord("z"))
    yield "[^a-z]", Element(ElementType.CHAR_SET_NOT, ord("a"), ord("z"))
    yield "[a]", Element(ElementType.CHAR_SET,
                         alternatives=[Element(ElementType.CHAR, ord("a"))])
    yield "[^a]", Element(ElementType.CHAR_SET_NOT,
                          alternatives=[Element(ElementType.CHAR, ord("a"))])
    yield "[ab]", Element(ElementType.CHAR_SET,
                          alternatives=[
                              Element(ElementType.CHAR, ord("a")),
                              Element(ElementType.CHAR, ord("b"))
                          ])
    yield "[^ab]", Element(ElementType.CHAR_SET_NOT,
                           alternatives=[
                               Element(ElementType.CHAR, ord("a")),
                               Element(ElementType.CHAR, ord("b"))
                           ])
    yield "[a-d0-7x-z]", Element(ElementType.CHAR_SET,
                                 alternatives=[
                                     Element(ElementType.CHAR_SET, ord("a"),
                                             ord("d")),
                                     Element(ElementType.CHAR_SET, ord("0"),
                                             ord("7")),
                                     Element(ElementType.CHAR_SET, ord("x"),
                                             ord("z"))
                                 ])


def generate_parse_alternates_cases():
    yield ("root ::= a | b", len("root ::= "), [
        ElementType.RULE_REF, ElementType.ALT, ElementType.RULE_REF,
        ElementType.END
    ])
    yield ("root ::= [a-z] | [0-9]", len("root ::= "), [
        ElementType.CHAR_SET, ElementType.ALT, ElementType.CHAR_SET,
        ElementType.END
    ])
    yield ("root ::= a | b | \"foo\"", len("root ::= "), [
        ElementType.RULE_REF, ElementType.ALT, ElementType.RULE_REF,
        ElementType.ALT, ElementType.CHAR, ElementType.CHAR, ElementType.CHAR,
        ElementType.END
    ])
    yield ("root ::= a", len("root ::= "),
           [ElementType.RULE_REF, ElementType.END])


def generate_repetition_test_cases():
    yield ("root ::= \"a\"*\"b\"", "root_1", [
        Element(ElementType.CHAR, ord("a")),
        Element(ElementType.RULE_REF, rule_name="root_1"),
        Element(ElementType.ALT),
        Element(ElementType.END)
    ], [
        Element(ElementType.RULE_REF, rule_name="root_1"),
        Element(ElementType.CHAR, ord("b"))
    ])
    yield ("root ::= \"a\"+\"b\"", "root_1", [
        Element(ElementType.CHAR, ord("a")),
        Element(ElementType.RULE_REF, rule_name="root_1"),
        Element(ElementType.ALT),
        Element(ElementType.CHAR, ord("a")),
        Element(ElementType.END)
    ], [
        Element(ElementType.RULE_REF, rule_name="root_1"),
        Element(ElementType.CHAR, ord("b"))
    ])
    yield ("root ::= \"a\"?\"b\"", "root_1", [
        Element(ElementType.CHAR, ord("a")),
        Element(ElementType.ALT),
        Element(ElementType.END),
    ], [
        Element(ElementType.RULE_REF, rule_name="root_1"),
        Element(ElementType.CHAR, ord("b"))
    ])


class TestGrammarCoreComponents:
    """Tests core components of GrammarParser."""

    def setup_method(self):
        self.parser = GrammarParser("")

    def set_parser_src(self, src):
        self.parser.src = src
        self.parser.reset()

    def test_parse_char_basic(self):
        self.set_parser_src("no")
        code_point = self.parser.parse_char()
        assert code_point == ord("n")
        code_point = self.parser.parse_char()
        assert code_point == ord("o")

    def test_parse_char_skip_char(self):
        self.set_parser_src("\n")
        code_point = self.parser.parse_char()
        assert code_point == ord("\n")

    def test_parse_char_literals(self):
        # Literal "\" + "n"
        self.set_parser_src("\\n")
        code_point = self.parser.parse_char()
        assert code_point == ord("\\")
        code_point = self.parser.parse_char()
        assert code_point == ord("n")

        # literal "\"
        self.set_parser_src("\\")
        code_point = self.parser.parse_char()
        assert code_point == ord("\\")

    def test_parse_char_single_byte_utf8(self):
        # 1-byte UTF-8 character
        self.set_parser_src("\u0061")
        code_point = self.parser.parse_char()
        assert code_point == ord("a")

        # Skipped 4-byte UTF-8 character
        self.set_parser_src("\\u0061")
        code_point = self.parser.parse_char()
        assert code_point == ord("\\")
        code_point = self.parser.parse_char()
        assert code_point == ord("u")
        code_point = self.parser.parse_char()
        assert code_point == ord("0")
        code_point = self.parser.parse_char()
        assert code_point == ord("0")
        code_point = self.parser.parse_char()
        assert code_point == ord("6")
        code_point = self.parser.parse_char()
        assert code_point == ord("1")

    def test_parse_char_multi_byte_utf8(self):

        # 2,3,4-byte UTF-8 character are not supported
        for c in ["æ", "㎱", "በ3"]:
            with pytest.raises(GrammarParserError):
                self.set_parser_src(c)
                self.parser.parse_char()

    def test_parse_literal(self):
        self.set_parser_src('"hello"')
        literal = self.parser.parse_literal()
        assert len(literal) == 5
        assert all(elem.etype == ElementType.CHAR for elem in literal)
        assert "".join(chr(elem.code_point) for elem in literal) == "hello"

    @pytest.mark.parametrize("src, expected_element",
                             generate_parse_charset_cases())
    def test_parse_char_set(self, src, expected_element):
        self.set_parser_src(src)
        element = self.parser.parse_char_set()
        assert element.freeze() == expected_element.freeze()

    def test_parse_name(self):
        self.set_parser_src("root ::= a | b")
        name = self.parser.parse_name()
        assert name == "root"
        assert self.parser.pos == len("root")

        self.set_parser_src("??")
        with pytest.raises(GrammarParserError):
            self.parser.parse_name()

    def test_skip_past(self):
        self.set_parser_src("root ::= a | b")
        self.parser.skip_past("root")
        assert self.parser.pos == len("root")

        with pytest.raises(GrammarParserError):
            self.parser.skip_past("root2")
        assert self.parser.pos == len("root")

    def test_skip_space_and_comments(self):
        # Comment
        self.set_parser_src("# Comment: This is a comment\nroot ::= a | b")
        self.parser.skip_space_and_comments()
        self.parser.skip_past("root ::= a | b")

        # Space
        self.set_parser_src("  root ::= a | b")
        self.parser.skip_space_and_comments()
        self.parser.skip_past("root ::= a | b")

    def test_parse_rule_ref(self):
        self.set_parser_src("a | b")
        rule_ref = self.parser.parse_rule_ref()
        assert rule_ref.etype == ElementType.RULE_REF
        assert rule_ref.rule_name == "a"

    def test_parse_sequence(self):
        self.set_parser_src("root ::= a | b")
        # Update the state of the parser as if we have already
        # parsed the name
        self.parser.pos = len("root ::= ")
        sequence = self.parser.parse_sequence("root")

        assert len(sequence) == 1
        assert sequence[0].etype == ElementType.RULE_REF
        assert sequence[0].rule_name == "a"

    @pytest.mark.parametrize("src, rst_pos, expected_types",
                             generate_parse_alternates_cases())
    def test_parse_alternates(self, src, rst_pos, expected_types):
        self.set_parser_src(src)
        self.parser.pos = rst_pos
        alternates = self.parser.parse_alternates("root")
        assert len(alternates) == len(expected_types)
        _validate_types(alternates, expected_types)

    def test_add_rule(self):
        self.parser.reset()
        self.parser.add_rule("root", [Element(ElementType.CHAR, ord("a"))])

        assert len(self.parser.rules) == 1
        assert len(self.parser.rules["root"]) == 1
        assert self.parser.rules["root"][0].etype == ElementType.CHAR
        assert self.parser.rules["root"][0].code_point == ord("a")

        assert self.parser.symbol_to_string == {"root": "a"}

    def test_parse_group(self):
        self.set_parser_src("root ::= (a | b)")
        self.parser.pos = len("root ::= ")
        group = self.parser.parse_group("root")
        assert group.etype == ElementType.RULE_REF
        assert group.rule_name == "root_1"
        assert len(self.parser.rules) == 1
        assert len(self.parser.rules["root_1"]) == 4
        expected_types = [
            ElementType.RULE_REF, ElementType.ALT, ElementType.RULE_REF,
            ElementType.END
        ]
        _validate_types(self.parser.rules["root_1"], expected_types)

    @pytest.mark.parametrize(
        "src, sub_rule_name, expected_sub_rules, expected_rule",
        generate_repetition_test_cases())
    def test_parse_repetition(self, src, sub_rule_name, expected_sub_rules,
                              expected_rule):
        self.set_parser_src(src)
        self.parser.pos = len("root ::= ")
        root_rule = self.parser.parse_sequence("root")

        assert len(self.parser.rules) == 1
        sub_rules = self.parser.rules[sub_rule_name]
        assert len(sub_rules) == len(expected_sub_rules)

        _assert_rules_equal(root_rule, expected_rule)
        _assert_rules_equal(sub_rules, expected_sub_rules)

    def test_invalid_component(self):
        self.set_parser_src("invalid")
        with pytest.raises(GrammarParserError):
            self.parser.parse_literal()

        self.set_parser_src("[a-")
        with pytest.raises(GrammarParserError):
            self.parser.parse_char_set()


class TestGrammarFormatRule:
    """Tests format_rule method of GrammarParser class"""

    def setup_method(self):
        self.parser = GrammarParser("")

    def test_format_rule_with_end_element(self):
        elements = [
            Element(ElementType.CHAR, code_point=ord("a")),
            Element(ElementType.END)
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "a"

    def test_format_rule_with_alternation(self):
        elements = [
            Element(ElementType.CHAR, code_point=ord("a")),
            Element(ElementType.ALT),
            Element(ElementType.CHAR, code_point=ord("b")),
            Element(ElementType.END),
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "a | b"

    def test_format_rule_with_rule_reference(self):
        elements = [
            Element(ElementType.RULE_REF, rule_name="rule2"),
            Element(ElementType.END)
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "rule2"

    def test_format_rule_with_character(self):
        elements = [
            Element(ElementType.CHAR, code_point=ord("x")),
            Element(ElementType.END)
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "x"

    def test_format_rule_with_character_set(self):
        elements = [
            Element(ElementType.CHAR_SET,
                    code_point=ord("a"),
                    upper_bound=ord("z")),
            Element(ElementType.END),
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "[a-z]"

    def test_format_rule_with_negated_character_set(self):
        elements = [
            Element(ElementType.CHAR_SET_NOT,
                    code_point=ord("0"),
                    upper_bound=ord("9")),
            Element(ElementType.END),
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "[^0-9]"

    def test_format_rule_with_character_set_alternatives(self):
        elements = [
            Element(
                ElementType.CHAR_SET,
                alternatives=[
                    Element(ElementType.CHAR, code_point=ord("a")),
                    Element(ElementType.CHAR, code_point=ord("b")),
                    Element(ElementType.CHAR, code_point=ord("c")),
                ],
            ),
            Element(ElementType.END),
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "[abc]"

    def test_format_rule_with_special_characters(self):
        elements = [
            Element(ElementType.CHAR, code_point=ord("-")),
            Element(ElementType.CHAR, code_point=ord("^")),
            Element(ElementType.CHAR, code_point=ord("]")),
            Element(ElementType.END),
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "\\-\\^\\]"

    def test_format_rule_with_unicode_characters(self):
        elements = [
            Element(ElementType.CHAR, code_point=0x1F600),
            Element(ElementType.CHAR, code_point=0x1F601),
            Element(ElementType.END),
        ]
        formatted_rule = self.parser.format_rule(elements)
        assert formatted_rule == "\\u1F600\\u1F601"
