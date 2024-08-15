from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .element import Element, ElementType
from .json_utils import json_schema_to_gbnf


# TODO (Kourosh): During raising GrammarParserError, we should also capture the
# src of the grammar so that we can attempt to repro any issues in prod.
class GrammarParserError(Exception):
    pass


def _is_newline(char: str) -> bool:
    return char in "\r\n"


@dataclass
class Grammar:
    """Represents a grammar consisting of rules and symbols."""
    rules: Dict[str, List[Element]]
    start_rule_name: str
    symbol_to_string: Optional[Dict[str, str]] = None

    @classmethod
    def from_str(cls,
                 grammar_str: str,
                 start_rule_name: str = "root") -> "Grammar":
        """Constructs a Grammar object from a string of a BNF grammar."""
        parser = GrammarParser(grammar_str)
        return parser.parse(start_rule_name)

    @classmethod
    def from_json_schema(cls, schema: Union[str, Dict[str, Any]]) -> "Grammar":
        grammar_str = json_schema_to_gbnf(schema)
        return cls.from_str(grammar_str)

    @classmethod
    def from_regex(cls, pattern: str) -> "Grammar":
        # TODO (Kourosh): Implement this
        raise NotImplementedError()


class GrammarParser:
    """Parses a string representation of a BNF grammar into a Grammar object.

    For more detail about the rules visit:
    https://mc-stan.org/docs/reference-manual/syntax.html#bnf-grammars
    """

    def __init__(self, grammar_str: str):
        self.src = grammar_str
        self.reset()

    def reset(self) -> None:
        """Resets the parser to the beginning of the input src.
        This is so that parse can be called again."""
        self.pos = 0
        self.rules: Dict[str, List[Element]] = {}

        # Symbol name to string format mapping
        self.symbol_to_string: Dict[str, str] = {}

    def parse(self, start_rule_name="root") -> Grammar:
        """Parses the grammar string and returns a Grammar object."""
        try:
            return self._parse(start_rule_name)
        except GrammarParserError as e:
            raise e
        except Exception as e:
            # Capture any unexpected exceptions and re-raise them as
            # GrammarParserError
            raise GrammarParserError(f"Error parsing grammar: {str(e)}") from e

    def _parse(self, start_rule_name="root") -> Grammar:
        # Skip any initial whitespace and comments
        self.skip_space_and_comments()
        while self.pos < len(self.src):
            self.parse_rule()
        self.validate_rules()
        return Grammar(self.rules, start_rule_name, self.symbol_to_string)

    def parse_rule(self) -> None:
        """Parses a single rule in the grammar."""
        rule_name = self.parse_name()
        self.skip_space_and_comments()
        self.skip_past("::=")
        self.skip_space_and_comments()
        alternates = self.parse_alternates(rule_name)
        self.add_rule(rule_name, alternates)
        self.skip_space_and_comments()

    def parse_alternates(self, rule_name: str) -> List[Element]:
        """Parses alternates within a rule."""
        alternates = []
        while True:
            sequence = self.parse_sequence(rule_name)
            alternates.extend(sequence)
            self.skip_space_and_comments()
            if self.pos >= len(self.src) or self.src[self.pos] != "|":
                break
            alternates.append(Element(ElementType.ALT))
            self.pos += 1
            self.skip_space_and_comments()
        alternates.append(Element(ElementType.END))
        return alternates

    def parse_sequence(self, rule_name: str) -> List[Element]:
        """Parses a sequence of elements within a rule."""
        sequence = []
        while self.pos < len(self.src) and not _is_newline(self.src[self.pos]):
            if self.src[self.pos] == '"':
                sequence.extend(self.parse_literal())
            elif self.src[self.pos] == "[":
                sequence.append(self.parse_char_set())
            elif self.is_name_char(self.src[self.pos]):
                sequence.append(self.parse_rule_ref())
            elif self.src[self.pos] == "(":
                sequence.append(self.parse_group(rule_name))
            elif self.src[self.pos] in "*+?":
                sequence = self.apply_repetition(rule_name, sequence)
            else:
                break
            self.skip_space_and_comments(skip_newline=False)
        return sequence

    def parse_literal(self) -> List[Element]:
        """Parses a literal string within a rule."""
        self.skip_past('"')
        literal = []
        while self.pos < len(self.src) and self.src[self.pos] != '"':
            # Handling skip character backslash
            if self.src[self.pos] == "\\":
                # In case we are skipping a character in a literal we need to
                # skip one `\` and then capture the character.
                # For example, "\"" means literal double quote (`"`).
                # Another example, "\\n" means a literal newline character, the
                # first backslash should be skipped, and then the `\n` is
                # captured.
                self.pos += 1
            code_point = self.parse_char()
            element = Element(ElementType.CHAR, code_point)
            literal.append(element.freeze())
        self.skip_past('"')
        return literal

    def parse_char_set(self) -> Element:
        """Parses Character set within a rule.

        Character set is a rule that is of type [...] format.
        It can represent two types of sets:
        1. A set of distinct characters: [abc]
        2. A range of characters: [a-z]
        They can also be concatenated together: [a-z0-9] or [a-z0-9ABC]
        Also when [^...] is used it represents the negation of the set, i.e.
        all characters except the ones in the set.
        """
        self.skip_past("[")
        negated = False
        if self.pos < len(self.src) and self.src[self.pos] == "^":
            self.pos += 1
            negated = True
        alternatives = []
        start_code_point = None
        while self.pos < len(self.src) and self.src[self.pos] != "]":
            if self.src[self.pos] == "-" and start_code_point is not None:
                self.pos += 1
                try:
                    end_code_point = self.parse_char()
                except IndexError as e:
                    # This is raised when we have a grammar error
                    raise (GrammarParserError(
                        f"Invalid grammar encountered at position {self.pos}")
                           ) from e
                if start_code_point <= end_code_point:
                    alternatives.append(
                        Element(ElementType.CHAR_SET,
                                code_point=start_code_point,
                                upper_bound=end_code_point))
                else:
                    raise GrammarParserError(
                        "Invalid character range: start code point "
                        f"{start_code_point} is greater than end code point "
                        f"{end_code_point}")
                start_code_point = None
            else:
                code_point = self.parse_char()
                if start_code_point is None:
                    start_code_point = code_point
                else:
                    alternatives.append(
                        Element(ElementType.CHAR, code_point=start_code_point))
                    alternatives.append(
                        Element(ElementType.CHAR, code_point=code_point))
                    start_code_point = None
        self.skip_past("]")
        if start_code_point is not None:
            alternatives.append(
                Element(ElementType.CHAR, code_point=start_code_point))
        if len(alternatives
               ) == 1 and alternatives[0].etype == ElementType.CHAR_SET:
            element = alternatives[0]
        else:
            element = Element(ElementType.CHAR_SET, alternatives=alternatives)
        if negated:
            element.etype = ElementType.CHAR_SET_NOT

        element.freeze()
        return element

    def parse_rule_ref(self) -> Element:
        """Parses a rule reference within a rule."""
        rule_name = self.parse_name()
        return Element(ElementType.RULE_REF, rule_name=rule_name)

    def parse_group(self, base_rule_name: str) -> Element:
        """Parses a grouped sequence of elements within a rule."""
        self.skip_past("(")
        self.skip_space_and_comments()
        sub_rule_name = self.generate_unique_symbol(base_rule_name)
        alternates = self.parse_alternates(sub_rule_name)
        self.add_rule(sub_rule_name, alternates)
        self.skip_past(")")
        return Element(ElementType.RULE_REF, rule_name=sub_rule_name)

    def apply_repetition(self, base_rule_name: str,
                         sequence: List[Element]) -> List[Element]:
        """Applies repetition operators (*, +, ?) to a sequence of elements."""
        if not sequence:
            raise GrammarParserError(
                f"Expecting preceding item to */+/? at position {self.pos}")
        operator = self.src[self.pos]
        self.pos += 1
        sub_rule_name = self.generate_unique_symbol(base_rule_name)
        sub_rule = sequence[-1:]
        # root ::= rule?
        # gets mapped to
        # root ::= root_1; root_1 ::= rule | END;
        if operator in "*+":
            # root ::= rule*
            # gets mapped to
            # root ::= root_1; root_1 ::= rule root_1 | END
            sub_rule.append(
                Element(ElementType.RULE_REF, rule_name=sub_rule_name))
        sub_rule.append(Element(ElementType.ALT))
        if operator == "+":
            # root ::= rule+
            # gets mapped to
            # root ::= root_1; root_1 ::= rule root_1 | rule
            sub_rule.extend(sequence[-1:])
        sub_rule.append(Element(ElementType.END))
        self.add_rule(sub_rule_name, sub_rule)
        sequence[-1] = Element(ElementType.RULE_REF, rule_name=sub_rule_name)
        return sequence

    def parse_char(self) -> int:
        """Parses a single character or escape sequence."""
        encoded_bytes = self.src[self.pos].encode("utf-8")
        if len(encoded_bytes) != 1:
            raise GrammarParserError(
                f"Found a >1 byte unicode character at position {self.pos}")
        code_point = encoded_bytes[0]
        self.pos += 1
        return code_point

    def generate_unique_symbol(self, base_name: str) -> str:
        """Generates a new unique identifier for a symbol based on a base name.
        """
        counter = 1
        while True:
            symbol = f"{base_name}_{counter}"
            if symbol in self.rules:
                counter += 1
            else:
                return symbol

    def add_rule(self, rule_name: str, elements: List[Element]) -> None:
        """Adds a rule to the grammar."""

        self.rules[rule_name] = elements
        self.symbol_to_string[rule_name] = self.format_rule(elements)

    def format_rule(self, elements: List[Element]) -> str:
        """Formats a rule as a string for debugging and printing purposes."""
        rule_str = ""
        for element in elements:
            if element.is_end():
                break
            elif element.is_alt():
                rule_str += " | "
            elif element.is_rule_ref():
                assert element.rule_name is not None
                rule_str += element.rule_name + " "
            elif element.is_char():
                assert element.code_point is not None
                rule_str += self.format_char(element.code_point)
            elif element.is_char_set():
                if element.alternatives is not None:
                    chars = "".join(
                        self.format_char(alt.code_point)
                        for alt in element.alternatives)
                else:
                    assert (element.code_point is not None
                            and element.upper_bound is not None)
                    chars = self.format_char(
                        element.code_point) + "-" + self.format_char(
                            element.upper_bound)

                if element.etype == ElementType.CHAR_SET_NOT:
                    rule_str += f"[^{chars}]"
                else:
                    rule_str += f"[{chars}]"
        return rule_str.strip()

    @staticmethod
    def format_char(code_point: int) -> str:
        """Formats a character for printing purposes."""

        if code_point in [ord("-"), ord("]"), ord("^")]:
            return "\\" + chr(code_point)
        elif 0x20 <= code_point <= 0x7E:
            # printable ASCII
            # i.e. 0x20 (32 is space) to 0x7E (126 is ~)
            # http://facweb.cs.depaul.edu/sjost/it212/documents/ascii-pr.htm
            return chr(code_point)
        else:
            return f"\\u{code_point:04X}"

    def validate_rules(self) -> None:
        """Validates the parsed rules to ensure all references are defined."""
        for rule_name, rule in self.rules.items():
            if not rule[-1].is_end():
                raise GrammarParserError(
                    "Expected an end of rule in the Grammar.")
            for element in rule:
                # During validation we freeze all the elements.
                element.freeze()
                if (element.is_rule_ref()
                        and element.rule_name not in self.rules):
                    raise GrammarParserError(
                        "Expected a valid rule reference for rule "
                        f"{rule_name}: {element.rule_name} is missing.")

    def skip_space_and_comments(self, skip_newline: bool = True) -> None:
        """Skips whitespace and comments in the grammar string.
        Skips newline if skip_newline is True."""
        while self.pos < len(self.src) and (self.src[self.pos].isspace()
                                            or self.src[self.pos] == "#"):
            if self.src[self.pos] == "#":
                # Skip the entire comment line
                while self.pos < len(self.src) and not _is_newline(
                        self.src[self.pos]):
                    self.pos += 1
            else:
                if skip_newline:
                    # Skip any whitespace including newlines
                    self.pos += 1
                else:
                    # Skip only if it's not a newline character
                    if not _is_newline(self.src[self.pos]):
                        self.pos += 1
                    else:
                        # If it's a newline character and skip_newline is
                        # False, stop skipping
                        break

    def skip_past(self, expected: str) -> None:
        """Expects a specific substring at the current position."""
        if self.src[self.pos:self.pos + len(expected)] != expected:
            raise GrammarParserError(
                f"Expecting '{expected}' at position {self.pos}")
        self.pos += len(expected)

    def parse_name(self) -> str:
        """Parses a rule name at the current position in the grammar string."""
        start = self.pos
        while self.pos < len(self.src) and self.is_name_char(
                self.src[self.pos]):
            self.pos += 1
        if start == self.pos:
            raise GrammarParserError(f"Expecting name at position {self.pos}")
        return self.src[start:self.pos]

    @staticmethod
    def is_name_char(char: str) -> bool:
        """Checks if a character is a valid name character."""
        return char.isalnum() or char in "_-"
