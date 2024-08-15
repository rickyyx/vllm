import os
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, FrozenSet, List, Optional, Union

INITIAL_RULE_ELEMENT_CACHE_SIZE = int(
    os.environ.get("VLLM_INITIAL_RULE_ELEMENT_CACHE_SIZE", 256))


class ElementType(IntEnum):
    END = auto()  # End of rule
    ALT = auto()  # Represents an "|"
    RULE_REF = auto()  # Represents a reference to another rule
    CHAR = auto()  # Represents a single character "a"
    CHAR_SET = auto()  # Represents [...]
    CHAR_SET_NOT = auto()  # Represents [^...]


@dataclass
class Element:
    """Represents a grammar element.

    Args:
        etype: The type of the element, can be one of the values in ElementType.
        code_point: The Unicode code point of the element, if this is a
            terminal element (e.g. CHAR, CHAR_SET, CHAR_SET_NOT).
        upper_bound: The upper bound of the character set range, if this is a
            character set element.
        alternatives: Determines the alternative characters in a character set.
            In case of a CHAR_SET (e.g. [abc]) we have three alternates of a,
            b, and c. Similarly for CHAR_SET_NOT of [^abc], the element
            alternatives will be interpreted as the characters that are
            excluded.
        rule_name: If this element is a rule reference, this is the name of the
            referenced rule.
    """
    etype: ElementType
    code_point: Optional[int] = None
    upper_bound: Optional[int] = None
    alternatives: Optional[Union[List["Element"], FrozenSet["Element"]]] = None
    rule_name: Optional[str] = None

    _cache: Optional[Dict[int, bool]] = None
    _frozen: bool = False

    def __post_init__(self):
        if self.is_char_set() and (self.alternatives is None ==
                                   self.upper_bound is None):
            raise ValueError("Either 'alternatives' or 'upper_bound' must be "
                             "provided for character set elements.")

        if self._cache is not None:
            raise ValueError(
                "Local Element cache should not be set externally.")
        self._cache = {}

    def freeze(self) -> "Element":
        """Freezes the element, making it immutable."""
        if self.is_char_set() or self.is_char():
            if self.alternatives is not None:
                for alt in self.alternatives:
                    alt.freeze()
                self.alternatives = frozenset(self.alternatives)
            # pre-populate the cache for fast lookup
            for code_point in range(INITIAL_RULE_ELEMENT_CACHE_SIZE):
                self.is_match(code_point)

        self._frozen = True
        return self

    def is_frozen(self) -> bool:
        """Returns whether this element is frozen."""
        return self._frozen

    def is_end(self) -> bool:
        """Checks if the element is an end element."""
        return self.etype == ElementType.END

    def is_alt(self) -> bool:
        """Checks if the element is an alternate element."""
        return self.etype == ElementType.ALT

    def is_rule_ref(self) -> bool:
        """Checks if the element is a rule reference element."""
        return self.etype == ElementType.RULE_REF

    def is_char(self) -> bool:
        """Checks if the element is a character element."""
        return self.etype == ElementType.CHAR

    def is_char_set(self) -> bool:
        """Checks if the element is a character range element."""
        return self.etype in (ElementType.CHAR_SET, ElementType.CHAR_SET_NOT)

    def is_char_alt(self) -> bool:
        """Checks if the element is a character alternate element."""
        return self.is_char_set and self.alternatives is not None

    def is_matchable(self) -> bool:
        return self.is_char() or self.is_char_set()

    def is_match(self, code_point: int) -> bool:
        """Checks if a given Unicode code point matches the element."""
        assert self._cache is not None
        if code_point in self._cache:
            return self._cache[code_point]

        matched = False
        if self.is_char():
            matched = self.code_point == code_point
        elif self.is_char_set():
            if self.upper_bound is not None:
                assert self.code_point is not None
                code_point_in_set = (self.code_point <= code_point <=
                                     self.upper_bound)
            else:
                assert self.alternatives is not None
                code_point_in_set = any(
                    alt.is_match(code_point) for alt in self.alternatives)

            if self.etype == ElementType.CHAR_SET_NOT:
                matched = not code_point_in_set
            else:
                matched = code_point_in_set

        self._cache[code_point] = matched
        return matched

    def __repr__(self) -> str:
        element_str = ""
        if self.is_end():
            element_str = "END"
        elif self.is_alt():
            element_str = "ALT"
        elif self.is_rule_ref():
            element_str = f"RULE_REF({self.rule_name})"
        elif self.is_char():
            element_str = f"CHAR({chr(self.code_point)})"
        elif self.is_char_set():
            char_set_str = ("CHAR_SET" if self.etype == ElementType.CHAR_SET
                            else "CHAR_SET_NOT")
            if self.upper_bound is not None:
                char_range = f"{chr(self.code_point)}-{chr(self.upper_bound)}"
                element_str = f"{char_set_str}({char_range})"
            else:
                element_str = f"{char_set_str}({self.alternatives})"
        else:
            element_str = "UNKNOWN"

        return f"Element({element_str})"

    def _raise_error_if_not_frozen(self):
        if not self.is_frozen():
            raise ValueError(
                "Element is mutable and cannot be hashed or used for equality "
                "comparisons Hint: Call freeze() on the object.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Element):
            return False

        self._raise_error_if_not_frozen()

        return ((self.etype == other.etype)
                and (self.code_point == other.code_point)
                and (self.upper_bound == other.upper_bound)
                and (self.alternatives == other.alternatives)
                and (self.rule_name == other.rule_name))

    def __hash__(self) -> int:
        self._raise_error_if_not_frozen()
        return hash(
            (self.etype, self.code_point, self.upper_bound,
             self.alternatives if self.alternatives is not None else None,
             self.rule_name))
