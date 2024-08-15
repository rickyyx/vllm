import pprint
from typing import Iterable, Optional, Union

from .element import Element


class Stack:
    """Represents the state of one alternate at a time

    Args:
        elements: The elements of the stack. The rightmost element is the
            top of the stack.
    """

    def __init__(self, elements: Optional[Iterable[Element]] = None):
        # elements are internally stored in a tuple so that they are
        # hashable for methods like `.startwith`
        self.elements = tuple(elements or [])

    def freeze(self) -> "Stack":
        """Freeze the stack so that it can be used as a key in a dictionary.
        """
        for el in self.elements:
            el.freeze()
        return self

    def is_frozen(self) -> bool:
        """Check if the stack is frozen.

        A stack is frozen if all its elements are frozen.
        """
        return all(el.is_frozen() for el in self.elements)

    def is_empty(self) -> bool:
        return len(self.elements) == 0

    def push(self, element: Element) -> None:
        """Push element to the top of the stack"""
        self.elements += (element, )

    def pop(self) -> Optional[Element]:
        """Pop the top element of the stack.

        If stack is empty it will return None.
        """
        if not self.is_empty():
            element = self.elements[-1]
            self.elements = self.elements[:-1]
            return element
        return None

    def size(self) -> int:
        """Returns the number of elements in the stack."""
        return len(self.elements)

    def top(self) -> Optional[Element]:
        """Returns the top element of the stack without popping it.

        If stack is empty it will return None.
        """
        if not self.is_empty():
            return self.elements[-1]
        return None

    def __repr__(self) -> str:
        """Returns a string representation of the stack

        We use the `pprint` module to make the representation more readable
        when stack is long.
        """
        # Use pprint to print the stack
        stack_str = pprint.pformat(self.elements[::-1])
        return f"Stack({stack_str})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Stack) and self.elements == other.elements

    def __hash__(self) -> int:
        return hash(self.elements)

    def __getitem__(self, index: Union[int,
                                       slice]) -> Union["Stack", "Element"]:
        """Basic indexing and slicing support for the stack.

        If index is integer then return the element at that index from the top.
        If index is a slice then it will return the elements from the top
        to the given index. (e.g. stack[:10] returns the top 10 elements,
        or stack[1:] returns everything below the top element).

        These operations should not create new element objects. It just changes
        views to the original list of elements.
        """

        if isinstance(index, int):
            return self.elements[-index - 1]

        # Calculate the effective slice indices accounting for the fact that
        # the top is at the end.
        start = (
            len(self.elements) -
            (index.stop if index.stop is not None else len(self.elements)))
        stop = (len(self.elements) -
                (index.start if index.start is not None else 0))
        step = index.step if index.step is not None else 1

        # Ensure we slice in the correct order since Python list slicing does
        # not support negative steps.
        if step < 0:
            # Adjust indices for reverse slicing
            start, stop = stop - 1, start - 1

        return Stack(self.elements[start:stop:step])

    def startswith(self, other: "Stack") -> bool:
        """Check if the other stack can be a prefix to this stack.

        For example if this stack is [1, 2, 3] (1 is top of the stack) and the
        other stack is [1, 2] then this method will return True.
        """

        # Prefix should be shorter than the stack
        return (len(self.elements) >= len(other.elements)
                and self.elements[-len(other.elements):] == other.elements)
