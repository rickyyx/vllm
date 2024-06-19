from vllm.anyscale.constrained_decoding.grammar.element import (Element,
                                                                ElementType)
from vllm.anyscale.constrained_decoding.grammar.stack import Stack


def create_element(etype=ElementType.CHAR,
                   code_point=None,
                   upper_bound=None,
                   alternatives=None,
                   rule_name=None):
    """A helper function to create an Element object quickly"""
    return Element(etype=etype,
                   code_point=code_point,
                   upper_bound=upper_bound,
                   alternatives=alternatives,
                   rule_name=rule_name).freeze()


class TestStackInitialization:
    """Test cases for initializing Stack instances."""

    def test_empty_initialization(self):
        """Test initializing an empty stack."""
        stack = Stack()
        assert stack.is_empty()
        assert stack.size() == 0

    def test_initialization_with_elements(self):
        """Test initializing a stack with elements."""
        elements = [
            create_element(code_point=ord('a')),
            create_element(code_point=ord('b')),
            create_element(code_point=ord('c'))
        ]
        stack = Stack(elements)
        assert not stack.is_empty()
        assert stack.size() == 3


class TestStackOperations:
    """Test cases for Stack operations like push, pop, and top."""

    def setup_method(self):
        """Setup for each method in this test class."""
        self.elements = [
            create_element(code_point=ord('a')),
            create_element(code_point=ord('b'))
        ]
        self.stack = Stack(self.elements)

    def test_push(self):
        """Test pushing an element onto the stack."""
        new_element = create_element(code_point=ord('c'))
        self.stack.push(new_element)
        assert self.stack.size() == 3
        assert self.stack.top() == new_element

    def test_pop(self):
        """Test popping an element from the stack."""
        top_element = self.stack.pop()
        assert top_element == self.elements[-1]
        assert self.stack.size() == 1

    def test_pop_empty_stack(self):
        """Test popping from an empty stack."""
        empty_stack = Stack()
        assert empty_stack.pop() is None

    def test_top(self):
        """Test retrieving the top element without popping."""
        assert self.stack.top() == self.elements[-1]


class TestStackAdvancedOperations:
    """Test cases for advanced Stack functionalities like indexing and
    startswith."""

    def setup_method(self):
        self.stack = Stack([create_element(code_point=i) for i in range(10)])

    def test_getitem_slice_normal(self):
        """Test slicing to retrieve sub-stacks from the top."""
        sub_stack = self.stack[:3]  # Top three elements
        assert isinstance(sub_stack, Stack)
        assert sub_stack.size() == 3
        assert sub_stack[0] == create_element(code_point=9)
        assert sub_stack[1] == create_element(code_point=8)
        assert sub_stack[2] == create_element(code_point=7)

    def test_getitem_slice_skip_top_element(self):
        """Test slicing to retrieve sub-stacks with negative step."""
        sub_stack = self.stack[1:]
        assert isinstance(sub_stack, Stack)
        assert sub_stack.size() == self.stack.size() - 1
        assert sub_stack.top() == create_element(code_point=8)

    def test_getitem_slice_bounds(self):
        """Test slicing with bounds that exceed the stack size."""
        sub_stack = self.stack[:100]
        assert isinstance(sub_stack, Stack)
        assert sub_stack.size() == 10
        sub_stack = self.stack[-100:]
        assert isinstance(sub_stack, Stack)
        assert sub_stack.size() == 10

    def test_startswith(self):
        """Test checking if one stack starts with another."""
        other = Stack()
        other.push(create_element(code_point=8))
        other.push(create_element(code_point=9))
        assert self.stack.startswith(other)


class TestStackComparison:
    """Test cases for equality and hash operations."""

    def test_stack_equality_basic(self):
        """Test that two stacks with the same elements are considered equal."""
        stack1 = Stack([create_element(code_point=ord('a'))])
        stack2 = Stack([create_element(code_point=ord('a'))])
        assert stack1 == stack2

    def test_stack_equality_charset(self):
        """Test that two stacks with different orders of the charset are equal.
        """
        stack1 = Stack([
            create_element(ElementType.CHAR_SET,
                           alternatives=[
                               create_element(code_point=ord('a')),
                               create_element(code_point=ord('b'))
                           ])
        ])
        stack2 = Stack([
            create_element(ElementType.CHAR_SET,
                           alternatives=[
                               create_element(code_point=ord('b')),
                               create_element(code_point=ord('a'))
                           ])
        ])
        assert stack1 == stack2

    def test_stack_inequality(self):
        """Test that two stacks with different elements are not considered
        equal."""
        stack1 = Stack([create_element(code_point=ord('a'))])
        stack2 = Stack([create_element(code_point=ord('b'))])
        assert stack1 != stack2
