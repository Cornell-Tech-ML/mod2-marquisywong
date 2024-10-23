from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Any, Protocol, Iterable, List, Set, Tuple, Deque, Dict


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.


    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    left_vals = list(vals)
    left_vals[arg] -= epsilon

    # Calculate x + h
    right_vals = list(vals)
    right_vals[arg] += epsilon

    return (f(*right_vals) - f(*left_vals)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None: ...

    """Accumulates the derivative of the output with respect to this variable."""

    @property
    def unique_id(self) -> int: ...

    """Returns a unique identifier for this variable."""

    def is_leaf(self) -> bool: ...

    """Checks if this variable is a leaf node in the computation graph."""

    def is_constant(self) -> bool: ...

    """Checks if this variable is a constant in the computation graph."""

    @property
    def parents(self) -> Iterable["Variable"]: ...

    """Returns the parent variables of this variable in the computation graph."""

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]: ...

    """Applies the chain rule to compute derivatives of the output with respect to this variable."""


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    in_degree: Dict[int, int] = defaultdict(int)
    in_degree[variable.unique_id] = 0

    # Stack to keep track of nodes to visit, using doubly ended queue for O(1) pop and append
    stack: Deque[Variable] = deque([variable])
    visited: Set[int] = set([variable.unique_id])  # Keep track of visited nodes
    result: List[Variable] = []  # List to store the topological order

    # First pass: Calculate in-degrees and identify all nodes, using iterative DFS
    while stack:
        cur_var = stack.pop()

        # Explore the parents of the current variable, counting the incoming edges
        for var in cur_var.parents:
            # Skip constant variables since they do not have derivatives
            # Otherwise, increment the in-degree of the parent
            if not var.is_constant():
                in_degree[var.unique_id] += 1

                # If the parent has not been visited, add it to the stack
                if var.unique_id not in visited:
                    stack.append(var)
                    visited.add(var.unique_id)

    # Reset the stack and add the variable to the stack
    stack.append(variable)

    # Second pass: Topological sorting using zero in-degree nodes
    # Only add variable to the result when all its dependencies (i.e. parents) have been processed (in_degree = 0)
    while stack:
        cur_var = stack.pop()
        result.append(cur_var)

        for var in cur_var.parents:
            # If the variable is not a constant, decrement the number of incoming edges because the parent will be visited
            if not var.is_constant():
                in_degree[var.unique_id] -= 1

                # If the parent has zero incoming edges, add it to the stack to be visited
                if in_degree[var.unique_id] == 0:
                    stack.append(var)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Performs backpropagation on the computation graph to compute derivatives for the leaf nodes.

    Parameters:
    ----------
    variable : Variable
        The right-most variable in the computation graph.
    deriv : Any
        The derivative of the variable that needs to be propagated backward to the leaf nodes.

    Notes:
    -----
    This function does not return a value. Instead, it updates the derivative values of each leaf node using `accumulate_derivative`.

    """
    queue = topological_sort(variable)
    derivatives = {}
    derivatives[variable.unique_id] = deriv
    for var in queue:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivatives.setdefault(v.unique_id, 0.0)
                derivatives[v.unique_id] = derivatives[v.unique_id] + d


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
