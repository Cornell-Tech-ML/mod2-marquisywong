"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: First number to multiply.
        y: Second number to multiply.

    Returns:
    -------
        Product of x and y.

    """
    return x * y


def id(x: float) -> float:
    """Return the input unchanged.

    Args:
    ----
        x: Input number.

    Returns:
    -------
        The input number unchanged.

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: First number to add.
        y: Second number to add.

    Returns:
    -------
        Sum of x and y.

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: Number to negate.

    Returns:
    -------
        Negated number.

    """
    return -x


def lt(x: float, y: float) -> bool:
    """Check if one number is less than another.

    Args:
    ----
        x: First number to compare.
        y: Second number to compare.

    Returns:
    -------
        True if x is less than y, False otherwise.

    """
    return x < y


def eq(x: float, y: float) -> bool:
    """Check if two numbers are equal.

    Args:
    ----
        x: First number to compare.
        y: Second number to compare.

    Returns:
    -------
        True if x is equal to y, False otherwise.

    """
    return x == y


def max(x: float, y: float) -> float:
    """Return the larger of two numbers.

    Args:
    ----
        x: First number to compare.
        y: Second number to compare.

    Returns:
    -------
        The larger of x and y.

    """
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if two numbers are close in value.

    Args:
    ----
        x: First number to compare.
        y: Second number to compare.

    Returns:
    -------
        True if x and y are close in value, False otherwise.

    """
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The sigmoid of x.

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU (Rectified Linear Unit) activation function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The ReLU of x.

    """
    return max(0.0, x)


def log(x: float) -> float:
    """Compute the natural logarithm of x.

    Args:
    ----
        x: Input value, must be positive.

    Returns:
    -------
        The natural logarithm of x.

    Raises:
    ------
        ValueError: If x is less than or equal to 0.

    """
    if x <= 0:
        raise ValueError("Input must be positive for logarithm calculation.")
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exponential function.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The exponential of x.

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the inverse of x.

    Args:
    ----
        x: Input value.

    Returns:
    -------
        The inverse of x.

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Compute the derivative of the log function times a second argument.

    Args:
    ----
        x: Input value.
        d: Derivative of the output with respect to the input.

    Returns:
    -------
        The derivative of the log function times a second argument.

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of the reciprocal function times a second argument.

    Args:
    ----
        x: Input value.
        d: Derivative of the output with respect to the input.

    Returns:
    -------
        The derivative of the reciprocal function times a second argument.

    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the ReLU function times a second argument.

    Args:
    ----
        x: Input value.
        d: Derivative of the output with respect to the input.

    Returns:
    -------
        The derivative of the ReLU function times a second argument.

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float], i: Iterable[float]) -> Iterable[float]:
    """Apply a function to each element of an iterable.

    Args:
    ----
        fn: Function to apply to each element.
        i: Iterable to apply the function to.

    Returns:
    -------
        An iterable with the function applied to each element.

    """
    for x in i:
        yield fn(x)


def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Combine elements from two iterables using a given function.

    Args:
    ----
        fn: Function to apply to pairs of elements from i1 and i2.
        lst1: First iterable.
        lst2: Second iterable.

    Returns:
    -------
        An iterable with elements combined using the given function.

    """
    # shorter iterable end
    for a, b in zip(lst1, lst2):
        yield fn(a, b)

    # shorter iterable extended
    # len_x = len(x)
    # len_y = len(y)
    # max_len = max(len_x, len_y)

    # result = []
    # for i in range(max_len):
    #    a = x[i] if i < len_x else 0.0
    #    b = y[i] if i < len_y else 0.0
    #    result.append(fn(a, b))

    # return result


def reduce(fn: Callable[[float, float], float], i: Iterable[float]) -> float:
    """Reduce an iterable to a single value using a given function.

    Args:
    ----
        fn: Function to apply to the iterable.
        i: Iterable to reduce.

    Returns:
    -------
        The reduced value.

    """
    iterator = iter(i)
    try:
        result = next(iterator)
    except StopIteration:
        return 0.0
        # raise ValueError("Iterable has no initial value")

    for x in iterator:
        result = fn(result, x)

    return result


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using the neg function.

    Args:
    ----
        lst: A list of floats to be negated.

    Returns:
    -------
        A list with all elements negated.

    """
    return list(map(neg, lst))


def addLists(lst: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith.

    Args:
    ----
        lst: The first list of floats.
        lst2: The second list of floats.

    Returns:
    -------
        A list with elements being the sum of corresponding elements from lst and lst2.

    """
    return zipWith(add, lst, lst2)


def sum(lst: Iterable[float]) -> float:
    """Sum all elements in a list using the reduce function.

    Args:
    ----
        lst: A list of floats to be summed.

    Returns:
    -------
        The sum of all elements in the list.

    """
    return reduce(add, lst)


def prod(lst: Iterable[float]) -> float:
    """Multiply all elements in a list using the reduce function.

    Args:
    ----
        lst: A list of floats to be multiplied.

    Returns:
    -------
        The product of all elements in the list.

    """
    return reduce(mul, lst)
