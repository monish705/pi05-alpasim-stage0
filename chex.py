"""Minimal chex compatibility shim for the pinned OpenPI/JAX environment.

OpenPI imports ``chex`` only for lightweight shape assertions in the code paths
we use. Recent chex releases require newer JAX versions than the pinned
``jax==0.5.3`` stack in OpenPI, so we provide just the assertion helpers needed
by the upstream modules we exercise.
"""

from __future__ import annotations

from collections.abc import Iterable


def _shape_of(value) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"Value of type {type(value)!r} does not expose a shape.")
    return tuple(int(dim) for dim in shape)


def assert_equal_shape(values: Iterable[object]) -> None:
    values = list(values)
    if not values:
        return
    expected = _shape_of(values[0])
    for value in values[1:]:
        actual = _shape_of(value)
        if actual != expected:
            raise AssertionError(f"Expected shape {expected}, got {actual}.")


def assert_shape(value, expected_shape: tuple[int, ...]) -> None:
    actual = _shape_of(value)
    expected = tuple(int(dim) for dim in expected_shape)
    if actual != expected:
        raise AssertionError(f"Expected shape {expected}, got {actual}.")
