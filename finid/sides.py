"""Canonical side values and identification policy for FinIdentification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, TypeAlias


Side: TypeAlias = Literal["LEFT", "RIGHT"]
SIDES: tuple[Side, ...] = ("LEFT", "RIGHT")


def identification_enabled(
    side: str | None,
    exclude_right_identification: bool,
) -> bool:
    """Return whether a detection side may participate in identification.

    Parameters:
        side: Normalized detection side, or ``None`` for an unsided class.
        exclude_right_identification: Whether RIGHT-side identification is disabled.

    Returns:
        ``True`` when the side is permitted to use identification results.
    """
    return side != "RIGHT" or not exclude_right_identification


def identification_sides(exclude_right_identification: bool) -> tuple[Side, ...]:
    """Return canonical sides currently enabled for identification.

    Parameters:
        exclude_right_identification: Whether RIGHT-side identification is disabled.

    Returns:
        Enabled sides in stable LEFT, RIGHT order.
    """
    return tuple(
        side
        for side in SIDES
        if identification_enabled(side, exclude_right_identification)
    )


def winning_side(sides: Iterable[str | None]) -> Side | None:
    """Choose LEFT on a bilateral conflict, otherwise the qualifying side.

    Parameters:
        sides: Normalized sides contributing to one classification.

    Returns:
        ``LEFT``, ``RIGHT``, or ``None`` when no canonical side was supplied.
    """
    values = set(sides)
    return "LEFT" if "LEFT" in values else "RIGHT" if "RIGHT" in values else None
