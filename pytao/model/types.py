from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

import numpy as np
import pydantic


def _sequence_helper(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


def _sequence_to_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (int, float)):
        return [value]
    return list(value)


FloatSequence = Annotated[Sequence[float], pydantic.BeforeValidator(_sequence_helper)]
IntSequence = Annotated[Sequence[int], pydantic.BeforeValidator(_sequence_to_list)]
ArgumentType = int | float | str | IntSequence | FloatSequence
