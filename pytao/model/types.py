from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Annotated, Any, Union

import numpy as np
import pydantic
import pydantic_core
from beamphysics import ParticleGroup
from beamphysics.units import pmd_unit
from typing_extensions import NotRequired, TypedDict


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


class ParticleData(TypedDict):
    """
    ParticleGroup raw data as a dictionary.

    The following keys are required:
    * `x`, `y`, `z` are np.ndarray in units of [m]
    * `px`, `py`, `pz` are np.ndarray momenta in units of [eV/c]
    * `t` is a np.ndarray of time in [s]
    * `status` is a status coordinate np.ndarray
    * `weight` is the macro-charge weight in [C], used for all statistical calculations.
    * `species` is a proper species name: `'electron'`, etc.
    The following keys are optional:
    * `id` is an optional np.ndarray of unique IDs
    """

    # `x`, `y`, `z` are positions in units of [m]
    x: NDArray
    y: NDArray
    z: NDArray

    # `px`, `py`, `pz` are momenta in units of [eV/c]
    px: NDArray
    py: NDArray
    pz: NDArray

    # `t` is time in [s]
    t: NDArray
    status: NDArray

    # `weight` is the macro-charge weight in [C], used for all statistical
    # calculations.
    weight: NDArray

    # `species` is a proper species name: `'electron'`, etc.
    species: str
    id: NotRequired[NDArray]


class _PydanticParticleGroup:
    data: ParticleData

    @staticmethod
    def _from_dict(data: ParticleData) -> ParticleGroup:
        return ParticleGroup(data=data)

    @staticmethod
    def _as_dict(obj) -> dict:
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in obj.data.items()
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        return pydantic_core.core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                cls._as_dict, when_used="json-unless-none"
            ),
        )

    @classmethod
    def _pydantic_validate(cls, value: ParticleData | ParticleGroup) -> ParticleGroup:
        if isinstance(value, ParticleGroup):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to ParticleGroup")  # type: ignore[unreachable]


class _PydanticPmdUnit:
    unitSI: float
    unitSymbol: str
    unitDimension: tuple[int, ...]

    @staticmethod
    def _from_dict(dct: dict) -> pmd_unit:
        dct = dict(dct)
        dim = dct.pop("unitDimension", None)
        if dim is not None:
            dim = tuple(dim)
        return pmd_unit(**dct, unitDimension=dim)

    @staticmethod
    def _as_dict(obj: pmd_unit) -> dict[str, Any]:
        return {
            "unitSI": obj.unitSI,
            "unitSymbol": obj.unitSymbol,
            "unitDimension": tuple(obj.unitDimension),
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        return pydantic_core.core_schema.no_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                cls._as_dict, when_used="json-unless-none"
            ),
        )

    @classmethod
    def _pydantic_validate(cls, value: dict[str, Any] | pmd_unit | Any) -> pmd_unit:
        if isinstance(value, pmd_unit):
            return value
        if isinstance(value, dict):
            return cls._from_dict(value)
        raise ValueError(f"No conversion from {value!r} to pmd_unit")


class _PydanticNDArray:
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source: type[Any],
        handler: pydantic.GetCoreSchemaHandler,
    ) -> pydantic_core.core_schema.CoreSchema:
        def serialize(obj: np.ndarray, info: pydantic.SerializationInfo):
            if not isinstance(obj, np.ndarray):
                raise ValueError(
                    f"Only supports numpy ndarray. Got {type(obj).__name__}: {obj}"
                )

            return obj.tolist()

        return pydantic_core.core_schema.with_info_plain_validator_function(
            cls._pydantic_validate,
            serialization=pydantic_core.core_schema.plain_serializer_function_ser_schema(
                serialize, when_used="json-unless-none", info_arg=True
            ),
        )

    #: Marker key used by msgpack serialization to identify ndarray dicts.
    _MSGPACK_MARKER: str = "__ndarray__"

    @classmethod
    def _pydantic_validate(
        cls,
        value: Any | np.ndarray | Sequence | dict,
        info: pydantic.ValidationInfo | None,
    ) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, dict) and cls._MSGPACK_MARKER in value:
            return np.frombuffer(value[cls._MSGPACK_MARKER], dtype=value["dtype"]).reshape(
                value["shape"]
            )
        if isinstance(value, Sequence):
            return np.asarray(value)
        raise ValueError(f"No conversion from {value!r} to numpy ndarray")


FloatSequence = Annotated[Sequence[float], pydantic.BeforeValidator(_sequence_helper)]
IntSequence = Annotated[Sequence[int], pydantic.BeforeValidator(_sequence_to_list)]
ArgumentType = int | float | str | IntSequence | FloatSequence
PydanticPmdUnit = Annotated[pmd_unit, _PydanticPmdUnit]
PydanticParticleGroup = Annotated[ParticleGroup, _PydanticParticleGroup]
AnyPath = Union[pathlib.Path, str]
FileKey = Union[str, int]
NDArray = Annotated[np.ndarray, _PydanticNDArray]
