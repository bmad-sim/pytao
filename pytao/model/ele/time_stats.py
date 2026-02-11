from __future__ import annotations

import contextlib
import functools
import time

import pydantic
from pydantic import Field


class _PytaoStatisticsCall(pydantic.BaseModel):
    num_calls: int = 0
    total_time: float = 0.0

    @pydantic.computed_field
    def time_per_call(self) -> float:
        if self.num_calls <= 0:
            return 0.0
        return self.total_time / self.num_calls


class _PytaoStatistics(pydantic.BaseModel):
    head: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    attrs: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    bunch_params: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    comb: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    control_vars: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    floor: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    lord_slave: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    photon: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    orbit: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    twiss: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    grid_field: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    mat6: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    chamber_walls: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    wall3d: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    multipoles: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)
    wake: _PytaoStatisticsCall = Field(default_factory=_PytaoStatisticsCall)

    @property
    def by_name(self):
        return {attr: getattr(self, attr) for attr in type(self).model_fields}

    def __str__(self):
        by_name = self.by_name
        stats = sorted(
            ((name, stat) for name, stat in by_name.items()),
            key=lambda pair: pair[1].total_time,
        )
        return "\n".join(f"{name}: {stat}" for name, stat in stats)

    def reset(self):
        for attr in type(self).model_fields:
            call: _PytaoStatisticsCall = getattr(self, attr)
            call.num_calls = 0
            call.total_time = 0.0

    @contextlib.contextmanager
    def time_context(self, attr: str):
        t0 = time.monotonic()
        yield
        elapsed = time.monotonic() - t0

        info = getattr(self, attr)
        info.num_calls += 1
        info.total_time += elapsed

    def time_decorator(self, func):
        attr = func.__name__.split("_fill_")[1]

        assert hasattr(self, attr)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            with self.time_context(attr):
                return func(*args, **kwargs)

        return wrapped


_pytao_stats = _PytaoStatistics()


def get_pytao_statistics() -> _PytaoStatistics:
    return _pytao_stats
