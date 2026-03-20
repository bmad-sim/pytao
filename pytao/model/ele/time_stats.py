from __future__ import annotations

import contextlib
import functools
import time

import pydantic


class _PytaoStatisticsCall(pydantic.BaseModel):
    num_calls: int = 0
    total_time: float = 0.0

    @pydantic.computed_field
    def time_per_call(self) -> float:
        if self.num_calls <= 0:
            return 0.0
        return self.total_time / self.num_calls


class _PytaoStatistics(pydantic.BaseModel):
    head: _PytaoStatisticsCall = _PytaoStatisticsCall()
    attrs: _PytaoStatisticsCall = _PytaoStatisticsCall()
    bunch_params: _PytaoStatisticsCall = _PytaoStatisticsCall()
    comb: _PytaoStatisticsCall = _PytaoStatisticsCall()
    control_vars: _PytaoStatisticsCall = _PytaoStatisticsCall()
    floor: _PytaoStatisticsCall = _PytaoStatisticsCall()
    lord_slave: _PytaoStatisticsCall = _PytaoStatisticsCall()
    photon: _PytaoStatisticsCall = _PytaoStatisticsCall()
    orbit: _PytaoStatisticsCall = _PytaoStatisticsCall()
    twiss: _PytaoStatisticsCall = _PytaoStatisticsCall()
    grid_field: _PytaoStatisticsCall = _PytaoStatisticsCall()
    mat6: _PytaoStatisticsCall = _PytaoStatisticsCall()
    chamber_walls: _PytaoStatisticsCall = _PytaoStatisticsCall()
    wall3d: _PytaoStatisticsCall = _PytaoStatisticsCall()
    multipoles: _PytaoStatisticsCall = _PytaoStatisticsCall()
    wake: _PytaoStatisticsCall = _PytaoStatisticsCall()

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
