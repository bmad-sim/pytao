from typing import Literal

from pydantic import Field, BaseModel

from pytao import Tao
from pytao.constraints.observables.base import CheckResult, IsClose, IsCloseResult, Observable, Observation
from pytao.constraints.observables.ele import TolComparison

_D2_NAME = "_pytao_tmp"
_D1_NAME = "v"
_DATUM_NAME = f"{_D2_NAME}.{_D1_NAME}[1]"


class DatumObservation(Observation):
    obs_type: Literal["datum"] = "datum"
    model_value: float
    design_value: float


class DatumObservable(Observable):
    obs_type: Literal["datum"] = "datum"
    data_type: str
    ele_name: str
    ele_start_name: str = ""
    ele_ref_name: str = ""
    eval_point: str = "END"
    data_source: str = "lat"

    @property
    def label(self) -> str:
        return f"{self.lattice_id}[{self.data_type}@{self.ele_name}]"

    def __call__(self, tao: Tao) -> DatumObservation:
        tao.data_d2_create(_D2_NAME, "1", f"{_D1_NAME}^^1^^1")
        tao.datum_create(
            _DATUM_NAME,
            self.data_type,
            ele_name=self.ele_name,
            ele_start_name=self.ele_start_name,
            ele_ref_name=self.ele_ref_name,
            eval_point=self.eval_point,
            data_source=self.data_source,
        )
        tao.data_set_design_value()
        result = tao.data_d_array(_D2_NAME, _D1_NAME)[0]
        tao.data_d2_destroy(_D2_NAME)
        return DatumObservation(
            model_value=result["model_value"],
            design_value=result["design_value"],
        )


class DatumLiteral(BaseModel):
    model_value: float
    design_value: float

    def to_observation(self) -> DatumObservation:
        return DatumObservation(model_value=self.model_value, design_value=self.design_value)


class DatumIsCloseResult(IsCloseResult):
    result_type: Literal["DatumIsCloseResult"] = "DatumIsCloseResult"
    model_value: CheckResult | None = None
    design_value: CheckResult | None = None


class DatumIsClose(IsClose):
    model_value_test: TolComparison | None = Field(default_factory=TolComparison)
    design_value_test: TolComparison | None = None

    def __call__(self, obja: DatumObservation, objb: DatumObservation) -> DatumIsCloseResult:
        model_value = None
        design_value = None

        if self.model_value_test is not None:
            model_value = self.model_value_test(obja.model_value, objb.model_value)
        if self.design_value_test is not None:
            design_value = self.design_value_test(obja.design_value, objb.design_value)

        ran = [r for r in [model_value, design_value] if r is not None]

        return DatumIsCloseResult(
            is_close=all(ran) if ran else True,
            model_value=model_value,
            design_value=design_value,
        )
