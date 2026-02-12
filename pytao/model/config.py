#!/usr/bin/env python
from __future__ import annotations

import logging
import typing
from typing import ClassVar, cast

import pydantic
from typing_extensions import Self, override

from ..core import TaoStartup
from ._generated import (
    Beam,
    BeamInit,
    BmadCom,
    SpaceChargeCom,
    TaoGlobal,
    TaoModel,
    TaoSettableModel,
)
from ._generated import logger as _tao_pystructs_logger

if typing.TYPE_CHECKING:
    from pytao import Tao

logger = logging.getLogger(__name__)
# make auto-generated log messages come from this module
setattr(_tao_pystructs_logger, "logger", logger)


class TaoConfig(TaoSettableModel):
    """
    Tao Configuration model which defines how to start Tao and configure it.

    Attributes
    ----------
    startup : TaoStartup
        The startup configuration for Tao.
    com : BmadCom
        Bmad common settings.
    space_charge_com : SpaceChargeCom
        The space charge common settings.
    beam_init : BeamInit
        Initial settings for the beam.
    beam : Beam
        The beam parameters and settings.
    globals : TaoGlobal
        Tao global settings.
    settings_by_element : dict of {str: dict of {str: str}}
        Specific settings for each element (or matching elements/ranges),
        defined as a dictionary where keys are element names and values are
        dictionary of settings keys to values.
    """

    _tao_skip_if_0_: ClassVar[tuple[str, ...]] = ()

    startup: TaoStartup = pydantic.Field(default_factory=TaoStartup)
    com: BmadCom = pydantic.Field(default_factory=BmadCom)
    space_charge_com: SpaceChargeCom = pydantic.Field(default_factory=SpaceChargeCom)
    beam_init: BeamInit = pydantic.Field(default_factory=BeamInit)
    beam: Beam = pydantic.Field(default_factory=Beam)
    globals: TaoGlobal = pydantic.Field(default_factory=TaoGlobal)
    settings_by_element: dict[str, dict[str, str]] = pydantic.Field(default_factory=dict)

    @classmethod
    @override
    def from_tao(cls, tao: Tao, **kwargs) -> TaoConfig:
        """
        Create this structure by querying Tao for its current values.

        Parameters
        ----------
        tao : Tao
        """
        return cls(
            startup=tao.init_settings,
            com=BmadCom.from_tao(tao, **kwargs),
            space_charge_com=SpaceChargeCom.from_tao(tao, **kwargs),
            beam_init=BeamInit.from_tao(tao, **kwargs),
            beam=Beam.from_tao(tao, **kwargs),
            globals=TaoGlobal.from_tao(tao, **kwargs),
        )

    @property
    def per_element_commands(self) -> list[str]:
        return [
            f"set ele {element} {attr} = {value}"
            for element, attr_to_value in self.settings_by_element.items()
            for attr, value in attr_to_value.items()
        ]

    @property
    @override
    def set_commands(self) -> list[str]:
        """
        Get all Tao 'set' commands to apply this configuration.

        Returns
        -------
        list of str
        """
        return sum(
            (
                self.com.set_commands,
                self.space_charge_com.set_commands,
                self.beam_init.set_commands,
                self.beam.set_commands,
                self.globals.set_commands,
                self.per_element_commands,
            ),
            [],
        )

    @override
    def get_set_commands(self, tao: Tao | None = None) -> list[str]:
        """
        Generate a list of set commands to apply this configuration to `tao`.

        Parameters
        ----------
        tao : Tao or None, optional
            An instance of the Tao class.
            If provided, only differing
            configuration parameters will be included in the list of set
            commands.
            If `None`, all attributes to be set will be used.

        Returns
        -------
        cmds : list of str
        """
        if tao is None:
            return self.set_commands

        return sum(
            (
                self.com.get_set_commands(tao=tao),
                self.space_charge_com.get_set_commands(tao=tao),
                self.beam_init.get_set_commands(tao=tao),
                self.beam.get_set_commands(tao=tao),
                self.globals.get_set_commands(tao=tao),
                # TODO by element if changed
                self.per_element_commands,
            ),
            [],
        )


class TaylorMap(TaoModel):
    _tao_command_: ClassVar[str] = "taylor_map"
    ele1: str
    ele2: str
    order: int
    taylor_map: dict[int, dict[tuple[int, ...], float]]

    def query(self, tao: Tao) -> Self:
        return self.from_tao(tao, ele1_id=self.ele1, ele2_id=self.ele2, order=self.order)

    @classmethod
    @override
    def from_tao(
        cls: type[Self],
        tao: Tao,
        ele1_id: str | int,
        ele2_id: str | int,
        *,
        order: str | int = 1,
    ) -> Self:
        """
        Create a TaylorMap instance from Tao.

        Parameters
        ----------
        tao : Tao
        ele1_id : str or int
            First element ID.
        ele2_id : str or int
            Second element ID.
        order : str or int, default = 1
            Talor map order.

        Returns
        -------
        TaylorMap
        """
        data = cast(
            dict[int, dict[tuple[int, ...], float]],
            tao.taylor_map(
                ele1_id=str(ele1_id),
                ele2_id=str(ele2_id),
                order=str(order),
                raises=True,
            ),
        )
        return cls(ele1=str(ele1_id), ele2=str(ele2_id), order=order, taylor_map=data)
