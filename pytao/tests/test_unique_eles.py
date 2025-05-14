from typing import Type

import pytest

from .. import AnyTao, TaoCommandError
from .test_interface_commands import new_tao


def test_unique_eles_superuniverse(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids() == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_universe(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1") == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_branch(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1@0") == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_element(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1@0>>0") == ["1@0>>0"]


def test_unique_eles_missing(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        with pytest.raises(TaoCommandError):
            assert tao.unique_ele_ids("foo")
