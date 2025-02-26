from typing import Type
from .. import AnyTao
from .test_interface_commands import new_tao


def test_get_active_beam_track_element(tao_cls: Type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.get_active_beam_track_element() == -1


# def test_cli_progress_bar(tao_cls: Type[AnyTao]):
#     with new_tao(
#         tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
#     ) as tao:
#         tao.get_active_beam_track_element()
