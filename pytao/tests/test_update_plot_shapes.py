import logging
from typing import Union

from .. import SubprocessTao, Tao
from .conftest import get_example

logger = logging.getLogger(__name__)


AnyTao = Union[Tao, SubprocessTao]


def test_update_plot_shapes(use_subprocess: bool):
    example = get_example("optics_matching")
    tao = example.run(use_subprocess=use_subprocess)

    orig_shapes = set(shape["shape"] for shape in tao.shape_list("lat_layout"))
    tao.update_plot_shapes(layout=True, shape="xbox")
    new_shapes = set(shape["shape"] for shape in tao.shape_list("lat_layout"))

    assert set(new_shapes) != set(orig_shapes)
    assert set(new_shapes) == {"xbox"}


def test_update_plot_shape_by_id():
    example = get_example("optics_matching")
    tao = example.run(use_subprocess=False)

    (orig_shape,) = [
        shape for shape in tao.shape_list("lat_layout") if shape["shape_index"] == 1
    ]

    expected_shape = "xbox" if orig_shape["shape"] == "box" else "box"
    tao.update_plot_shapes(layout=True, shape_index=1, shape=expected_shape)
    (new_shape,) = [
        shape for shape in tao.shape_list("lat_layout") if shape["shape_index"] == 1
    ]
    assert new_shape["shape"] == expected_shape

    # Minus the shape, they should be identical
    orig_shape.pop("shape")
    new_shape.pop("shape")

    assert orig_shape == new_shape


def test_update_plot_shape_by_name():
    example = get_example("optics_matching")
    tao = example.run(use_subprocess=False)

    ele_name = "quadrupole::*"
    (orig_shape,) = [
        shape for shape in tao.shape_list("lat_layout") if shape["ele_name"] == ele_name
    ]

    expected_shape = "xbox" if orig_shape["shape"] == "box" else "box"
    tao.update_plot_shapes(layout=True, ele_name=ele_name, shape=expected_shape)
    (new_shape,) = [
        shape for shape in tao.shape_list("lat_layout") if shape["ele_name"] == ele_name
    ]
    assert new_shape["shape"] == expected_shape

    # Minus the shape, they should be identical
    orig_shape.pop("shape")
    new_shape.pop("shape")

    assert orig_shape == new_shape
