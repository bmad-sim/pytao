from __future__ import annotations

import gzip
import pathlib
from typing import cast

import numpy as np
import orjson
import pytest
import yaml

from pytao.model import Comb, Element, ElementHead, Lattice
from pytao.model.base import (
    ArchiveFormat,
    ArchiveFormatLike,
    date_coded_rename,
    dump_model,
    load_model,
    load_model_data,
)

extension_by_format: dict[ArchiveFormat, str] = {fmt: fmt.extension for fmt in ArchiveFormat}


@pytest.fixture
def comb() -> Comb:
    return Comb(
        s=np.asarray([0.0, 1.5, 3.0]),
        charge_live=np.asarray([4.0, 5.0, 6.0]),
        mean_x=np.asarray([-1.0, 0.0, 1.0]),
    )


@pytest.fixture(params=list(ArchiveFormat))
def format(request: pytest.FixtureRequest) -> ArchiveFormat:
    return cast(ArchiveFormat, request.param)


def test_round_trip_from_extension(
    tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat
) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    dump_model(fn, comb)

    assert fn.exists()
    assert Comb.from_file(fn) == comb


@pytest.mark.parametrize("as_str", [False, True], ids=["enum", "str"])
def test_round_trip_with_explicit_format(
    tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat, as_str: bool
) -> None:
    # An extension that `ArchiveFormat.from_filename` would call "json":
    fn = tmp_path / "comb.dat"
    format_like: ArchiveFormatLike = format.value if as_str else format  # type: ignore[assignment]
    dump_model(fn, comb, format=format_like)

    assert load_model(fn, Comb, format=format_like) == comb


def test_extension_maps_back_to_format(format: ArchiveFormat) -> None:
    assert ArchiveFormat.from_filename(f"comb{format.extension}") == format


@pytest.mark.parametrize(
    "filename, format_like, expected",
    [
        pytest.param(
            "comb.yaml",
            ArchiveFormat.msgpack,
            ArchiveFormat.msgpack,
            id="enum_overrides_filename",
        ),
        pytest.param(
            "comb.yaml", "json.gz", ArchiveFormat.json_gz, id="str_overrides_filename"
        ),
        pytest.param("comb.yaml", None, ArchiveFormat.yaml, id="none_uses_filename"),
        pytest.param(
            pathlib.Path("comb.msgpack"), None, ArchiveFormat.msgpack, id="none_uses_path"
        ),
    ],
)
def test_from_format_or_file(
    filename: str | pathlib.Path,
    format_like: ArchiveFormatLike | None,
    expected: ArchiveFormat,
) -> None:
    result = ArchiveFormat.from_format_or_file(filename, format_like)
    assert result is expected


@pytest.mark.parametrize("format_like", ["hdf5", "", "YAML"])
def test_from_format_or_file_rejects_invalid(format_like: str) -> None:
    with pytest.raises(ValueError):
        ArchiveFormat.from_format_or_file("comb.json", format_like)  # type: ignore[arg-type]


def test_dump_returns_dumped_data(
    tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat
) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    data = dump_model(fn, comb, exclude_defaults=True)

    assert set(data) == {"s", "charge_live", "mean_x", "__class_name__"}


def test_exclude_defaults_false_includes_all_fields(
    tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat
) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    data = dump_model(fn, comb, exclude_defaults=False)

    assert set(Comb.model_fields).issubset(set(data))
    assert Comb.from_file(fn) == comb


@pytest.mark.parametrize("extension", [".yaml", ".yml"])
def test_yaml_output_is_plain_safe_yaml(
    tmp_path: pathlib.Path, comb: Comb, extension: str
) -> None:
    fn = tmp_path / f"comb{extension}"
    dump_model(fn, comb)

    # `safe_load` refuses python-specific tags, so this also asserts that the
    # dumper emitted nothing but plain YAML nodes.
    data = yaml.safe_load(fn.read_text())
    assert data["s"] == [0.0, 1.5, 3.0]
    assert data["charge_live"] == [4.0, 5.0, 6.0]


@pytest.mark.parametrize("attr", ["CSafeLoader", "CSafeDumper"])
def test_yaml_falls_back_without_libyaml(
    tmp_path: pathlib.Path,
    comb: Comb,
    monkeypatch: pytest.MonkeyPatch,
    attr: str,
) -> None:
    monkeypatch.delattr(yaml, attr, raising=False)

    fn = tmp_path / "comb.yaml"
    dump_model(fn, comb)
    assert Comb.from_file(fn) == comb


def test_yaml_c_and_python_dumpers_agree(
    tmp_path: pathlib.Path, comb: Comb, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(yaml, "CSafeDumper"):
        pytest.skip("libyaml (CSafeDumper) unavailable")

    c_fn = tmp_path / "c.yaml"
    dump_model(c_fn, comb)

    monkeypatch.delattr(yaml, "CSafeDumper")
    py_fn = tmp_path / "py.yaml"
    dump_model(py_fn, comb)

    assert c_fn.read_text() == py_fn.read_text()


def test_msgpack_raw_keeps_encoded_ndarrays(tmp_path: pathlib.Path, comb: Comb) -> None:
    fn = tmp_path / "comb.msgpack"
    dump_model(fn, comb)

    raw = load_model_data(fn, raw=True)
    assert isinstance(raw["s"], dict)
    assert raw["s"]["dtype"] == "float64"
    assert raw["s"]["shape"] == [3]

    restored = load_model_data(fn, raw=False)
    assert isinstance(restored["s"], np.ndarray)
    np.testing.assert_allclose(restored["s"], comb.s)


def test_json_gz_is_gzipped(tmp_path: pathlib.Path, comb: Comb) -> None:
    fn = tmp_path / "comb.json.gz"
    dump_model(fn, comb)

    assert fn.read_bytes()[:2] == b"\x1f\x8b"
    assert orjson.loads(gzip.decompress(fn.read_bytes()))["s"] == [0.0, 1.5, 3.0]


@pytest.mark.parametrize("format", ["json", "json.gz"])
@pytest.mark.parametrize("indent", [False, True])
def test_json_indent(
    tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat, indent: bool
) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    dump_model(fn, comb, indent=indent)

    if format == "json.gz":
        contents = gzip.decompress(fn.read_bytes())
    else:
        contents = fn.read_bytes()

    assert (b"\n" in contents) == indent
    assert Comb.from_file(fn) == comb


@pytest.mark.parametrize("format", ["json", "json.gz"])
def test_json_sort_keys(tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    data = dump_model(fn, comb, sort_keys=True)

    if format == "json.gz":
        contents = gzip.decompress(fn.read_bytes())
    else:
        contents = fn.read_bytes()

    assert list(orjson.loads(contents)) == sorted(data)


def test_load_unsupported_format(tmp_path: pathlib.Path, comb: Comb) -> None:
    fn = tmp_path / "comb.json"
    dump_model(fn, comb)

    with pytest.raises(ValueError):
        load_model_data(fn, format="hdf5")  # type: ignore


def test_dump_unsupported_format(tmp_path: pathlib.Path, comb: Comb) -> None:
    with pytest.raises(ValueError):
        dump_model(tmp_path / "comb.hdf5", comb, format="hdf5")  # type: ignore


def test_dump_unsupported_format_keeps_existing_file(
    tmp_path: pathlib.Path, comb: Comb
) -> None:
    fn = tmp_path / "comb.json"
    dump_model(fn, comb)
    original = fn.read_bytes()

    with pytest.raises(ValueError):
        dump_model(fn, comb, backup_existing=True, format="hdf5")  # type: ignore

    assert [path.name for path in tmp_path.iterdir()] == [fn.name]
    assert fn.read_bytes() == original


def test_backup_existing(tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    dump_model(fn, comb, backup_existing=True)
    dump_model(fn, comb, backup_existing=True, datefmt="backup")

    assert fn.exists()
    assert (tmp_path / f"{fn.stem}-backup{fn.suffix}").exists()


def test_no_backup_overwrites(
    tmp_path: pathlib.Path, comb: Comb, format: ArchiveFormat
) -> None:
    fn = tmp_path / f"comb{extension_by_format[format]}"
    dump_model(fn, comb, backup_existing=False)
    dump_model(fn, comb, backup_existing=False)

    assert [path.name for path in tmp_path.iterdir()] == [fn.name]


def test_date_coded_rename_missing_destination(tmp_path: pathlib.Path) -> None:
    assert date_coded_rename(tmp_path / "does-not-exist.json") is None


def test_msgpack_restores_ndarrays_in_nested_lists(tmp_path: pathlib.Path) -> None:
    lat = Lattice(
        which="model",
        elements=(
            Element(
                ele_id="0",
                which="model",
                head=ElementHead(key="BEGINNING"),
                comb=Comb(s=np.asarray([0.0, 1.0])),
            ),
            Element(
                ele_id="1",
                which="model",
                head=ElementHead(key="PIPE"),
                comb=Comb(s=np.asarray([2.0, 3.0])),
            ),
        ),
    )

    fn = tmp_path / "lat.msgpack"
    dump_model(fn, lat)

    data = load_model_data(fn, raw=False)
    for ele, expected in zip(data["elements"], [[0.0, 1.0], [2.0, 3.0]]):
        assert isinstance(ele["comb"]["s"], np.ndarray)
        np.testing.assert_allclose(ele["comb"]["s"], expected)

    assert Lattice.from_file(fn) == lat
