"""
HDF5 (de)serialization for Pydantic models.

Borrowed/adapted from LUME-Genesis v4.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import numpy as np
import pydantic
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.units import pmd_unit

from ..util import import_by_name
from .types import PydanticPmdUnit

logger = logging.getLogger(__name__)


_reserved_h5_attrs = {
    "__python_class_name__",
    "__python_key_map__",
    "__python_key_order__",
    "__num_items__",
}


def _hdf5_dictify(
    data, encoding: str
) -> dict[str, Any] | np.ndarray | np.bytes_ | int | float | bool:
    """Convert ``data`` for storing into an HDF5 group."""
    if isinstance(data, dict):
        return {key: _hdf5_dictify(value, encoding=encoding) for key, value in data.items()}

    if data is None:
        return {
            "__python_class_name__": type(data).__name__,
        }
    if isinstance(data, pathlib.Path):
        return {
            "__python_class_name__": "pathlib.Path",
            "value": str(data),
        }
    if isinstance(data, str):
        return np.bytes_(data.encode(encoding))
    if isinstance(data, (int, float, bool)):
        return data
    if isinstance(data, bytes):
        # Bytes are less common than strings in our library; assume
        # byte datasets are encoded strings unless marked like so:
        return {
            "__python_class_name__": "bytes",
            "value": np.bytes_(data),
        }
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        try:
            as_array = np.asarray(data)
        except ValueError:
            pass
        else:
            if as_array.dtype.kind not in "OU":
                return {
                    "__python_class_name__": type(data).__name__,
                    "value": as_array,
                }
        items = {
            f"index_{idx}": _hdf5_dictify(value, encoding=encoding)
            for idx, value in enumerate(data)
        }
        return {
            "__python_class_name__": type(data).__name__,
            "__num_items__": len(items),
            **items,
        }
    if isinstance(data, pmd_unit):
        adapter = pydantic.TypeAdapter(PydanticPmdUnit)
        return _hdf5_dictify(adapter.dump_python(data, mode="json"), encoding=encoding)
    if isinstance(data, ParticleGroup):
        particle_data = _hdf5_dictify({"data": data.data}, encoding=encoding)
        assert isinstance(particle_data, dict)
        return {
            "__python_class_name__": "ParticleGroup",
            **particle_data,
        }

    raise NotImplementedError(type(data))


def _hdf5_make_key_map(data) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Make a key map, since not all Python keys are valid HDF5 keys.

    Returns
    -------
    Dict[str, str]
        Python key to h5 key.

    Dict[str, str]
        h5 key to Python key.
    """
    key_to_h5_key = {}
    h5_key_to_key = {}
    key_format = "{prefix}__key{idx}__"
    for idx, key in enumerate(data):
        if isinstance(key, int):
            key_to_h5_key[key] = str(key)
            h5_key_to_key[str(key)] = key
            continue
        elif key.isascii() and "/" not in key:
            # No mapping required
            h5_key_to_key[key] = key
            key_to_h5_key[key] = key
            continue

        prefix = "".join(ch for ch in key if ch.isascii() and ch not in "/")
        newkey = key_format.format(prefix=prefix, idx=idx)
        while newkey in h5_key_to_key or newkey in data:
            newkey += "_"
        h5_key_to_key[newkey] = key
        key_to_h5_key[key] = newkey
    return key_to_h5_key, h5_key_to_key


def _hdf5_store_dict(group: h5py.Group, data, encoding: str, depth=0) -> None:
    """
    Store ``data`` in ``group``.

    Parameters
    ----------
    data :
        The data to store.
    group : h5py.Group
        Group to store the data in.
    encoding : str
        Encoding to use when storing strings.
    depth : int, default=0
        Recursion depth.
    """
    for attr in _reserved_h5_attrs:
        value = data.pop(attr, None)
        if value:
            group.attrs[attr] = value

    key_to_h5_key, h5_key_to_key = _hdf5_make_key_map(data)
    if key_to_h5_key:
        group.attrs["__python_key_map__"] = json.dumps(h5_key_to_key).encode(encoding)

    for key, value in data.items():
        h5_key = key_to_h5_key.get(key, key)
        if h5_key != key:
            logger.debug(f"{depth * ' '}|- {key} (renamed: {h5_key})")
        else:
            logger.debug(f"{depth * ' '}|- {key}")

        if isinstance(value, dict):
            _hdf5_store_dict(
                group.create_group(h5_key),
                value,
                depth=depth + 1,
                encoding=encoding,
            )
        elif isinstance(value, (str, int, float, bool)):
            group.attrs[h5_key] = value
        elif isinstance(value, bytes):
            group.create_dataset(h5_key, data=np.bytes_(value))
        elif isinstance(value, np.ndarray):
            group.create_dataset(h5_key, data=value)
        else:
            raise NotImplementedError(type(value))


def store_in_hdf5_file(
    h5: h5py.Group,
    obj: pydantic.BaseModel,
    encoding: str = "utf-8",
) -> None:
    """
    Store a generic Pydantic model instance in an HDF5 file.

    Numpy arrays are handled specially, where each array in the object
    corresponds to an h5py dataset in the group.  The remainder of the data is
    stored as Pydantic-serialized JSON.

    This has limitations but is intended to support Genesis 4 input and output
    types.

    Parameters
    ----------
    h5 : h5py.Group
        The file or group to store ``dct`` in.
    obj : pydantic.BaseModel
        The object to store.
    encoding : str, default="utf-8"
        String encoding for the data.
    """

    data = obj.model_dump()

    h5.attrs["__python_class_name__"] = f"{obj.__module__}.{obj.__class__.__name__}"
    _hdf5_store_dict(h5, _hdf5_dictify(data, encoding=encoding), encoding=encoding)


def _hdf5_restore_dict(item: Union[h5py.Group, h5py.Dataset, Any], encoding: str, depth=0):
    """
    Restore a Python dictionary or native type from the given group.

    Passes through non-HDF5 types for convenience.
    """
    if isinstance(item, h5py.Dataset):
        logger.debug(f"{depth * ' '} -> dataset {item.dtype}")
        if item.dtype.kind in "SO":
            return item.asstr()[()]
        return np.asarray(item)

    if isinstance(item, h5py.Datatype):
        raise NotImplementedError(str(item))
    if not isinstance(item, h5py.Group):
        return item

    python_class_name = item.attrs.get("__python_class_name__", None)
    if python_class_name:
        logger.debug(f"{depth * ' '}|- Restoring class {python_class_name}")

    if python_class_name == "pathlib.Path":
        value = item.attrs["value"]
        return pathlib.Path(str(value))

    if python_class_name == "NoneType":
        return None

    if python_class_name == "bytes":
        if "value" in item.attrs:
            return item.attrs["value"]
        return bytes(item["value"][()])

    if python_class_name == "ParticleGroup":
        return ParticleGroup(
            data=_hdf5_restore_dict(
                item["data"],
                encoding=encoding,
                depth=depth + 1,
            )
        )

    if python_class_name in ("list", "tuple"):
        data = dict(item)
        data.update(item.attrs)
        if "__num_items__" in data:
            num_items = int(data["__num_items__"])
            items = [
                _hdf5_restore_dict(
                    data[f"index_{idx}"],
                    encoding=encoding,
                    depth=depth + 1,
                )
                for idx in range(num_items)
            ]
        elif "value" in data:
            items = np.asarray(data["value"]).tolist()
        else:
            raise NotImplementedError("Unsupported list format")

        return tuple(items) if python_class_name == "tuple" else items

    if python_class_name:
        cls = import_by_name(python_class_name)
        if not issubclass(cls, pydantic.BaseModel):
            raise NotImplementedError(python_class_name)

    logger.debug(f"{depth * ' '}| Restoring dictionary group with keys:")
    res_by_hdf5_key = {}

    for key, group in item.items():
        logger.debug(f"{depth * ' '}|- {key}")
        res_by_hdf5_key[key] = _hdf5_restore_dict(group, encoding=encoding, depth=depth + 1)

    for key, value in item.attrs.items():
        logger.debug(f"{depth * ' '}|- {key}")

        if hasattr(value, "tolist"):
            # Get the native type from a numpy scalar
            value = value.tolist()

        if isinstance(value, bytes):
            value = value.decode(encoding)

        res_by_hdf5_key[key] = value

    key_map_json = item.attrs.get("__python_key_map__", None)
    key_map = json.loads(key_map_json) if key_map_json else {}
    if not key_map:
        return res_by_hdf5_key

    return {python_key: res_by_hdf5_key[hdf5_key] for hdf5_key, python_key in key_map.items()}


def restore_from_hdf5_file(
    h5: h5py.Group,
    encoding: str = "utf-8",
) -> Optional[pydantic.BaseModel]:
    """
    Restore a Pydantic model instance from an HDF5 file stored using
    `store_in_hdf5_file`.

    Parameters
    ----------
    h5 : h5py.Group
        The file or group to restore from.
    encoding : str, default="utf-8"
        String encoding for the data.
    """
    final_classname = str(h5.attrs["__python_class_name__"])
    try:
        cls = import_by_name(final_classname)
    except Exception:
        logger.exception("Failed to import class: %s", final_classname)
        return None

    assert issubclass(cls, pydantic.BaseModel)

    logger.debug(f"Restoring h5 group as a dictionary: {h5}")
    data = _hdf5_restore_dict(h5, encoding=encoding)
    assert isinstance(data, dict)
    for key in _reserved_h5_attrs:
        data.pop(key, None)
    return cls.model_validate(data)


def pick_from_archive(h5: h5py.Group, encoding: str = "utf-8"):
    """
    Restore a single instance from an HDF5 file archive.

    Parameters
    ----------
    h5 : h5py.Group
        The file or group to restore from.
    """
    return _hdf5_restore_dict(h5, encoding=encoding)
