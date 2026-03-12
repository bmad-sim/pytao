from __future__ import annotations

import contextlib
import datetime
import orjson
import gzip
import logging
import os
import re
import pathlib
import textwrap
from collections.abc import Generator
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
)

import h5py
import numpy as np
import pydantic
from pydantic.fields import FieldInfo
from typing_extensions import Self, override

from .hdf import restore_from_hdf5_file, store_in_hdf5_file

from .types import ArgumentType

try:
    from rich.pretty import pretty_repr
except ImportError:
    pretty_repr = None


if TYPE_CHECKING:
    from pytao import Tao

logger = logging.getLogger(__name__)

DEFAULT_DATEFMT = "%Y%m%d_%H%M%S"

FILTER_TAB_COMPLETION: bool = os.environ.get("PYTAO_FILTER_TAB", "y").lower() in ("y",)


def toggle_tab_completion_filtering(enabled: bool):
    global FILTER_TAB_COMPLETION
    FILTER_TAB_COMPLETION = enabled


def _check_equality(obj1: Any, obj2: Any) -> bool:
    """
    Check equality of `obj1` and `obj2`.`

    Parameters
    ----------
    obj1 : Any
    obj2 : Any

    Returns
    -------
    bool
    """
    if not isinstance(obj1, type(obj2)):
        return False

    if isinstance(obj1, pydantic.BaseModel):
        return all(
            _check_equality(
                getattr(obj1, attr),
                getattr(obj2, attr),
            )
            for attr, fld in type(obj1).model_fields.items()
            if not fld.exclude
        )

    if isinstance(obj1, dict):
        if set(obj1) != set(obj2):
            return False

        return all(
            _check_equality(
                obj1[key],
                obj2[key],
            )
            for key in obj1
        )

    if isinstance(obj1, (list, tuple)):
        if len(obj1) != len(obj2):
            return False
        return all(
            _check_equality(obj1_value, obj2_value)
            for obj1_value, obj2_value in zip(obj1, obj2)
        )

    if isinstance(obj1, np.ndarray):
        if not obj1.shape and not obj2.shape:
            return True
        return np.allclose(obj1, obj2)

    if isinstance(obj1, float):
        return np.allclose(obj1, obj2)

    return bool(obj1 == obj2)


def _strip_metadata(data: Any, remove_keys: set[str]) -> Any:
    """Recursively remove metadata keys from nested dicts/lists."""
    if isinstance(data, dict):
        return {
            key: _strip_metadata(value, remove_keys)
            for key, value in data.items()
            if key not in remove_keys
        }
    if isinstance(data, list):
        return [_strip_metadata(item, remove_keys) for item in data]
    return data


class TaoBaseModel(
    pydantic.BaseModel,
    validate_assignment=True,
    validate_by_name=True,  # Alias or attribute name for validation is OK
):
    """
    A helper base class which allows for creating/updating an instance with Tao objects.
    """

    def write(
        self,
        filename: str | pathlib.Path,
        *,
        exclude_defaults: bool = True,
        backup_existing: bool = True,
        datefmt: str = DEFAULT_DATEFMT,
        format: ArchiveFormat | None = None,
        indent: bool = False,
        sort_keys: bool = False,
    ):
        """
        Write the model data to a file in JSON or YAML format.

        Parameters
        ----------
        filename : str or pathlib.Path
            The path to the file where the model data should be written.
            The file format is determined by the extension.
        exclude_defaults : bool, optional
            Exclude model defaults from the output. Defaults to True.
        backup_existing : bool, optional
            If a file exists at the target, rename it using `datefmt` as a backup.
            Defaults to True.
        datefmt : str, optional
            The date format for the backup file.
        format : ArchiveFormat or None, optional
            File format.  If not specified, determined by file extension.
        indent : bool, optional
            Indent the output, if applicable based on the file format.
        sort_keys : bool, optional
            Sort the output keys, if applicable based on the file format.
        """

        return dump_model(
            filename,
            self,
            exclude_defaults=exclude_defaults,
            backup_existing=backup_existing,
            datefmt=datefmt,
            format=format,
            indent=indent,
            sort_keys=sort_keys,
        )

    @classmethod
    def from_file(
        cls,
        filename: str | pathlib.Path,
        *,
        format: ArchiveFormat | None = None,
    ) -> Self:
        """
        Load Tao model data from a previously-written file.

        Parameters
        ----------
        filename : str or pathlib.Path

        Returns
        -------
        TaoModel
        """
        return load_model(pathlib.Path(filename).resolve(), cls, format=format)

    @classmethod
    def _get_all_subclasses(cls) -> set[type[Self]]:
        all_subs = set(cls.__subclasses__())
        for subclass in list(all_subs):
            all_subs.update(subclass._get_all_subclasses())
        return all_subs

    def __eq__(self, other) -> bool:
        return _check_equality(self, other)

    def __repr__(self):
        if pretty_repr is not None:
            return pretty_repr(self)
        return super().__repr__()

    @override
    def __dir__(self) -> Iterable[str]:
        if not FILTER_TAB_COMPLETION:
            return super().__dir__()

        attrs = set(super().__dir__()) - set(dir(pydantic.BaseModel))
        return [attr for attr in attrs if not attr.startswith("_")]

    def to_dict(self, include_metadata: bool = False, **kwargs) -> dict[str, Any]:
        """
        Serialize the model to a dictionary.

        Parameters
        ----------
        include_metadata : bool, optional
            If True, includes base metadata fields like `command_args` and
            `__class_name__` in the serialized output. Default is False.
        **kwargs
            Additional arguments to pass to Pydantic's `model_dump`.

        Returns
        -------
        dict[str, Any]
        """
        res = self.model_dump(**kwargs)
        if include_metadata:
            return res
        return _strip_metadata(res, {"__class_name__", "command_args"})


class TaoModel(
    TaoBaseModel,
    str_strip_whitespace=True,  # Strip whitespace from strings
    str_min_length=0,  # We can't write empty strings currently
    validate_assignment=True,
    validate_by_name=True,  # Alias or attribute name for validation is OK
    extra="allow",
):
    """
    A helper base class which allows for creating/updating an instance with Tao objects.
    """

    # The `Tao.cmd_attr` command to query this information.
    _tao_command_attr_: ClassVar[str]
    # Default arguments to pass to `Tao.cmd_attr(**default_args)`
    _tao_command_default_args_: ClassVar[dict[str, Any]]

    command_args: dict[str, ArgumentType] = pydantic.Field(
        default_factory=dict,
        frozen=True,
        description="Arguments used for the pytao command to generate this structure",
        repr=False,
    )

    def query(self, tao: Tao) -> Self:
        """Query Tao again to generate a new instance of this model."""
        return self.from_tao(tao, **self.command_args)

    @classmethod
    def _process_tao_data(cls, data) -> dict:
        return data

    @classmethod
    def from_tao(cls: type[Self], tao: Tao, **kwargs) -> Self:
        """
        Create this structure by querying Tao for its current values.

        Parameters
        ----------
        tao : Tao
        **kwargs
            Keyword arguments to pass to the relevant ``tao`` command.
        """
        cmd_kwargs = dict(cls._tao_command_default_args_)
        cmd_kwargs.update(**kwargs)

        if cls._tao_command_attr_.startswith("pipe "):
            data = tao.cmd(cls._tao_command_attr_.format(**cmd_kwargs))
        else:
            cmd = getattr(tao, cls._tao_command_attr_)
            data = cmd(**cmd_kwargs)
        data = cls._process_tao_data(data)
        return cls(command_args=cmd_kwargs, **data)

    @pydantic.model_serializer(mode="wrap")
    def _serialize_with_class_name(
        self, handler: pydantic.SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        result = handler(self)
        if isinstance(result, dict):
            result["__class_name__"] = type(self).__name__
        return result

    @pydantic.model_validator(mode="wrap")
    @classmethod
    def _discriminator_validator(
        cls, value: Any, handler: pydantic.ValidatorFunctionWrapHandler
    ) -> Any:
        if isinstance(value, dict) and "__class_name__" in value:
            value_copy = dict(value)
            clsname = value_copy.pop("__class_name__")
            if cls.__name__ == clsname:
                return handler(value_copy)

            registry = {sub.__name__: sub for sub in {cls} | cls._get_all_subclasses()}
            if clsname in registry:
                return registry[clsname].model_validate(value_copy)
            raise ValueError(f"Unable to find '{clsname}' subclass of {cls.__name__}.")

        return handler(value)


class SetField(NamedTuple):
    attr: str
    idx: int | None
    fld: FieldInfo
    value: Any


def _fix_tao_attr_name(item: SetField) -> str:
    tao_name = item.fld.alias or item.attr
    if item.idx is None:
        return tao_name
    return f"{tao_name}({item.idx + 1})"


class TaoSettableModel(TaoModel):
    """
    A helper base class which allows for setting Tao parameters based on
    instance attributes.
    """

    # Do not set these keys if the values are 0, avoiding setting other things.
    _tao_skip_if_0_: ClassVar[tuple[str, ...]]
    # The 'name' of `set name attr = value`.  If unset, uses `_tao_command_attr_`.
    _tao_set_name_: ClassVar[str] = ""

    @property
    def settable_fields(self) -> dict[str, FieldInfo]:
        """Names of all 'settable' (modifiable) fields."""
        return {
            attr: field_info
            for attr, field_info in type(self).model_fields.items()
            if not field_info.frozen
        }

    @property
    def _all_attributes_to_set(self) -> Generator[SetField]:
        for attr, fld in self.settable_fields.items():
            value = getattr(self, attr)

            if attr in self._tao_skip_if_0_ and value == 0:
                continue
            if value is None:
                continue

            if np.isscalar(value):
                yield SetField(attr, None, fld, value)
            else:
                for index, val in enumerate(value):
                    yield SetField(attr, index, fld, val)

    def _get_changed_attributes(self, tao: Tao) -> Generator[SetField]:
        current = self.query(tao)

        for attr, index, fld, value in self._all_attributes_to_set:
            current_value = getattr(current, attr)
            new_value = getattr(self, attr)
            if index is not None:
                new_value = new_value[index]
                current_value = current_value[index]

            if not _check_equality(current_value, new_value):
                yield SetField(attr, index, fld, value)

    @staticmethod
    def _make_set_value(value) -> str:
        if isinstance(value, str) and not value:
            return "''"
        return str(value)

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
        cmds = []
        if tao is not None:
            to_set = self._get_changed_attributes(tao)
        else:
            to_set = self._all_attributes_to_set

        set_name = self._tao_set_name_ or self._tao_command_attr_
        for item in to_set:
            tao_attr_name = _fix_tao_attr_name(item)
            set_value = self._make_set_value(item.value)
            cmds.append(f"set {set_name} {tao_attr_name} = {set_value}")

        if self.model_extra:
            logger.warning(
                "Unhandled extra fields - code regeneration required. Class '%s': %s.",
                type(self).__name__,
                ", ".join(sorted(self.model_extra)),
            )
        return cmds

    @property
    def set_commands(self) -> list[str]:
        """
        Get all Tao 'set' commands to apply this configuration.

        Returns
        -------
        list of str
        """
        return self.get_set_commands(tao=None)

    def set(
        self,
        tao: Tao,
        *,
        allow_errors: bool = False,
        only_changed: bool = False,
        suppress_plotting: bool = True,
        suppress_lattice_calc: bool = True,
        log: str = "DEBUG",
        exclude_matches: list[str] | None = None,
    ) -> bool:
        """
        Apply this configuration to Tao.

        Parameters
        ----------
        tao : Tao
            The Tao instance to which the configuration will be applied.
        allow_errors : bool, default=False
            Allow individual commands to raise errors.
        only_changed : bool, default=False
            Only apply changes that differ from the current configuration in Tao.
        suppress_plotting : bool, default=True
            Suppress any plotting updates during the commands.
        suppress_lattice_calc : bool, default=True
            Suppress lattice calculations during the commands.
        log : str, default="DEBUG"
            The log level to use during the configuration application.

        Returns
        -------
        success : bool
            Returns True if the configuration was applied without errors.
        """
        cmds = self.get_set_commands(tao=tao if only_changed else None)
        if not cmds:
            return True

        success = True

        tao_global = cast(dict[str, Any], tao.tao_global())
        plot_on = tao_global["plot_on"]
        lat_calc_on = tao_global["lattice_calc_on"]

        exclude_matches = list(exclude_matches or [])

        if suppress_plotting and plot_on:
            tao.cmd("set global plot_on = F")
        if suppress_lattice_calc and lat_calc_on:
            tao.cmd("set global lattice_calc_on = F")

        exclude = [re.compile(line, flags=re.IGNORECASE) for line in exclude_matches]

        log_level: int = getattr(logging, log.upper())

        try:
            for cmd in self.get_set_commands(tao=tao if only_changed else None):
                if any(regex.match(cmd) for regex in exclude):
                    continue
                try:
                    logger.log(log_level, f"Tao> {cmd}")
                    for line in tao.cmd(cmd):
                        logger.log(log_level, f"{line}")
                except Exception as ex:
                    if not allow_errors:
                        raise
                    success = False
                    reason = textwrap.indent(str(ex), "  ")
                    logger.error(f"{cmd!r} failed with:\n{reason}")
        finally:
            if suppress_plotting and plot_on:
                tao.cmd("set global plot_on = T")
            if suppress_lattice_calc and lat_calc_on:
                tao.cmd("set global lattice_calc_on = T")

        return success

    @contextlib.contextmanager
    def set_context(self, tao: Tao):
        """
        Apply this configuration to Tao **only** for the given ``with`` block.

        Examples
        --------

        Set an initial value for a parameter:

        >>> new_state.param = 1
        >>> new_state.set()

        Then temporarily set it to another value, just for the `with` block:

        >>> new_state.param = 3
        >>> with new_state.set_context(tao):
        ...     assert new_state.query(tao).param == 3

        After the ``with`` block, ``param`` will be reset to its previous
        value:

        >>> assert new_state.query(tao).param == 1
        """
        pre_state = self.query(tao)
        for cmd in self.set_commands:
            tao.cmd(cmd)
        yield pre_state
        pre_state.set(tao)


T = TypeVar("T", bound=pydantic.BaseModel)
ArchiveFormat = Literal["yaml", "json.gz", "json", "hdf5"]


def format_from_filename(fn: pathlib.Path) -> ArchiveFormat:
    if fn.suffix.lower() in (".yml", ".yaml"):
        return "yaml"
    if fn.suffix.lower() in (".h5", ".hdf5"):
        return "hdf5"

    suffixes = [suffix.lower() for suffix in fn.suffixes][-2:]
    if suffixes == [".json", ".gz"]:
        return "json.gz"
    return "json"


def load_model_data(
    filename: str | pathlib.Path,
    *,
    format: ArchiveFormat | None = None,
):
    """
    Read the model from a file in JSON, YAML, or custom HDF5 format.

    For HDF5 format, the deserialized class instance will be returned.
    For JSON and YAML, the underlying model data will be returned.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the file where the model data should be written.
        The file format is determined by the extension.
    format : ArchiveFormat or None, optional
        File format.  If not specified, determined by file extension.
    """
    fname = pathlib.Path(filename)

    format = format or format_from_filename(fname)

    if format == "hdf5":
        with h5py.File(fname) as h5g:
            return restore_from_hdf5_file(h5g)

    if format == "yaml":
        import yaml  # NOTE: yaml is not a required dependency

        with open(fname, "rt") as fp:
            return yaml.safe_load(fp)
    elif format == "json.gz":
        with gzip.open(fname, "rb") as fp:
            return orjson.loads(fp.read())
    elif format == "json":
        return orjson.loads(fname.read_bytes())
    raise NotImplementedError(format)


def load_model(
    filename: str | pathlib.Path,
    cls: type[T],
    *,
    format: ArchiveFormat | None = None,
) -> T:
    """
    Read the model from a file in JSON, YAML, or custom HDF5 format.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the file where the model data should be written.
        The file format is determined by the extension.
    cls : pydantic.BaseModel class

    format : ArchiveFormat or None, optional
        File format.  If not specified, determined by file extension.
    """
    data = load_model_data(filename, format=format)
    if isinstance(data, pydantic.BaseModel):
        if not isinstance(data, cls):
            raise TypeError(
                f"Unexpected class returned from restore process. Expected {cls.__name__} "
                f"got {type(data).__name__}"
            )
        return data
    return cls.model_validate(data)


def date_coded_rename(
    dest: pathlib.Path, datefmt: str = "%Y%m%d_%H%M%S"
) -> pathlib.Path | None:
    """
    Rename a destination path, adding a date-coded name suffix.

    This is used in place of overwriting files, saving backups of the previous
    data.
    """
    if not dest.exists():
        return None

    dt = datetime.datetime.now().strftime(datefmt)
    backup_fn = f"{dest.stem}-{dt}{dest.suffix}"
    dest.rename(dest.with_name(backup_fn))
    return dest


def dump_model(
    filename: str | pathlib.Path,
    model: pydantic.BaseModel,
    *,
    exclude_defaults: bool = True,
    backup_existing: bool = True,
    datefmt: str = DEFAULT_DATEFMT,
    format: ArchiveFormat | None = None,
    indent: bool = False,
    sort_keys: bool = False,
):
    """
    Write the model data to a file in JSON, YAML, or custom HDF5 format.

    Parameters
    ----------
    filename : str or pathlib.Path
        The path to the file where the model data should be written.
        The file format is determined by the extension.
    model : pydantic.BaseModel

    exclude_defaults : bool, optional
        Exclude model defaults from the output. Defaults to True.
    backup_existing : bool, optional
        If a file exists at the target, rename it using `datefmt` as a backup.
        Defaults to True.
    datefmt : str, optional
        The date format for the backup file.
    format : ArchiveFormat or None, optional
        File format.  If not specified, determined by file extension.
    indent : bool, optional
        Indent the output, if applicable based on the file format.
    sort_keys : bool, optional
        Sort the output keys, if applicable based on the file format.
    """
    fname = pathlib.Path(filename)

    format = format or format_from_filename(fname)

    if backup_existing:
        date_coded_rename(fname, datefmt=datefmt)

    if format == "hdf5":
        with h5py.File(fname, "w") as h5g:
            store_in_hdf5_file(h5g, model)
        return

    data = model.model_dump(
        exclude_defaults=exclude_defaults,
        exclude_computed_fields=True,
        mode="json",
    )
    if format == "yaml":
        with fname.open("wt") as fp:
            import yaml  # NOTE: yaml is not a required dependency

            yaml.safe_dump(data, fp)
    elif format in ("json.gz", "json"):
        options = 0
        if indent:
            options |= orjson.OPT_INDENT_2
        if sort_keys:
            options |= orjson.OPT_SORT_KEYS
        dumped = orjson.dumps(data, option=options)

        if format == "json.gz":
            with gzip.open(fname, "wb") as fp:
                fp.write(dumped)
        else:
            pathlib.Path(fname).write_bytes(dumped)
    else:
        raise NotImplementedError(format)

    return data
