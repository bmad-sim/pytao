import code
import contextlib
import os
import pathlib
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from ..cli import (
    PytaoArgs,
    init,
    main_ipython,
    main_python,
)
from ..core import register_input_transformer


def test_split_args_basic():
    args = PytaoArgs.from_cli_args(["--pyplot", "mpl", "-init", "init.foo", "-noplot"])
    assert args.pyplot == "mpl"
    assert args.init_file == "init.foo"
    assert args.noplot


def test_split_args_common():
    args = PytaoArgs.from_cli_args(["--pyplot", "mpl", "-noplot", "-lat", "latfile"])

    assert args.pyplot == "mpl"
    assert args.lattice_file == "latfile"
    assert args.noplot


def test_split_args_all_options():
    args = [
        "--pyplot",
        "bokeh",
        "--pyscript",
        "script.py",
        "--pycommand",
        "print('hello')",
        "--pylog",
        "DEBUG",
        "-init",
        "fooinit",
    ]
    args = PytaoArgs.from_cli_args(args)

    assert args.pyplot == "bokeh"
    assert args.pyscript == "script.py"
    assert args.pycommand == "print('hello')"
    assert args.pylog == "DEBUG"
    assert args.pysubprocess is True
    assert args.init_file == "fooinit"


@pytest.fixture
def mock_tao_startup() -> Generator[tuple[MagicMock, dict[str, Any]], None, None]:
    """
    Fixture that patches TaoStartup.run and TaoStartup.run_context.

    Returns
    -------
    Tuple[MagicMock, dict[str, Any]]
        A tuple containing the mock Tao instance, and a dictionary that captures
        the populated `TaoStartup` instance and the arguments passed to the methods.
    """
    mock_tao = MagicMock()
    call_info: dict[str, Any] = {}

    def fake_run(self, use_subprocess: bool = False) -> MagicMock:
        call_info["self"] = self
        call_info["use_subprocess"] = use_subprocess
        call_info["method"] = "run"
        return mock_tao

    @contextlib.contextmanager
    def fake_run_context(self, use_subprocess: bool = False):
        call_info["self"] = self
        call_info["use_subprocess"] = use_subprocess
        call_info["method"] = "run_context"
        yield mock_tao

    with (
        patch("pytao.startup.TaoStartup.run", autospec=True, side_effect=fake_run),
        patch(
            "pytao.startup.TaoStartup.run_context", autospec=True, side_effect=fake_run_context
        ),
    ):
        yield mock_tao, call_info


def test_init_regular_tao(mock_tao_startup):
    """Test initialization with regular Tao"""
    mock_instance, call_info = mock_tao_startup

    python_args, user_ns = init(
        ["pytao", "--pyplot", "mpl", "--pyno-subprocess"], ipython=False
    )

    assert python_args.pyplot == "mpl"
    assert "tao" in user_ns
    assert user_ns["tao"] == mock_instance
    assert "plt" in user_ns

    # Assert TaoStartup execution details
    assert "self" in call_info, "TaoStartup.run was never called!"
    assert call_info["use_subprocess"] is False, "Expected regular Tao, not SubprocessTao"


def test_init_subprocess_tao(mock_tao_startup):
    """Test initialization with Subprocess Tao"""
    mock_instance, call_info = mock_tao_startup

    python_args, user_ns = init(["pytao"], ipython=True)

    assert python_args.pysubprocess is True
    assert "tao" in user_ns
    assert user_ns["tao"] == mock_instance

    # Assert TaoStartup execution details
    assert "self" in call_info, "TaoStartup.run was never called!"
    assert call_info["use_subprocess"] is True, "Expected SubprocessTao to be requested"


@patch.dict(os.environ, {"PYTAO_PLOT": "bokeh"})
def test_init_env_plot_backend(mock_tao_startup):
    """Test plot backend from environment variable"""
    mock_instance, call_info = mock_tao_startup

    init(["pytao"], ipython=False)

    assert "self" in call_info, "TaoStartup.run was never called!"
    startup_instance = call_info["self"]

    params = startup_instance.tao_class_params
    assert startup_instance.pyplot == "bokeh"
    assert params["init_file"] == "tao.init"  # default


@patch("logging.basicConfig")
def test_init_logging(mock_logging, mock_tao_startup):
    """Test logging configuration"""
    mock_instance, call_info = mock_tao_startup

    python_args, _ = init(["pytao", "--pylog", "DEBUG"], ipython=False)

    assert python_args.pylog == "DEBUG"
    mock_logging.assert_called_once()
    assert "self" in call_info, "TaoStartup.run was never called!"


@patch("pytao.cli.init")
@patch("code.InteractiveConsole")
def test_main_python_interactive(mock_console, mock_init):
    """Test Python backend with interactive console"""
    python_args = PytaoArgs(pysubprocess=False)
    mock_init.return_value = (python_args, {"tao": MagicMock()})

    console_instance = MagicMock()
    mock_console.return_value = console_instance

    main_python()

    mock_console.assert_called_once()
    console_instance.interact.assert_called_once()


@patch("code.InteractiveConsole")
@patch("pytao.cli.init")
@patch("builtins.exec")
def test_main_python_command(mock_exec, mock_init, mock_console):
    """Test Python backend with command execution"""
    python_args = PytaoArgs(pycommand="print('hello')", pysubprocess=False)
    user_ns = {"tao": MagicMock()}
    mock_init.return_value = (python_args, user_ns)

    main_python()

    mock_exec.assert_called_with("print('hello')", user_ns)
    mock_console.assert_called_once()


def test_main_python_script(tmp_path: pathlib.Path):
    """Test Python backend with script execution"""

    fn = tmp_path / "test.py"
    with open(fn, "w") as fp:
        print("print('script')", file=fp)

    with patch.object(code, "InteractiveConsole", Mock()):
        with patch.object(
            sys,
            "argv",
            [
                "pytao",
                "--pyscript",
                str(fn),
                "-noplot",
                "-lat",
                "$ACC_ROOT_DIR/bmad-doc/tao_examples/fodo/fodo.bmad",
            ],
        ):
            main_python()


@patch("pytao.cli.init")
@patch("IPython.start_ipython")
def test_main_ipython_basic(mock_start_ipython, mock_init):
    """Test basic IPython backend"""
    python_args = PytaoArgs(pysubprocess=False, pyprefix="<")
    user_ns = {"tao": MagicMock()}
    mock_init.return_value = (python_args, user_ns)

    main_ipython()

    mock_start_ipython.assert_called_with(
        config={
            "InteractiveShellApp": {
                "exec_lines": [
                    "tao.register_cell_magic()",
                    "tao.register_input_transformer('<')",
                ]
            }
        },
        user_ns=user_ns,
        argv=["--no-banner", "-i"],
    )


@patch("pytao.cli.init")
@patch("IPython.start_ipython")
def test_main_ipython_command(mock_start_ipython, mock_init):
    """Test IPython backend with command"""
    python_args = PytaoArgs(
        pycommand="print('hello')",
        pysubprocess=False,
    )
    user_ns = {"tao": MagicMock()}
    mock_init.return_value = (python_args, user_ns)

    main_ipython()

    mock_start_ipython.assert_called_with(
        config={
            "InteractiveShellApp": {
                "exec_lines": [
                    "tao.register_cell_magic()",
                    "tao.register_input_transformer('`')",
                ]
            }
        },
        user_ns=user_ns,
        argv=["--no-banner", "-i", "-c", "print('hello')"],
    )


@patch("pytao.cli.init")
@patch("IPython.start_ipython")
def test_main_ipython_script_no_interactive(mock_start_ipython, mock_init):
    """Test IPython backend with script"""
    python_args = PytaoArgs(
        pyscript="script.py",
        pysubprocess=False,
        pyinteractive=False,
    )
    user_ns = {"tao": MagicMock()}
    mock_init.return_value = (python_args, user_ns)

    main_ipython()

    mock_start_ipython.assert_called_with(
        config={
            "InteractiveShellApp": {
                "exec_lines": [
                    "tao.register_cell_magic()",
                    "tao.register_input_transformer('`')",
                ]
            }
        },
        user_ns=user_ns,
        argv=["--no-banner", "script.py"],
    )


@patch("pytao.cli.init")
@patch("IPython.start_ipython")
def test_main_ipython_script(mock_start_ipython, mock_init):
    """Test IPython backend with script"""
    python_args = PytaoArgs(
        pyscript="script.py",
        pysubprocess=False,
        pyinteractive=True,
    )
    user_ns = {"tao": MagicMock()}
    mock_init.return_value = (python_args, user_ns)

    main_ipython()

    mock_start_ipython.assert_called_with(
        config={
            "InteractiveShellApp": {
                "exec_lines": [
                    "tao.register_cell_magic()",
                    "tao.register_input_transformer('`')",
                ]
            }
        },
        user_ns=user_ns,
        argv=["--no-banner", "-i", "script.py"],
    )


def test_register_input_transformer():
    """Test the register_input_transformer function."""

    # Create a mock IPython instance
    mock_ipython = Mock()
    mock_ipython.input_transformers_post = []

    # Patch get_ipython to return our mock
    with patch("IPython.get_ipython", return_value=mock_ipython):
        # Call the function we're testing
        register_input_transformer(prefix="`")

        assert len(mock_ipython.input_transformers_post) == 1
        transformer = mock_ipython.input_transformers_post[0]

        input_lines = [
            "`show lat -element=quad::*",
            'print("Hello world")',
            "`python show_info()",
        ]

        transformed_lines = transformer(input_lines)
        assert transformed_lines == [
            "print(\"\\n\".join(tao.cmd('show lat -element=quad::*')))",
            'print("Hello world")',
            "print(\"\\n\".join(tao.cmd('python show_info()')))",
        ]


def test_register_input_transformer_no_ipython():
    """Test when IPython is not available."""

    with patch("IPython.get_ipython", return_value=None):
        result = register_input_transformer(prefix="`")
        assert result is None


def test_register_input_transformer_custom_prefix():
    """Test with a custom prefix."""

    mock_ipython = Mock()
    mock_ipython.input_transformers_post = []

    with patch("IPython.get_ipython", return_value=mock_ipython):
        register_input_transformer(prefix="<")

        assert len(mock_ipython.input_transformers_post) == 1
        transformer = mock_ipython.input_transformers_post[0]

        input_lines = [
            "<show lat -element=quad::*",
            'print("Hello world")',
            "<python show_info()",
        ]

        transformed_lines = transformer(input_lines)

        assert transformed_lines == [
            "print(\"\\n\".join(tao.cmd('show lat -element=quad::*')))",
            'print("Hello world")',
            "print(\"\\n\".join(tao.cmd('python show_info()')))",
        ]


def test_import_error_handling():
    with patch.dict(sys.modules, {"IPython": None}):
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'IPython'")
        ):
            with pytest.raises(ImportError):
                register_input_transformer(prefix="`")
