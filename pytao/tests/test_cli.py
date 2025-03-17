import code
import os
import pathlib
import pytest
import sys
from unittest.mock import MagicMock, Mock, patch

from ..cli import (
    PytaoArgs,
    init,
    main_ipython,
    main_python,
    split_pytao_tao_args,
)
from ..tao_ctypes.core import register_input_transformer


def test_split_args_basic():
    args = ["--pyplot", "mpl", "tao_command", "-args"]
    pytao_args, tao_args = split_pytao_tao_args(args)

    assert pytao_args.pyplot == "mpl"
    assert tao_args == "tao_command -args"


def test_split_args_common():
    args = ["--pyplot", "mpl", "-noplot", "-args", "-lat", "latfile"]
    pytao_args, tao_args = split_pytao_tao_args(args)

    assert pytao_args.pyplot == "mpl"
    assert tao_args == "-noplot -args -lat latfile"


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
        "tao_command",
    ]
    pytao_args, tao_args = split_pytao_tao_args(args)

    assert pytao_args.pyplot == "bokeh"
    assert pytao_args.pyscript == "script.py"
    assert pytao_args.pycommand == "print('hello')"
    assert pytao_args.pylog == "DEBUG"
    assert pytao_args.pysubprocess is True
    assert tao_args == "tao_command"


@patch("pytao.cli.SubprocessTao")
@patch("pytao.cli.Tao")
def test_init_regular_tao(mock_tao, mock_subprocess_tao):
    """Test initialization with regular Tao"""
    mock_instance = MagicMock()
    mock_tao.return_value = mock_instance
    with patch.object(sys, "argv", ["pytao", "--pyplot", "mpl", "--pyno-subprocess"]):
        python_args, user_ns = init(ipython=False)

        assert python_args.pyplot == "mpl"
        assert "tao" in user_ns
        assert user_ns["tao"] == mock_instance
        assert "plt" in user_ns
        mock_tao.assert_called_once()
        mock_subprocess_tao.assert_not_called()


@patch("pytao.cli.SubprocessTao")
@patch("pytao.cli.Tao")
def test_init_subprocess_tao(mock_tao, mock_subprocess_tao):
    """Test initialization with Subprocess Tao"""
    mock_instance = MagicMock()
    mock_subprocess_tao.return_value = mock_instance

    with patch.object(sys, "argv", ["pytao"]):
        python_args, user_ns = init(ipython=True)

        assert python_args.pysubprocess is True
        assert "tao" in user_ns
        assert user_ns["tao"] == mock_instance
        mock_subprocess_tao.assert_called_once()
        mock_tao.assert_not_called()


@patch("pytao.cli.SubprocessTao")
@patch.dict(os.environ, {"PYTAO_PLOT": "bokeh"})
def test_init_env_plot_backend(mock_tao):
    """Test plot backend from environment variable"""
    with patch.object(sys, "argv", ["pytao"]):
        mock_tao.return_value = MagicMock()

        init(ipython=False)

        mock_tao.assert_called_with(init="", plot="bokeh")


@patch("pytao.cli.SubprocessTao")
@patch("logging.basicConfig")
def test_init_logging(mock_logging, mock_tao):
    """Test logging configuration"""
    mock_tao.return_value = MagicMock()

    with patch.object(sys, "argv", ["pytao", "--pylog", "DEBUG"]):
        python_args, _ = init(ipython=False)

        assert python_args.pylog == "DEBUG"
        mock_logging.assert_called_once()


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
    with open(fn, "wt") as fp:
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
