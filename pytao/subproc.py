import ctypes
import logging
import os
import pickle
import subprocess
import sys
import threading
import traceback
import typing

import numpy as np

from .interface_commands import Tao

logger = logging.getLogger(__name__)


class TaoDisconnectedError(Exception):
    pass


def read_length_from_pipe(pipe) -> int:
    length_bytes = pipe.read(ctypes.sizeof(ctypes.c_uint32))
    if not length_bytes:
        logger.debug("<- disconnected")
        raise TaoDisconnectedError()
    length = ctypes.c_uint32.from_buffer(bytearray(length_bytes))
    return length.value


def read_pickled_data(pipe) -> dict:
    length = read_length_from_pipe(pipe)
    buffer = pipe.read(length)
    if len(buffer) != length:
        logger.debug("<- disconnected")
        raise TaoDisconnectedError()
    data = pickle.loads(buffer)
    res = data.get("result", None)
    if isinstance(res, dict) and "__type__" in res:
        data["result"] = dict_to_array(res)
    logger.debug(f"<- {data}")
    return data


def write_pickled_data(pipe, data):
    logger.debug(f"-> {data}")
    message_bytes = pickle.dumps(data) + b"\n"
    to_write = bytes(ctypes.c_uint32(len(message_bytes))) + message_bytes
    if isinstance(pipe, int):
        os.write(pipe, to_write)
    else:
        pipe.write(to_write)
        pipe.flush()


class SerializedArray(typing.TypedDict):
    __type__: typing.Literal["array"]
    shape: typing.Tuple[int, ...]
    dtype: np.dtype
    data: bytes


def array_to_dict(arr: np.ndarray) -> SerializedArray:
    return {
        "__type__": "array",
        "shape": arr.shape,
        "dtype": arr.dtype,
        "data": arr.tobytes(),
    }


def dict_to_array(data: SerializedArray) -> np.ndarray:
    assert isinstance(data, dict)
    assert data["__type__"] == "array"
    arr = np.frombuffer(bytearray(data["data"]), dtype=data["dtype"])
    return arr.reshape(data["shape"])


def _get_result(value, raises: bool = True):
    if not isinstance(value, dict):
        raise ValueError(f"Unexpected result type: {type(value)}")
    if "result" in value:
        return value["result"]
    if "error" in value:
        error = value.get("error")
        error_cls = value.get("error_cls")
        tb = value.get("traceback", "")
        if raises:
            ex = RuntimeError(f"Tao in subprocess raised {error_cls}: {error}")
            if hasattr(ex, "add_note") and callable(ex.add_note):
                ex.add_note(f"Subprocess {tb}")
            raise ex
        logger.error(f"Tao in subprocess raised an error {error_cls}:\n{tb}")


class _TaoPipe:
    def __init__(self):
        self._subproc = None
        self._read_pipe = None
        self._monitor_thread = None
        self._subproc_in_fd = None
        self._subproc_out_fd = None

    def _tao_subprocess_monitor(self, subproc: subprocess.Popen):
        if subproc is None:
            return
        in_fd = self._subproc_in_fd
        out_fd = self._subproc_out_fd
        try:
            code = subproc.wait()
            if code != 0:
                logger.warning(f"Subprocess exited with error code {code}")
            else:
                logger.debug("Subprocess exited without error")
            if in_fd is not None:
                os.close(in_fd)
            if out_fd is not None:
                os.close(out_fd)
        finally:
            self._subproc = None
            self._read_pipe = None
            self._subproc_in_fd = None
            self._subproc_out_fd = None
            self._monitor_thread = None

    def _send(self, cmd: str, argument: str):
        if self._subproc is None:
            self._subproc = self._init_subprocess()

        assert self._subproc.stdin is not None
        return write_pickled_data(
            self._subproc.stdin,
            {"command": cmd, "args": [argument]},
        )

    def _receive(self):
        return read_pickled_data(self._read_pipe)

    def _init_subprocess(self) -> subprocess.Popen:
        logger.debug("Initializing Tao subprocess")
        assert self._subproc is None
        self._subproc_in_fd, self._subproc_out_fd = os.pipe()
        subproc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pytao.subproc",
                str(self._subproc_out_fd),
            ],
            stdin=subprocess.PIPE,
            pass_fds=[self._subproc_out_fd],
        )
        self._read_pipe = os.fdopen(self._subproc_in_fd, "rb")
        assert subproc.stdin is not None
        self._monitor_thread = threading.Thread(
            target=self._tao_subprocess_monitor,
            args=(subproc,),
            daemon=True,
        )
        self._monitor_thread.start()
        return subproc

    def send_receive(self, cmd: str, argument: str, raises: bool):
        self._send(cmd, argument)
        return _get_result(self._receive(), raises=raises)


class SubprocessTao(Tao):
    def __init__(self, *args, **kwargs):
        self._pipe = _TaoPipe()
        super().__init__(*args, **kwargs)

    def init(self, cmd):
        return self._pipe.send_receive("init", cmd, raises=True)

    def cmd(self, cmd, raises=True):
        return self._pipe.send_receive("cmd", cmd, raises=raises)

    def cmd_real(self, cmd, raises=True):
        return self._pipe.send_receive("cmd_real", cmd, raises=raises)

    def cmd_integer(self, cmd, raises=True):
        return self._pipe.send_receive("cmd_integer", cmd, raises=raises)


def _tao_subprocess(output_fd: int) -> None:
    logger.debug("Tao subprocess handler started")

    tao = Tao()

    with os.fdopen(output_fd, "wb") as output_pipe:
        while True:
            message = read_pickled_data(sys.stdin.buffer)
            try:
                command = message["command"]
                if command == "init":
                    output = tao.init(*message["args"])
                elif command == "cmd":
                    output = tao.cmd(*message["args"])
                elif command == "cmd_real":
                    output = tao.cmd_real(*message["args"])
                elif command == "cmd_integer":
                    output = tao.cmd_integer(*message["args"])
                elif command == "quit":
                    sys.exit(0)
                else:
                    output = "unknown command"
            except Exception as ex:
                write_pickled_data(
                    output_pipe,
                    {
                        "error": str(ex),
                        "error_cls": type(ex).__name__,
                        "traceback": traceback.format_exc(),
                    },
                )
            else:
                if isinstance(output, np.ndarray):
                    output = array_to_dict(output)

                write_pickled_data(output_pipe, {"result": output})


if __name__ == "__main__":
    try:
        output_fd = int(sys.argv[1])
    except (IndexError, ValueError):
        print(
            f"Usage: {sys.executable} {__file__} (output_file_descriptor)",
            file=sys.stderr,
        )
        exit(1)
    _tao_subprocess(output_fd)
