import os
import time

import numpy as np
import pytest

from .. import SubprocessTao
from ..subproc import TaoDisconnectedError


def test_crash_and_recovery() -> None:
    init = os.path.expandvars(
        os.path.expandvars(
            "-init $ACC_ROOT_DIR/regression_tests/python_test/csr_beam_tracking/tao.init -noplot"
        )
    )
    tao = SubprocessTao(init=init)
    # tao.init("-init regression_tests/python_test/tao.init_plot_line -external_plotting")
    bunch1 = tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")
    print("bunch1=", bunch1)

    with pytest.raises(TaoDisconnectedError):
        # Close the pipe earlier than expected
        tao._pipe.send_receive("quit", "", raises=True)
    time.sleep(0.5)

    print("Re-initializing:")
    tao.init(init)
    retry = tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")
    assert np.allclose(bunch1, retry)
