import datetime

import pytao

tao = pytao.Tao(
    init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall3d",
    noplot=True,
)
date = tao.version(as_date=True)
assert isinstance(date, datetime.datetime)
print(f"datetime.datetime({date.year}, {date.month}, {date.day})")
