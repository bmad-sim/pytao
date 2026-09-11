def assert_result_fields(result, fields: dict) -> None:
    """
    Assert per-field CheckResult state on a constraint result object.

    Parameters
    ----------
    result : ConstraintResult
        The result object whose fields are checked.
    fields : dict
        Mapping of field name to expected state: None (not run), True (passed),
        or False (failed with non-empty detail).
    """
    for field, expected in fields.items():
        value = getattr(result, field)
        if expected is None:
            assert value is None, f"{field} expected None, got {value!r}"
        elif expected is True:
            assert value is not None, f"{field} expected CheckResult, got None"
            assert value.passed, f"{field}.passed expected True"
        else:
            assert value is not None, f"{field} expected CheckResult, got None"
            assert not value.passed, f"{field}.passed expected False"
            assert value.detail, f"{field}.detail expected non-empty on failure"
