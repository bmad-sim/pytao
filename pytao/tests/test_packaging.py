def test_version_set():
    from .. import __version__

    assert __version__ != "0.0.0+unknown"
