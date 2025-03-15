from .cli import main_python, main_ipython

try:
    import IPython  # noqa
except ImportError:
    main_python()
else:
    main_ipython()
