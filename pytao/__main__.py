from .cli import main_ipython, main_python

try:
    import IPython  # noqa
except ImportError:
    main_python()
else:
    main_ipython()
