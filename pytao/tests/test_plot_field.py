from .conftest import BackendName, get_example, test_artifacts


def test_plot_field(plot_backend: BackendName):
    example = get_example("cbeta_cell")
    example.plot = plot_backend
    with example.run_context(use_subprocess=True) as tao:
        tao.plot_field(
            "FF.QUA01#1",
            save=test_artifacts / f"test_plot_field-{plot_backend}",
        )
