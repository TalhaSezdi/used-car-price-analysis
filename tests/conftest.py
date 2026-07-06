import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_all_figures():
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
