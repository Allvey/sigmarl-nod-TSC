from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utilities.evaluation_base import Evaluation


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("num_methods", [3, 4])
def test_method_labels_and_reference_match_distribution_positions(monkeypatch, collision, num_methods):
    labels = ["NOD-DGPPO", "SigmaRL", "MAPPO", "XP-MARL"]
    paths = ["/outputs/nod/", "/checkpoints/M0 (our)/", "/checkpoints/mappo/", "/checkpoints/xp/"]
    if num_methods == 3:
        labels.pop(2)
        paths.pop(2)
    evaluator = Evaluation(
        model_paths=paths,
        x_ticks=labels,
        fitst_model_index=0, x_tick_label_rotation=15,
    )
    assert evaluator.idx_our == 0  # A baseline's folder name cannot choose the reference.
    evaluator.parameters = SimpleNamespace(is_save_eval_results=True)
    figures = []
    monkeypatch.setattr(evaluator, "_save_metric_figure", lambda fig, name: figures.append(fig))
    data = torch.tensor([[1., 2., 3.], [10., 11., 12.], [20., 21., 22.], [30., 31., 32.]])[:num_methods]
    if collision:
        evaluator._plot_collision_breakdown(
            data, data * .4, data * .6,
            y_label="Collision", fig_name="collision_rate", y_limits=[0, None],
        )
    else:
        evaluator._boxplot(
            figsize=(3.5, 2), data=data, y_label="Speed",
            fig_name="average_speed", y_limits=[0, None],
        )
    ax = figures[0].axes[0]
    positions = np.arange(1, num_methods + 1)
    np.testing.assert_array_equal(ax.get_xticks(), positions)
    assert [label.get_text() for label in ax.get_xticklabels()] == evaluator.x_ticks
    assert ax.get_xlim() == (.5, num_methods + .5)
    reference = next(text for text in ax.texts if text.get_text().startswith("Our:"))
    assert reference.get_position()[0] == 1
    assert "2.00" in reference.get_text()
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2, 2])
    # Matplotlib's median segments identify each plotted distribution exactly.
    medians = [collection for collection in ax.collections
               if hasattr(collection, "get_segments") and len(collection.get_segments()) == num_methods
               and all(len(segment) == 2 and segment[0, 1] == segment[1, 1]
                       for segment in collection.get_segments())]
    expected = positions - .2 if collision else positions
    assert any(np.allclose([segment[:, 0].mean() for segment in collection.get_segments()], expected)
               and np.allclose([segment[0, 1] for segment in collection.get_segments()], [2, 11, 21, 31][:num_methods])
               for collection in medians)
