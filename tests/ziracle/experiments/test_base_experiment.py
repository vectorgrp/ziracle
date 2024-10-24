# Copyright (C) 2024  Vector Informatik GmbH.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from unittest.mock import MagicMock

import numpy as np
import pytest

import ziracle.experiments.base_experiment
from ziracle.data import DocId, Query
from ziracle.experiments.base_experiment import ExpBlip2Base

# ==============================================================================
# Test utils
# ==============================================================================
EMB_SHAPE = (32, 768)


@pytest.fixture
def mock_emb_dict():
    np.random.seed(42)
    images = ["42.jpg", "0.jpg", "1.jpg", "2.jpg", "3.jpg"]
    emb_dict = {img: np.random.rand(*EMB_SHAPE) for img in images}
    return emb_dict


@pytest.fixture
def exp_obj(monkeypatch: pytest.MonkeyPatch, mock_emb_dict: dict[str, np.ndarray]):
    monkeypatch.setattr(
        ziracle.experiments.base_experiment.ExpBlip2Base,
        "_load_experiment_data",
        lambda _: None,
    )

    def mock_start_mlflow(self, experiment_name):
        pass

    monkeypatch.setattr(
        ziracle.experiments.base_experiment.ExpBlip2Base,
        "_start_mlflow_run",
        mock_start_mlflow,
    )
    monkeypatch.setattr(ziracle.experiments.base_experiment, "mlflow", MagicMock())
    exp_obj = ExpBlip2Base(focus_coir_path="", emb_path="", run_name="", device="cpu")
    exp_obj.emb_dict = mock_emb_dict
    exp_obj.step = 1
    return exp_obj


@pytest.fixture
def query_in():
    return Query(
        id=42,
        query_img="42.jpg",
        labels=[("0.jpg", 1), ("1.jpg", 1), ("2.jpg", 0), ("3.jpg", 0)],
        name_text="Textual name of the query",
        desc_text="Textual description of the query",
    )


# ==============================================================================
# Unit tests
# ==============================================================================
def test_get_similarity_scores(exp_obj: ExpBlip2Base, query_in: Query):
    scores_ret = exp_obj._get_similarity_scores(query_in)
    assert isinstance(scores_ret, list)
    assert len(scores_ret) == len(query_in.labels)
    assert all(isinstance(docid, DocId) for docid, _ in scores_ret)
    assert all(isinstance(score, float) for _, score in scores_ret)


def test_evaluate_scores(exp_obj: ExpBlip2Base, query_in: Query):
    scores_in: list[tuple[DocId, float]] = [
        ("0.jpg", 0.9),
        ("1.jpg", 0.8),
        ("2.jpg", 0.2),
        ("3.jpg", 0.1),
    ]
    metrics_ret = exp_obj._evaluate_scores(query=query_in, scores=scores_in)
    assert len(metrics_ret) == len(exp_obj.metrics)
    assert metrics_ret["avg_prec"] == 1


def test_run_query(exp_obj: ExpBlip2Base, query_in: Query):
    metrics_ret = exp_obj._run_query(query=query_in)
    assert len(metrics_ret) == len(exp_obj.metrics)
