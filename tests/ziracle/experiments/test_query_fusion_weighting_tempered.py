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
import ziracle.experiments.query_fusion_weighting_tempered
from ziracle.data import DocId, Query
from ziracle.experiments.query_fusion_weighting_tempered import (
    ExpBlip2QueryFusionWeighting,
)

# ==============================================================================
# Test utils
# ==============================================================================
EMB_SHAPE = (32, 768)


@pytest.fixture
def mock_emb_dict():
    np.random.seed(42)
    images = ["42.jpg", "0.jpg", "1.jpg", "2.jpg", "3.jpg"]
    emb_dict = {img: np.random.rand(*EMB_SHAPE).astype(np.float32) for img in images}
    return emb_dict


@pytest.fixture
def exp_obj(monkeypatch: pytest.MonkeyPatch, mock_emb_dict: dict[str, np.ndarray]):
    monkeypatch.setattr(
        ziracle.experiments.query_fusion_weighting_tempered.ExpBlip2BaseModel,
        "_load_experiment_data",
        lambda _: None,
    )

    def mock_start_mlflow(self, experiment_name):
        pass

    monkeypatch.setattr(
        ziracle.experiments.query_fusion_weighting_tempered.ExpBlip2BaseModel,
        "_start_mlflow_run",
        mock_start_mlflow,
    )
    monkeypatch.setattr(ziracle.experiments.base_experiment, "mlflow", MagicMock())
    monkeypatch.setattr(
        ziracle.experiments.query_fusion_weighting_tempered, "mlflow", MagicMock()
    )
    exp_obj = ExpBlip2QueryFusionWeighting(
        model_name="feat",
        target_text="name",
        target_token="full",
        focus_coir_path="",
        emb_path="",
        run_name="",
        device="cpu",
    )
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
def test_get_similarity_scores(exp_obj: ExpBlip2QueryFusionWeighting, query_in: Query):
    scores_ret = exp_obj._get_similarity_scores(query_in)
    assert isinstance(scores_ret, list)
    assert len(scores_ret) == len(query_in.labels)
    assert all(isinstance(docid, DocId) for docid, _ in scores_ret)
    assert all(isinstance(score, float) for _, score in scores_ret)
