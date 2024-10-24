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

import numpy as np
import pytest
import torch
from PIL import Image

from ziracle.model.blip2_model import Blip2Model, Blip2ModelFeatureExtractor

# ==============================================================================
# Test utils
# ==============================================================================
IMG_SHAPE = (1280, 720, 3)
IMG_BATCH_SIZE = 4
N_BLIP2_IMG_QUERIES = 32
EMB_SIZE_EXP = 768
DEVICE = "cpu"


@pytest.fixture(scope="session")
def blip2_model():
    return Blip2ModelFeatureExtractor(device=DEVICE)


@pytest.fixture
def image_batch():
    np.random.seed(42)
    img_list = []
    for _ in range(IMG_BATCH_SIZE):
        img_array = np.random.rand(*IMG_SHAPE) * 255
        img = Image.fromarray(img_array.astype("uint8")).convert("RGB")
        img_list.append(img)
    return img_list


@pytest.fixture
def sample_text():
    return "This is a sample text"


@pytest.fixture
def sample_text_n_tokens():
    return 7


# ==============================================================================
# Unit tests
# ==============================================================================
# ==============================================================================
# Unit tests
# ==============================================================================
def test_embed_image(image_batch: list[Image.Image], blip2_model: Blip2Model):
    img_in = image_batch[0]
    img_emb_ret = blip2_model.embed_image(imgdata=img_in)
    assert isinstance(img_emb_ret, torch.Tensor)
    assert img_emb_ret.shape == (1, N_BLIP2_IMG_QUERIES, EMB_SIZE_EXP)


def test_embed_image_batch(image_batch: list[Image.Image], blip2_model: Blip2Model):
    img_embs_ret = blip2_model.embed_image_batch(img_list=image_batch)
    assert isinstance(img_embs_ret, torch.Tensor)
    assert img_embs_ret.shape == (IMG_BATCH_SIZE, N_BLIP2_IMG_QUERIES, EMB_SIZE_EXP)


def test_embed_text(
    sample_text: str,
    sample_text_n_tokens: int,
    blip2_model: Blip2Model,
):
    txt_emb_ret = blip2_model.embed_text(txt=sample_text)
    assert isinstance(txt_emb_ret, torch.Tensor)
    assert txt_emb_ret.shape == (1, sample_text_n_tokens, EMB_SIZE_EXP)


def test_embed_multimodal(
    image_batch: list[Image.Image],
    sample_text: str,
    sample_text_n_tokens: int,
    blip2_model: Blip2Model,
):
    img_in = image_batch[0]
    txt_emb_ret = blip2_model.embed_multimodal(imgdata=img_in, txt=sample_text)
    shape_exp = (1, N_BLIP2_IMG_QUERIES + sample_text_n_tokens, EMB_SIZE_EXP)
    assert isinstance(txt_emb_ret, torch.Tensor)
    assert txt_emb_ret.shape == shape_exp
