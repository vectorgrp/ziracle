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

from __future__ import annotations

from typing import Literal

import torch
from lavis.models import load_model_and_preprocess
from PIL import Image


class Blip2Model:
    """Implements the embedding methods using BLIP2."""

    def __init__(self, device: Literal["cuda", "cpu"] = "cuda") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model, self.vis_proc, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="pretrain_opt2.7b",
            is_eval=True,
            device=self.device,
        )
        self._add_txt_weights()

    def _add_txt_weights(self):
        # Add txt weights back to the opt model
        model_feat, _, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=self.device,
        )
        self.model.Qformer.cls = model_feat.Qformer.cls
        self.model.Qformer.bert.embeddings.word_embeddings = (
            model_feat.Qformer.bert.embeddings.word_embeddings
        )
        self.model.Qformer.bert.embeddings.position_embeddings = (
            model_feat.Qformer.bert.embeddings.position_embeddings
        )
        for i, layer in enumerate(self.model.Qformer.bert.encoder.layer):
            layer.output = model_feat.Qformer.bert.encoder.layer[i].output
            layer.intermediate = model_feat.Qformer.bert.encoder.layer[i].intermediate

    # --------------------------------------------------------------------------
    # Embedding methods
    # --------------------------------------------------------------------------
    def embed_image(self, imgdata: Image.Image) -> torch.Tensor:
        """Embed a single image."""
        assert self.vis_proc is not None
        img_proc = self.vis_proc["eval"](imgdata).unsqueeze(0).to(self.device)
        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(self.model.visual_encoder(img_proc))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        emb = query_output.last_hidden_state.cpu().detach()
        return emb

    def embed_image_batch(self, img_list: list[Image.Image]) -> torch.Tensor:
        """Embed a batch of images."""
        assert self.vis_proc is not None
        img_proc_batch = torch.cat(
            [
                self.vis_proc["eval"](_img).unsqueeze(0).to(self.device)
                for _img in img_list
            ]
        )
        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(
                self.model.visual_encoder(img_proc_batch)
            )
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        emb = query_output.last_hidden_state.cpu().detach()
        return emb

    def embed_text(self, txt: str) -> torch.Tensor:
        """Embed a text"""
        text = self.model.tokenizer(
            txt,
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        query_output = self.model.Qformer.bert(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
        )
        emb = query_output.last_hidden_state.cpu().detach()
        return emb

    def embed_multimodal(self, imgdata: Image.Image, txt: str) -> torch.Tensor:
        """Create a BLIP2 multimodal embedding for the given image and text."""
        assert self.vis_proc is not None
        # Preprocess and embed image
        imgdata_proc = self.vis_proc["eval"](imgdata).unsqueeze(0).to(self.device)
        with self.model.maybe_autocast():
            image_embeds = self.model.ln_vision(self.model.visual_encoder(imgdata_proc))

        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text = self.model.tokenizer(
            txt,
            truncation=True,
            max_length=self.model.max_txt_len,
            return_tensors="pt",
        ).to(self.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
        query_output = self.model.Qformer.bert(
            text.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        emb = query_output.last_hidden_state.cpu().detach()
        return emb


class Blip2ModelFeatureExtractor(Blip2Model):
    """Integrates the feature extraction BLIP2 model variant."""

    def __init__(self, device: Literal["cuda", "cpu"] = "cuda") -> None:
        if device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model, self.vis_proc, _ = load_model_and_preprocess(
            name="blip2_feature_extractor",
            model_type="pretrain",
            is_eval=True,
            device=self.device,
        )


def get_blip2_model(model_name: Literal["opt", "feat"]) -> Blip2Model:
    """Get and initialize the BLIP2 model defined by the name.

    Parameters
    ----------
    model_name : Literal[&quot;opt&quot;, &quot;feat&quot;]
        BLIP2 model variant to load.
        `opt` loads the BLIP2 Q-former from the opt caption generation model.
        `feat` loads the designated feature extraction Q-former model.
    """
    match model_name:
        case "opt":
            model = Blip2Model()
        case "feat":
            model = Blip2ModelFeatureExtractor()
        case other:
            raise ValueError(f"Given model {other} is invalid!")
    return model
