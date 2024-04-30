import math
from collections import namedtuple

import torch

from modules import devices, sd_hijack_optimizations, shared, script_callbacks, errors, sd_unet, patches
from modules import prompt_parser, sd_hijack, sd_hijack_clip, sd_hijack_open_clip, sd_hijack_unet, sd_hijack_xlmr, xlmr, xlmr_m18, sd_emphasis
from modules.script_callbacks import on_ui_settings
from modules.sd_hijack import EmbeddingsWithFixes, model_hijack, apply_weighted_forward, undo_weighted_forward, weighted_forward
from modules.sd_hijack_clip import FrozenCLIPEmbedderWithCustomWords, FrozenCLIPEmbedderWithCustomWordsBase, PromptChunk
from modules.shared import opts

import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import ldm.modules.diffusionmodules.openaimodel
import ldm.models.diffusion.ddpm
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import ldm.modules.encoders.modules

import sgm.modules.attention
import sgm.modules.diffusionmodules.model
import sgm.modules.diffusionmodules.openaimodel
import sgm.modules.encoders.modules

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward

# new memory efficient cross attention blocks do not support hypernets and we already
# have memory efficient cross attention anyway, so this disables SD2.0's memory efficient cross attention
ldm.modules.attention.MemoryEfficientCrossAttention = ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = ldm.modules.attention.CrossAttention

# silence new console spam from SD2
ldm.modules.attention.print = shared.ldm_print
ldm.modules.diffusionmodules.model.print = shared.ldm_print
ldm.util.print = shared.ldm_print
ldm.models.diffusion.ddpm.print = shared.ldm_print

optimizers = []
current_optimizer: sd_hijack_optimizations.SdOptimization = None

ldm_patched_forward = sd_unet.create_unet_forward(ldm.modules.diffusionmodules.openaimodel.UNetModel.forward)
ldm_original_forward = patches.patch(__file__, ldm.modules.diffusionmodules.openaimodel.UNetModel, "forward", ldm_patched_forward)

sgm_patched_forward = sd_unet.create_unet_forward(sgm.modules.diffusionmodules.openaimodel.UNetModel.forward)
sgm_original_forward = patches.patch(__file__, sgm.modules.diffusionmodules.openaimodel.UNetModel, "forward", sgm_patched_forward)


OPT_NAME = "enable_xl_clip_skip"
       
def ext_on_ui_settings():
    # [setting_name], [default], [label], [component(blank is checkbox)], [component_args]debug_level_choices = []
    xlclipskip_options = [
        (OPT_NAME, False, "Enable the option to set clip skip setting for the small clip model in SDXL")
    ]
    section = ('Enable clip-skip in SDXL', "Enable clip-skip in SDXL")

    for cur_setting_name, *option_info in xlclipskip_options:
        shared.opts.add_option(cur_setting_name, shared.OptionInfo(*option_info, section=section))

on_ui_settings(ext_on_ui_settings)

def hijack(self, m):
    conditioner = getattr(m, 'conditioner', None)
    if conditioner:
        text_cond_models = []

        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            typename = type(embedder).__name__
            if typename == 'FrozenOpenCLIPEmbedder':
                embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self)
                conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(embedder, self)
                text_cond_models.append(conditioner.embedders[i])
            if typename == 'FrozenCLIPEmbedder':
                model_embeddings = embedder.transformer.text_model.embeddings
                model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
                conditioner.embedders[i] = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(embedder, self)
                text_cond_models.append(conditioner.embedders[i])
            if typename == 'FrozenOpenCLIPEmbedder2':
                embedder.model.token_embedding = EmbeddingsWithFixes(embedder.model.token_embedding, self, textual_inversion_key='clip_g')
                conditioner.embedders[i] = sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords(embedder, self)
                text_cond_models.append(conditioner.embedders[i])

        if len(text_cond_models) == 1:
            m.cond_stage_model = text_cond_models[0]
        else:
            m.cond_stage_model = conditioner

    if type(m.cond_stage_model) == xlmr.BertSeriesModelWithTransformation or type(m.cond_stage_model) == xlmr_m18.BertSeriesModelWithTransformation:
        model_embeddings = m.cond_stage_model.roberta.embeddings
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
        m.cond_stage_model = sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

    elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
        m.cond_stage_model = sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

    elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
        m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
        m.cond_stage_model = sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

    apply_weighted_forward(m)
    if m.cond_stage_key == "edit":
        sd_hijack_unet.hijack_ddpm_edit()

    self.apply_optimizations()

    self.clip = m.cond_stage_model

    def flatten(el):
        flattened = [flatten(children) for children in el.children()]
        res = [el]
        for c in flattened:
            res += c
        return res

    self.layers = flatten(m)

    import modules.models.diffusion.ddpm_edit

    if isinstance(m, ldm.models.diffusion.ddpm.LatentDiffusion):
        sd_unet.original_forward = ldm_original_forward
    elif isinstance(m, modules.models.diffusion.ddpm_edit.LatentDiffusion):
        sd_unet.original_forward = ldm_original_forward
    elif isinstance(m, sgm.models.diffusion.DiffusionEngine):
        sd_unet.original_forward = sgm_original_forward
    else:
        sd_unet.original_forward = None

def undo_hijack(self, m):
    conditioner = getattr(m, 'conditioner', None)
    if conditioner:
        for i in range(len(conditioner.embedders)):
            embedder = conditioner.embedders[i]
            if isinstance(embedder, (sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords, sd_hijack_open_clip.FrozenOpenCLIPEmbedder2WithCustomWords)):
                embedder.wrapped.model.token_embedding = embedder.wrapped.model.token_embedding.wrapped
                conditioner.embedders[i] = embedder.wrapped
            if isinstance(embedder, sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords):
                embedder.wrapped.transformer.text_model.embeddings.token_embedding = embedder.wrapped.transformer.text_model.embeddings.token_embedding.wrapped
                conditioner.embedders[i] = embedder.wrapped

        if hasattr(m, 'cond_stage_model'):
            delattr(m, 'cond_stage_model')

    elif type(m.cond_stage_model) == sd_hijack_xlmr.FrozenXLMREmbedderWithCustomWords:
        m.cond_stage_model = m.cond_stage_model.wrapped

    elif type(m.cond_stage_model) == sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords:
        m.cond_stage_model = m.cond_stage_model.wrapped

        model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
        if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
            model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
    elif type(m.cond_stage_model) == sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords:
        m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
        m.cond_stage_model = m.cond_stage_model.wrapped

    undo_optimizations()
    undo_weighted_forward(m)

    self.apply_circular(False)
    self.layers = None
    self.clip = None

class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        elif opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        return z

sd_hijack_clip.FrozenCLIPEmbedderForSDXLWithCustomWords.encode_with_transformers = encode_with_transformers
sd_hijack.StableDiffusionModelHijack.hijack = hijack
sd_hijack.StableDiffusionModelHijack.undo_hijack = undo_hijack
