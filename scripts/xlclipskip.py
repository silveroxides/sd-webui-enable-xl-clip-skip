import torch
from modules import shared, devices, sd_hijack_clip
from modules.script_callbacks import on_ui_settings

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

class FrozenCLIPEmbedderForSDXLWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

    def encode_with_transformers(self, tokens):
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=self.wrapped.layer == "hidden")

        if self.wrapped.layer == "last":
            z = outputs.last_hidden_state
        elif opts.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-opts.CLIP_stop_at_last_layers]
            z = self.wrapped.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.hidden_states[self.wrapped.layer_idx]

        return z
