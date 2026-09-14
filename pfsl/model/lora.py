import torch
import torch.nn as nn
import esm
from peft import LoraConfig, get_peft_model
import gc

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from .linear import LinearProbing

__all__ = ["LoRA"]


@MODEL_REGISTRY.register()
class LoRA(nn.Module):

    @configurable
    def __init__(self, 
                 num_end_lora_layers: int, num_lora_r: int, 
                 num_lora_alpha: int, lora_dropout: float,
                 in_channels: int, out_channels: int,
                 device: str) -> None:
        super().__init__()
        self.device = device

        model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()

        if num_end_lora_layers > 0:
            target_modules = []
            start_layer_idx = model.num_layers - num_end_lora_layers
            for idx in range(start_layer_idx, model.num_layers):
                for layer_name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]:
                    target_modules.append(f"layers.{idx}.{layer_name}")
        
        config = LoraConfig(inference_mode=False,
                            r=num_lora_r,
                            lora_alpha=num_lora_alpha,
                            target_modules=target_modules,
                            lora_dropout=lora_dropout,
                            bias="none",)
        self.model = get_peft_model(model, config)

        self.head = LinearProbing(in_channels, out_channels)

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_end_lora_layers": cfg.MODEL.NUM_END_LORA_LAYERS,
            "num_lora_r": cfg.MODEL.NUM_LORA_R,
            "num_lora_alpha": cfg.MODEL.NUM_LORA_ALPHA,
            "lora_dropout": cfg.MODEL.LORA_DROPOUT,
            "in_channels": cfg.MODEL.IN_CHANNELS,
            "out_channels": cfg.MODEL.OUT_CHANNELS,
            "device": cfg.MODEL.DEVICE,
        }

    def esm_batch_infer(self, x: list[tuple[str]]) -> torch.Tensor:
        _, _, batch_tokens = self.batch_converter(x)  # all proteins' names, all proteins sequences, torch.Size([#proteins, max_protein_length_in_this_batch])
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        batch_tokens = batch_tokens.to(self.device)

        model_outputs = self.model(batch_tokens, repr_layers=[33])
        token_representations = model_outputs["representations"][33]
        xs = []
        for i, tokens_len in enumerate(batch_lens):
            xs.append(token_representations[i, 1 : tokens_len - 1].mean(0))
        xs = torch.stack(xs, dim=0)

        del batch_tokens
        del token_representations
        del model_outputs
        return xs

    def forward(self, gene1: list[tuple[str]], gene2: list[tuple[str]]) -> torch.Tensor:
        gene1 = self.esm_batch_infer(gene1)
        gene2 = self.esm_batch_infer(gene2)
        x = torch.cat([gene1, gene2], dim=1)
        x = self.head(x)

        del gene1
        del gene2
        gc.collect()
        return x.squeeze(-1)
