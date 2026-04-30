import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from peft import get_peft_model, LoraConfig

from models.losses.triplet import online_triplet_loss


class SSLTeacher(nn.Module):
    def __init__(self, model_name, out_dim=276, lora_r=8, lora_alpha=16, margin=0.5):
        super(SSLTeacher, self).__init__()

        backbone = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)
        self.proj = nn.Linear(backbone.config.hidden_size, out_dim)
        self.criterion = online_triplet_loss({'margin': margin})
        self.emb_norm = True

    def get_embeddings(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        out = self.backbone(input_values=x).last_hidden_state
        z = out.mean(dim=1)
        z = self.proj(z)
        z = F.normalize(z, p=2.0, dim=-1)
        return z

    @classmethod
    def load(cls, path, map_location=None):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(ckpt['model_name'], ckpt['out_dim'], ckpt['lora_r'], ckpt['lora_alpha'], ckpt['margin'])
        model.load_state_dict(ckpt['model_state_dict'])
        return model

    def loss(self, x):
        n_class = x.size(0)
        n_sample = x.size(1)
        x = x.view(n_class * n_sample, *x.size()[2:])
        zq = self.get_embeddings(x)
        loss_val = self.criterion.compute(zq, n_sample, n_class)
        return loss_val, {'loss': loss_val.item()}
