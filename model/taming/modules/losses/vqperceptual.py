"""
VQ-VAE reconstruction loss (no discriminator). Matches training_step in VQModel_SEP.
"""
import torch
import torch.nn as nn


class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0, pixelloss_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.pixelloss_weight = pixelloss_weight

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
    ):
        rec_l1 = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        rec_loss = torch.mean(rec_l1)
        l2_loss = torch.mean((inputs.contiguous() - reconstructions.contiguous()) ** 2)
        nll_loss = rec_loss
        loss = self.pixelloss_weight * nll_loss + self.codebook_weight * codebook_loss.mean()
        log = {
            f"{split}/total_loss": loss.detach().clone(),
            f"{split}/quant_loss": codebook_loss.detach().mean(),
            f"{split}/nll_loss": nll_loss.detach(),
            f"{split}/rec_loss": rec_loss.detach(),
            f"{split}/l2_loss": l2_loss.detach(),
        }
        return loss, log
