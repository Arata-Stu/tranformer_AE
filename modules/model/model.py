import torch
import torch.nn.functional as F
import math
from omegaconf import OmegaConf
import lightning.pytorch as pl
from models.AE.maxvit_ae import MaxVITAE  # AEモデルを使用
from models.AE.maxvit_vae import MaxVITVAE  # VAEモデルを使用

class VAE_LightningModule(pl.LightningModule):
    """
    PyTorch Lightning 用の VAE モジュール
    """

    def __init__(self, config_path: str, lr: float = 1e-3):
        super().__init__()

        cfg = OmegaConf.load(config_path)
        self.model = MaxVITVAE(cfg=cfg, latent_dim=cfg.latent_dim)  # VAEモデルを初期化
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch  # バッチは画像のみのテンソル
        x_recon, mu, log_var = self(x)
        loss, recon_loss, kl_loss = self.model.loss_function(x, x_recon, mu, log_var)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, log_var = self(x)
        loss, recon_loss, kl_loss = self.model.loss_function(x, x_recon, mu, log_var)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        warmup_steps = 10000
        total_steps = 1000000

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }


class AE_LightningModule(pl.LightningModule):
    """
    PyTorch Lightning 用の AE モジュール
    """

    def __init__(self, config_path: str, lr: float = 1e-3):
        super().__init__()

        cfg = OmegaConf.load(config_path)
        self.model = MaxVITAE(cfg=cfg, latent_dim=cfg.latent_dim)  # AEモデルを初期化
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch  # バッチは画像のみのテンソル
        x_recon, _ = self(x)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
        loss = recon_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, _ = self(x)
        recon_loss = F.mse_loss(x_recon, x, reduction="sum") / x.size(0)
        loss = recon_loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        warmup_steps = 10000
        total_steps = 1000000

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
