import sys
sys.path.append('../..')
import torch
import torch.nn as nn
from omegaconf import DictConfig
from ..timm.maxvit_encoder import MaxxVitEncoder
from ..timm.maxvit_decoder import MaxxVitDecoder

class MaxVITVAE(nn.Module):
    """
    MaxVIT Variational Autoencoder (VAE) のクラス実装。
    
    画像を潜在ベクトルに圧縮し、VAEの特性として潜在空間に確率的なサンプリングを導入。
    """
    def __init__(self, encoder_cfg: DictConfig, latent_dim: int = 512, device: torch.device = 'cpu'):
        super(MaxVITVAE, self).__init__()
        
        self.device = device
        encoder_cfg = encoder_cfg
        decoder_cfg = encoder_cfg

        # エンコーダの初期化
        self.encoder = MaxxVitEncoder(encoder_cfg.maxvit, img_size=encoder_cfg.maxvit.img_size)

        # エンコーダの出力チャネル数・特徴マップサイズを取得
        encoder_out_chs = self.encoder.feature_info[-1]['num_chs']
        encoder_out_size = encoder_cfg.maxvit.img_size // self.encoder.feature_info[-1]['reduction']
        self.latent_shape = (encoder_out_chs, encoder_out_size, encoder_out_size)

        # Flatten (特徴マップ → 1D 潜在ベクトル)
        self.flatten = nn.Flatten()

        # 潜在ベクトルの次元数
        latent_dim_from_encoder = encoder_out_chs * encoder_out_size * encoder_out_size
        self.latent_dim = min(latent_dim, latent_dim_from_encoder)

        # 平均と分散を学習する層
        self.fc_mu = nn.Linear(latent_dim_from_encoder, self.latent_dim)
        self.fc_log_var = nn.Linear(latent_dim_from_encoder, self.latent_dim)
        
        # デコーダの入力層
        self.fc_decode = nn.Linear(self.latent_dim, latent_dim_from_encoder)

        # デコーダの初期化
        self.decoder = MaxxVitDecoder(decoder_cfg.maxvit, in_chans=encoder_out_chs, input_size=encoder_out_size)
        
        # ckptの読み込み
        if encoder_cfg.ckpt_path is not None:
            print(f"Load checkpoint from {encoder_cfg.ckpt_path}")
            ckpt = torch.load(encoder_cfg.ckpt_path, map_location=device)
            state_dict = ckpt['state_dict']
            new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
            self.load_state_dict(new_state_dict)

    def reparameterization_trick(self, mu, log_var):
        """
        Reparameterization Trick により、ガウス分布からサンプリング
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        """
        入力 x をエンコードし、潜在変数 (mu, log_var) を計算
        """
        z = self.encoder(x)  # (B, C, H, W)
        z = self.flatten(z)  # (B, C*H*W)
        mu = self.fc_mu(z)  # (B, latent_dim)
        log_var = self.fc_log_var(z)  # (B, latent_dim)
        return mu, log_var

    def decode(self, z):
        """
        潜在ベクトル z から画像を復元
        """
        z = self.fc_decode(z)  # (B, C*H*W)
        z = z.view(-1, *self.latent_shape)  # (B, C, H, W)
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        """
        画像をエンコードして潜在変数を得て、再構成画像を出力
        """
        mu, log_var = self.encode(x)
        z = self.reparameterization_trick(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def loss_function(self, x, x_recon, mu, log_var):
        """
        VAE の損失関数: 再構成誤差 + KLダイバージェンス
        """
        recon_loss = nn.MSELoss()(x_recon, x)  # 再構成誤差
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x.shape[0]  # KLダイバージェンス
        return recon_loss + kl_loss, recon_loss, kl_loss

if __name__ == "__main__":
    # テスト用：ダミー入力
    model = MaxVITVAE('../../config/MaxVIT.yaml', latent_dim=512)
    dummy_input = torch.randn(1, 3, 256, 256)
    x_recon, mu, log_var = model(dummy_input)
    loss, recon_loss, kl_loss = model.loss_function(dummy_input, x_recon, mu, log_var)
    print("Reconstructed output shape:", x_recon.shape)  # (B, 3, 256, 256)
    print("Latent mean shape:", mu.shape)  # (B, 512)
    print("Latent log variance shape:", log_var.shape)  # (B, 512)
    print("Total loss:", loss.item(), "Recon loss:", recon_loss.item(), "KL loss:", kl_loss.item())
