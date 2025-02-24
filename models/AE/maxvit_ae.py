import sys
sys.path.append('../..')
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from models.timm.maxvit_encoder import MaxxVitEncoder
from models.timm.maxvit_decoder import MaxxVitDecoder

class MaxVITAE(nn.Module):
    """
    MaxVIT Autoencoder (AE) のクラス実装。

    画像を潜在ベクトルに圧縮し、リプレイバッファに保存しやすくする。
    YAML設定ファイルからエンコーダとデコーダの設定を読み込み、
    MaxVITEncode と MaxVITDecode のインスタンスを生成。
    """
    def __init__(self, config_path: str = '../../config/MaxVIT_AE.yaml'):
        super(MaxVITAE, self).__init__()
        
        print(f'Loading model from {config_path}')
        with open(config_path, 'r') as f:
            config_content = f.read()
            print('Model config:')
            print('----------------')
            print(config_content)
            print('----------------')
        
        # YAML設定を読み込み
        cfg = OmegaConf.load(config_path)
        encoder_cfg = cfg.encoder
        decoder_cfg = cfg.decoder

        # エンコーダの初期化
        self.encoder = MaxxVitEncoder(encoder_cfg.maxvit, img_size=encoder_cfg.maxvit.img_size)

        # エンコーダの出力チャネル数・特徴マップサイズを取得
        encoder_out_chs = self.encoder.feature_info[-1]['num_chs']
        encoder_out_size = encoder_cfg.maxvit.img_size // self.encoder.feature_info[-1]['reduction']

        # デコーダの初期化
        self.decoder = MaxxVitDecoder(decoder_cfg.maxvit, in_chans=encoder_out_chs, input_size=encoder_out_size)

    def encode(self, x):
        """
        入力 x をエンコードし、潜在ベクトルに変換
        """
        z = self.encoder(x)  # shape: (B, C, H, W)
        return z

    def decode(self, z):
        """
        潜在ベクトル z をデコードし、画像を復元
        """
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        """
        画像を潜在ベクトルに圧縮 → 再構成
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


if __name__ == "__main__":
    # テスト用：ダミー入力（例: 1サンプル, チャンネル3, 256x256画像）
    model = MaxVITAE('../../config/MaxVIT_AE.yaml')
    dummy_input = torch.randn(1, 3, 256, 256)
    x_recon, z = model(dummy_input)
    print("Reconstructed output shape:", x_recon.shape)  # (B, 3, 256, 256)
    print("Latent vector shape:", z.shape)  # (B, C, H, W)
