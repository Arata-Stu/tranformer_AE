import os
import torch
import lightning.pytorch as pl
import torchvision.transforms as transforms
from PIL import Image
import argparse
from modules.model.model import AE_LightningModule

def load_model(checkpoint_path, config_path):
    """
    学習済みモデルをロードする
    """
    model = AE_LightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        config_path=config_path
    )
    model.eval()  # 推論モード
    model.freeze()  # 勾配を停止
    return model

def preprocess_image(image_path, img_size=224):
    """
    画像を前処理し、テンソルに変換
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # バッチ次元を追加

def save_output(image_tensor, output_path):
    """
    推論結果のテンソルを画像として保存
    """
    image = transforms.ToPILImage()(image_tensor.squeeze(0).cpu().detach())  # テンソルを PIL 画像に変換
    image.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaxVIT AutoEncoder 推論スクリプト")
    parser.add_argument("--checkpoint", type=str, required=True, help="学習済みモデルの .ckpt ファイル")
    parser.add_argument("--config", type=str, required=True, help="モデル設定ファイル (.yaml)")
    parser.add_argument("--input", type=str, required=True, help="入力画像のパス")
    parser.add_argument("--output", type=str, required=True, help="出力画像の保存先")

    args = parser.parse_args()

    # モデルのロード
    model = load_model(args.checkpoint, args.config)

    # 画像の前処理
    img_size = args.config.encoder.maxvit.img_size
    input_tensor = preprocess_image(args.input, img_size=img_size)

    # 推論
    with torch.no_grad():
        reconstructed, _ = model(input_tensor)

    # 出力画像を保存
    save_output(reconstructed, args.output)

    print(f"入力画像: {args.input}")
    print(f"再構成画像を保存しました: {args.output}")
