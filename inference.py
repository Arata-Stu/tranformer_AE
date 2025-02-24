import os
import random
import torch
import lightning.pytorch as pl
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import argparse
from datetime import datetime
from omegaconf import OmegaConf
from torchvision.datasets import CocoDetection
from utils.timers import CudaTimer
from modules.model.model import AE_LightningModule

def load_model(checkpoint_path, config_path, device):
    """
    学習済みモデルをロードする
    """
    model = AE_LightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path, 
        config_path=config_path,
        map_location=device
    )
    model.eval()
    model.to(device)
    model.freeze()
    return model

def preprocess_images(image_paths, img_size=224, device="cpu"):
    """
    画像リストを前処理し、テンソルに変換
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    images = [transform(Image.open(p).convert("RGB")) for p in image_paths]
    return torch.stack(images).to(device)  # バッチ化してデバイスに移動

def save_images(originals, reconstructions, output_dir, filename="output.jpg"):
    """
    元画像と再構成画像を1枚にまとめて保存
    """
    os.makedirs(output_dir, exist_ok=True)

    # (N, C, H, W) を torch.cat で結合 (元画像 → 再構成画像)
    grid = torch.cat([originals, reconstructions], dim=0)
    grid = vutils.make_grid(grid, nrow=len(originals), padding=5, normalize=True)

    # 保存
    output_path = os.path.join(output_dir, filename)
    vutils.save_image(grid, output_path)
    print(f"出力画像を保存しました: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COCO の test データから複数画像を推論し、まとめて保存")
    parser.add_argument("--checkpoint", type=str, required=True, help="学習済みモデルの .ckpt ファイル")
    parser.add_argument("--config", type=str, required=True, help="モデル設定ファイル (.yaml)")
    parser.add_argument("--coco_test", type=str, required=True, help="COCO test 画像フォルダ")
    parser.add_argument("--num_samples", type=int, default=5, help="推論する画像の枚数")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="出力画像の保存先")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="使用するデバイス")

    args = parser.parse_args()

    # デバイスの設定
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    print(f"Using device: {device}")

    # モデルのロード
    model = load_model(args.checkpoint, args.config, device)

    # COCO test ディレクトリから画像を取得
    image_files = [os.path.join(args.coco_test, f) for f in os.listdir(args.coco_test) if f.endswith(('.jpg', '.png'))]
    selected_images = random.sample(image_files, min(args.num_samples, len(image_files)))

    # 画像の前処理
    cfg = OmegaConf.load(args.config)
    img_size = cfg.encoder.maxvit.img_size
    input_tensors = preprocess_images(selected_images, img_size=img_size, device=device)

    # 推論の計測開始
    with CudaTimer(device) as timer:
        with torch.no_grad():
            reconstructed, _ = model(input_tensors)
    elapsed_time = timer.get_time()
    
    # 画像をまとめて保存
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_images(input_tensors, reconstructed, args.output_dir, filename=f"batch_{timestamp}.jpg")

    print(f"推論時間: {elapsed_time:.4f} 秒")