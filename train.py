import os
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from modules.model.model import AE_LightningModule
from modules.data.coco import CocoDataModule
from datetime import datetime

if __name__ == "__main__":
    # 実行ごとの保存ディレクトリを設定 (YYYYMMDD-HHMMSS 形式)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = os.path.join("./ckpts", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    yaml_path = "./config/MaxVIT.yaml"
    cfg = OmegaConf.load(yaml_path)

    # DataModuleの設定
    coco_dm = CocoDataModule(
        train_dir='./datasets/coco/images/train2017',
        train_ann='./datasets/coco/annotations/instances_train2017.json',
        val_dir='./datasets/coco/images/val2017',
        val_ann='./datasets/coco/annotations/instances_val2017.json',
        img_size=cfg.encoder.maxvit.img_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # TensorBoard ロガーの設定 (実行ごとに異なるディレクトリ)
    logger = TensorBoardLogger("logs/", name=f"maxvit_ae_{run_id}")

    # チェックポイント保存の設定 (実行ごとのサブフォルダ)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_dir,
        filename="ae-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        mode="min"
    )

    # Lightning Trainer の初期化
    trainer = pl.Trainer(
        precision=16,
        max_epochs=300,
        accelerator='gpu' if torch.cuda.is_available() else None,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_callback],
        logger=logger,  # ログディレクトリも変更
    )

    # モデルの初期化
    model = AE_LightningModule(config_path="./config/MaxVIT.yaml", lr=cfg.lr)

    # 学習の実行
    trainer.fit(model, coco_dm)
