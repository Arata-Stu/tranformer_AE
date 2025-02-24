import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection

def collate_fn_ignore_target(batch):
    # バッチ内の各サンプルから画像だけを抽出
    images = [sample[0] for sample in batch]
    images = torch.stack(images, dim=0)
    return images

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, train_ann, val_dir, val_ann, img_size=224, batch_size=32, num_workers=10):
        super().__init__()
        self.train_dir = train_dir
        self.train_ann = train_ann
        self.val_dir = val_dir
        self.val_ann = val_ann
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        self.train_dataset = CocoDetection(root=self.train_dir, annFile=self.train_ann, transform=self.transform)
        self.val_dataset = CocoDetection(root=self.val_dir, annFile=self.val_ann, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn_ignore_target
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn_ignore_target
        )
