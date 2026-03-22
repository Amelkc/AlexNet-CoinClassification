from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

#adapter images en entrée au pré-requis du modèle
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CoinDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=transform, istrain=True):
        self.df       = pd.read_csv(csv_file)          
        self.img_dir  = img_dir
        self.transform = transform
        self.train = istrain

        if istrain==True:
            self.classes    = sorted(self.df["Class"].unique().tolist())
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.df["label"] = self.df["Class"].map(self.class_to_idx)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row      = self.df.iloc[idx]
        img_id = str(row["Id"])
        candidates = [f for f in os.listdir(self.img_dir) if os.path.splitext(f)[0] == img_id and os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']]
        if not candidates:
            return None
        img_path = os.path.join(self.img_dir, candidates[0])
        image    = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.train == True :
            label = int(row["label"])
            return image, label
        return image, img_id
    
#nettoyer les donnees en retirant les img sans ext
def collate(batch):
    batch = [sample for sample in batch if sample is not None]
    return torch.utils.data.dataloader.default_collate(batch)