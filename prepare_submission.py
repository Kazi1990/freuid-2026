import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader

DATA_DIR = '/data'
OUTPUT_DIR = '/submissions'
WEIGHTS_PATH = '/app/model.pth'

image_files = []
for f in os.listdir(DATA_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff')):
        image_files.append(f)

print(f"Found {len(image_files)} images in {DATA_DIR}")

img_lookup = {}
for f in image_files:
    img_id = os.path.splitext(f)[0]
    img_lookup[img_id] = os.path.join(DATA_DIR, f)

class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FraudNet().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

class FraudDataset(Dataset):
    def __init__(self, lookup, transform):
        self.items = list(lookup.items())
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        img_id, path = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_id

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = FraudDataset(img_lookup, test_tf)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

preds = {}
with torch.no_grad():
    for imgs, ids in loader:
        imgs = imgs.to(device)
        out = model(imgs)
        scores = torch.sigmoid(out).cpu().numpy()
        for img_id, sc in zip(ids, scores):
            preds[img_id] = float(sc)

os.makedirs(OUTPUT_DIR, exist_ok=True)
sub = pd.DataFrame({'id': list(preds.keys()), 'label': list(preds.values())})
sub = sub.sort_values('id').reset_index(drop=True)
sub.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)

print(f"Done! {len(sub)} rows -> {OUTPUT_DIR}/submission.csv")
print(f"Min: {sub['label'].min():.6f}, Max: {sub['label'].max():.6f}")
