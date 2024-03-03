### Purpose of this script: make dataloader for flickr8k

import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name, transform=None):
        self.df = pd.read_csv(data_dir+'/captions.txt')
        self.image_dir = data_dir+'/Images'
        self.transform = transform
        self.tokenizer= AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = 50

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.image_dir, image)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        tokenized_captions = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length).input_ids                            

        return img, torch.tensor(tokenized_captions), caption

def get_dataloaders(data_dir, tokenizer_name, batch_size=16, num_workers=2):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=10),
            ], p=0.5)
        ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
        ])

    train_dataset = CustomDataset(data_dir, tokenizer_name, transform=train_transform)
    val_dataset = CustomDataset(data_dir, tokenizer_name, transform=val_transform)
    indices = torch.randperm(len(train_dataset)).tolist()
    train_size = int(0.8*len(train_dataset))  # 80% train, 20% val
    # print(f"Train size: {train_size}, Val size: {len(train_dataset)-train_size}")
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:train_size])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[train_size:])

    # dataset = CustomDataset(data_dir, tokenizer_name, transform=val_transform)
    # train_size = int(0.8*len(dataset))  # 80% train, 20% val
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader