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
        #train_df, val_df = train_test_split(self.df, test_size=0.2)

        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.image_dir, image)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        tokenized_captions = self.tokenizer(caption, padding='max_length', truncation=True, max_length=self.max_length).input_ids                            
        #tokenized_captions = [caption if caption != self.tokenizer.pad_token_id else -100 for caption in tokenized_captions]

        return img, torch.tensor(tokenized_captions), caption

def get_dataloaders(data_dir, tokenizer_name, batch_size=2, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    dataset = CustomDataset(data_dir, tokenizer_name, transform=transform)
    train_size = int(0.8*len(dataset))  # 80% train, 20% val
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader