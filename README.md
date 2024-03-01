# Image2Emoji

This repository contains the code for Image2Emoji, a Zero-shot Emoji Prediction model. The model is based on OpenAI's CLIP model and is fine-tuned on Flickr-8k Dataset. The model is able to predict the most relevant emoji for a given image.

## 📎 Model
OpenAI's CLIP (Contrastive Language-Image Pretraining) model can embed different types of data into a common feature space by learning to predict which images correspond to which text descriptions.
I opted for lightweight models for each encoder (Pretrained ResNet18 for the image encoder and Pretrained ALBERT for the text encoder), aiming to deploy as a web application.
After fine-tuning the model on the Flickr-8k dataset, the model is able to predict the most similar emoji for a given image.
![clip](pictures/clip.png)

## 📦 Installation
To install the required packages, run the following command:
```bash
pip install -r requirements.txt
```

## 📚 Dataset
Dataset used for fine-tuning the model is Flickr-8k dataset. The dataset contains 8,000 images that are each paired with captions. You have to put the dataset in root directory of the project.
```bash
root -- Flickr8k
          |-- images
          |-- captions.txt
```

## 🤔 Zero-shot Emoji Prediction

## 🗺️ Emoji Caption Feature Space


## 📝 Refferences
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [CLIP wandb Implementation](https://github.com/soumik12345/clip-lightning)
- [Flickr-8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Emoji Dataset](https://huggingface.co/datasets/valhalla/emoji-dataset)