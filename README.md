# Multimodal Hotel Review Prediction

This project aims to predict hotel review ratings by combining textual and visual inputs using a multimodal deep learning approach. The model processes both text (customer reviews) and images (uploaded with the reviews) to predict ratings from 1 to 5.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)

## Project Overview

The goal of this project is to improve hotel review rating predictions by leveraging both textual reviews and the accompanying images. We combine text-based features extracted using BERT and image features extracted using a Convolutional Neural Network (CNN).

## Dataset

The dataset consists of:
- **Text Reviews**: Customer feedback in textual format.
- **Image Reviews**: Images submitted with the reviews.
- **Ratings**: Ratings ranging from 1 to 5.


## Model Architecture

The model is a combination of:
- **BERT (Bidirectional Encoder Representations from Transformers)** for extracting text features from customer reviews.
- **CNN (Convolutional Neural Network)** for extracting image features from the corresponding review images.

The text and image features are concatenated and passed through fully connected layers to predict the rating.

### Key Components
- **Text Features**: Extracted using the pre-trained BERT model.
- **Image Features**: Extracted by flattening the image after applying necessary transformations (resizing, normalization).
- **Combined Model**: Combines both text and image features and passes through a dense layer for final prediction.

### Technologies Used
**Python**: Programming language
**PyTorch**: Deep learning framework for model development and training
**Hugging Face Transformers**: Pre-trained BERT for text feature extraction
**Torchvision**: Image processing and CNN-based feature extraction
**Pandas/Scikit-learn**: Data manipulation and preprocessing
**Google Colab/GPU Server**: For model training with GPU acceleration
