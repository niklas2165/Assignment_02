# Binary Sentiment Classification Using Transformers

## Introduction

This project demonstrates fine-tuning a pre-trained transformer model to perform binary sentiment classification using the IMDb dataset. The task involves classifying movie reviews as either negative (0) or positive (1). The implementation leverages the Hugging Face Transformers and Datasets libraries, along with PyTorch, to preprocess the data, fine-tune a pre-trained DistilBERT model, evaluate the model, and save the final model for future use.

## Task Description

The assignment includes the following key steps:

1. **Dataset Selection and Preprocessing**  
   - Using the IMDb dataset from Hugging Face, which contains text reviews and their corresponding binary sentiment labels.
   - Tokenizing the dataset with a pre-trained DistilBERT tokenizer.
   - Splitting the data into training, validation, and test sets.

2. **Model Selection and Fine-Tuning**  
   - Loading a pre-trained DistilBERT model for sequence classification.
   - Fine-tuning the model on the processed dataset using the Hugging Face `Trainer` API.
   - Configuring training parameters, including learning rate, batch size, number of epochs, and evaluation strategy.

3. **Evaluation**  
   - Evaluating model performance using metrics such as accuracy, F1-score, precision, and recall.
   - Analyzing the model's performance on the test set.

4. **Saving the Model**  
   - Saving the fine-tuned model for later use.

## Requirements

- Python 3.x
- [Transformers](https://github.com/huggingface/transformers)
- [Datasets](https://github.com/huggingface/datasets)
- [Scikit-Learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- (Optional) Google Colab for easy experimentation

## Installation

Install the necessary libraries using pip:

```bash
pip install -U transformers datasets scikit-learn torch
