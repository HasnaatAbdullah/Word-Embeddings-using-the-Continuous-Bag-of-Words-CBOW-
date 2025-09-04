# Word-Embeddings-using-the-Continuous-Bag-of-Words-CBOW-
Implementation of Word Embeddings using the Continuous Bag of Words (CBOW) model and applying them to Sentiment Analysis.
# 🧠 Word Embeddings with Continuous Bag of Words (CBOW)

This project demonstrates how to **train word embeddings from scratch** using the **Continuous Bag of Words (CBOW)** architecture and leverage them for **sentiment analysis**.  
The model learns vector representations of words that capture semantic and syntactic relationships, which are then used to analyze the sentiment of text data.

---

## 🚀 Project Overview

Word embeddings are a powerful way to represent words numerically in a high-dimensional vector space.  
Unlike traditional **bag-of-words** approaches, embeddings preserve **semantic relationships** between words.

For example, the trained model learns that:

In this project, we:
1. Build a **CBOW model** from scratch.
2. Train it on text data to learn embeddings.
3. Visualize word vectors.
4. Use embeddings to perform **sentiment classification**.

---

## 🧠 Key Features

### **1. Data Preprocessing**
- Tokenization and cleaning of text.
- Creating context-target pairs for training.
- Building a vocabulary and mapping words to indices.

### **2. Continuous Bag of Words (CBOW) Model**
- Predicts the **target word** given surrounding context words.
- Implemented from scratch using **NumPy** and **PyTorch/TensorFlow**.
- Supports custom embedding dimensions and batch sizes.

### **3. Training & Backpropagation**
- Forward pass for predictions.
- Loss calculation using **cross-entropy**.
- Backpropagation to update embedding weights.

### **4. Sentiment Analysis**
- Uses trained embeddings as input features.
- Classifies text into **positive** and **negative** sentiment.
- Evaluates performance using **accuracy** and **confusion matrices**.

### **5. Visualization**
- 2D projection of high-dimensional word embeddings using **t-SNE** or **PCA**.
- Visualizes semantic clustering of related words.

---

## 🏗️ Model Architecture


      Context Words (Input)
            ↓
   [Embedding Layer]
            ↓
     Average Context
            ↓
    [Dense Prediction]
            ↓
  Target Word (Output)

# project structure
word-embeddings-cbow/
│── data/
│   ├── train.txt
│   ├── test.txt
│── notebooks/
│   └── word_embeddings_cbow.ipynb
│── src/
│   ├── preprocess.py
│   ├── cbow_model.py
│   ├── train.py
│   ├── sentiment_analysis.py
│── results/
│   └── embeddings_visualization.png
│── requirements.txt
└── README.md
