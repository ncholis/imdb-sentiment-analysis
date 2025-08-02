# Sentiment & Summarization of IMDb Movie Reviews using IBM Granite

## 📌 Project Overview
This project aims to classify movie review sentiments and generate summaries using AI/LLM from IBM Granite. It demonstrates how AI can enhance understanding of public opinion through natural language processing.

## 📊 Raw Dataset
[IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## 🧪 Insight & Findings
- Majority of reviews are positive.
- Common themes in negative reviews: acting, pacing, plot holes.
- LLM-generated summaries provide efficient short-form opinions.

## 🤖 AI Support Explanation
- **Sentiment Classification**: Using IBM Granite LLM for zero-shot sentiment tagging.
- **Summarization**: Reviews passed to LLM for key-points abstraction.

## 🔧 Tech Stack

- IBM Granite 3.3 8B Instruct (via Replicate)
- Retrieval-Augmented Generation (RAG)
- Python + FAISS
- VS Code as IDE
- Dataset: IMDb Movie Reviews (Kaggle)

## 🔐 How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
