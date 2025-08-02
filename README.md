# 🎮 Sentiment & Summary Analysis of IMDb Movie Reviews Using IBM Granite

## 📌 Project Overview

This project explores the use of a Large Language Model (LLM) — specifically IBM Granite 3.3B — to classify the sentiment and generate summaries of movie reviews. The approach is built on a Retrieval-Augmented Generation (RAG) pipeline to enhance context-aware reasoning, and it supports **both English and Indonesian** inputs.

The entire project runs in **VS Code**, powered by **FAISS**, **TF-IDF**, and inference through **Replicate**.

---

## 🗖️ Dataset

* **Source**: [IMDb Movie Reviews Dataset – Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* **Size**: 50,000 labeled reviews (`positive` / `negative`)
* **Custom Batch Input**: 10 reviews (5 English + 5 Indonesian) selected randomly for demo and testing
* **File**: `data/input_reviews.csv`

---

## 🔍 Objectives

* ✅ Perform sentiment classification (positive/negative)
* ✅ Generate concise summaries for each review
* ✅ Support multilingual input (Bahasa Indonesia & English)
* ✅ Provide insight on public perception and recommendation to stakeholders

---

## 🧐 Insight & Findings

* Out of **10 movie reviews**, **7 were classified as positive**, **3 as negative**
* **Positive reviews** emphasize emotional satisfaction, acting, and humor
* **Negative reviews** criticize predictable plots, slow pacing, and weak story structure

---

## 🧠 Conclusion & Recommendation

**Conclusion**:
Majority of viewers express satisfaction, but consistent feedback highlights dissatisfaction with storytelling aspects.

**Recommendation**:
Filmmakers should prioritize originality and pacing in storytelling, especially for long-duration films. Creative, engaging plots with strong character development are critical for audience retention.

---

## 🤖 AI Support Explanation

* **LLM Used**: `ibm-granite/granite-3.3-8b-instruct` via [Replicate](https://replicate.com)
* **Why Granite**: Supports multilingual prompts, fine-tuned for classification and summarization
* **RAG Pipeline**:

  * **TF-IDF + FAISS** to retrieve top-K relevant reviews
  * Prompt LLM with retrieved context to generate structured output:

    ```
    Sentiment: [Positive/Negative]
    Summary: [One-sentence summary]
    Reason: [Justification]
    ```

---

## 🛠️ Tech Stack

* 🧠 IBM Granite LLM (8B Instruct)
* 🔎 Retrieval: TF-IDF + FAISS
* 🧪 Inference: Replicate API
* 💻 Environment: VS Code + Python
* 🛠️ Libraries: pandas, faiss-cpu, langdetect, dotenv, sklearn

---

## 🚀 How to Run

### 1. Clone and Install

```bash
pip install -r requirements.txt
```

### 2. Prepare `.env`

```env
REPLICATE_API_TOKEN=your_key_here
```

### 3. Run Inference (Single)

```bash
python app/infer.py --query "Film ini membosankan dan terlalu panjang."
```

### 4. Run Inference (Batch)

```bash
python app/infer_batch.py
```

---

## 📁 Project Structure

```
imdb-sentiment-analysis/
├── data/
│   ├── IMDB_Dataset.csv
│   ├── input_reviews.csv
│   └── results.csv
├── app/
│   ├── rag_pipeline.py
│   ├── infer.py
│   └── infer_batch.py
├── utils/
├── slides/
├── README.md
└── requirements.txt
```

---

## 📊 Sample Output

```text
Sentiment: Negative
Summary: The plot was predictable and slow-paced.
Reason: Many users criticized the story for lacking originality.
```

---

## 📌 Submission Links

* 🔗 GitHub Repository: [https://github.com/ncholis/imdb-sentiment-analysis]
* 🖼️ Slides: [belum ada]
* 🗂️ Dataset Source: [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🙏 Acknowledgements

* IBM & Replicate for model access
* Hacktiv8 Capstone guidelines
* IMDb & Kaggle for dataset availability

```
```