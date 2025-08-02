# rag_pipeline.py

import pandas as pd
import faiss
import numpy as np
import replicate
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

class RAGPipeline:
    def __init__(self, csv_path, top_k=3):
        self.df = pd.read_csv(csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.top_k = top_k
        self.embeddings = None
        self.index = None
        self._prepare_index()

    def _prepare_index(self):
        print("[INFO] Generating TF-IDF embeddings...")
        tfidf_matrix = self.vectorizer.fit_transform(self.df['review'].values)
        self.embeddings = tfidf_matrix.toarray().astype('float32')

        print("[INFO] Building FAISS index...")
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print(f"[INFO] FAISS index built with {self.index.ntotal} documents.")

    def retrieve(self, query):
        print(f"[INFO] Retrieving documents for query: {query}")
        query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        distances, indices = self.index.search(query_vec, self.top_k)
        return self.df.iloc[indices[0]]['review'].tolist()

    def construct_prompt(self, query):
        retrieved_docs = self.retrieve(query)
        context = "\n\n".join(retrieved_docs)
        prompt = f"""Classify the sentiment and summarize the following movie reviews:\n\n{context}\n\nRespond in this format:\nSentiment: [Positive/Negative]\nSummary: [One-sentence summary]\nReason: [Justification]"""
        return prompt

    def query_model(self, prompt, system_prompt):
        print("[INFO] Sending prompt to IBM Granite via Replicate API...")
        output = replicate.run(
            "ibm-granite/granite-3.3-8b-instruct",
            input={
                "prompt": prompt,
                "system_prompt": system_prompt
            }
        )
        return "".join(output)

    def run(self, query):
        prompt = self.construct_prompt(query)
        default_system_prompt = (
            "You are a helpful assistant that classifies sentiment and summarizes movie reviews. "
            "Respond in the same language as the input text."
        )
        result = self.query_model(prompt, system_prompt=default_system_prompt)
        print("\n[RESULT]")
        print(result)
        return result