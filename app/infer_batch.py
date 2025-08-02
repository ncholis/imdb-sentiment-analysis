# infer_batch.py

import pandas as pd
from langdetect import detect
from rag_pipeline import RAGPipeline
import time

def detect_language(text):
    try:
        lang = detect(text)
        return 'id' if lang == 'id' else 'en'
    except:
        return 'unknown'

def build_prompt(review_text, lang):
    if lang == 'id':
        system_prompt = (
            "Kamu adalah asisten AI yang mengklasifikasikan dan meringkas ulasan film. "
            "Jawablah dalam Bahasa Indonesia.\n\n"
            "Format:\nSentimen: [Positif/Negatif]\nRingkasan: [Satu kalimat ringkasan]\nAlasan: [Penjelasan singkat]"
        )
        user_prompt = f"Klasifikasikan sentimen dan buat ringkasan dari review berikut:\n\n{review_text}"
    else:
        system_prompt = (
            "You are an AI assistant that classifies and summarizes movie reviews.\n\n"
            "Format:\nSentiment: [Positive/Negative]\nSummary: [One-sentence summary]\nReason: [Short explanation]"
        )
        user_prompt = f"Classify and summarize the following review:\n\n{review_text}"
    return user_prompt, system_prompt

def batch_process(csv_path, output_path="data/results.csv", limit=None):
    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip', encoding='utf-8')
    if limit:
        df = df.head(limit)

    rag = RAGPipeline(csv_path="data/IMDB_Dataset.csv", top_k=3)

    results = []

    for i, row in df.iterrows():
        review_text = row['review']
        review_id = row.get('review_id', i)
        lang = detect_language(review_text)
        print(f"\n[🔍] Processing ID: {review_id} | Language: {lang}")

        try:
            retrieved_docs = rag.retrieve(review_text)
            context = "\n\n".join(retrieved_docs)
            user_prompt, system_prompt = build_prompt(context, lang)
            result = rag.query_model(user_prompt, system_prompt)
        except Exception as e:
            print(f"[⚠️] Error processing ID {review_id}: {e}")
            result = "ERROR"

        results.append({
            "review_id": review_id,
            "original_review": review_text,
            "language": lang,
            "output": result
        })

        time.sleep(1.5)  # throttle to avoid rate limits

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)
    print(f"\n✅ Done! Results saved to {output_path}")

if __name__ == "__main__":
    batch_process("data/input_reviews.csv", limit=10)  # Change limit=None for full run
