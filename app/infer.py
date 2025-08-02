# infer.py

from rag_pipeline import RAGPipeline
from langdetect import detect
import argparse

def detect_language(text):
    try:
        lang = detect(text)
        if lang == 'id':
            return 'id'
        else:
            return 'en'
    except:
        return 'unknown'

def build_prompt(review_text, lang):
    if lang == 'id':
        system_prompt = (
            "Kamu adalah asisten AI yang membantu mengklasifikasikan dan merangkum ulasan film. "
            "Berikan jawabanmu dalam Bahasa Indonesia.\n"
            "Format:\n"
            "Sentimen: [Positif/Negatif]\n"
            "Ringkasan: [Ringkasan satu kalimat]\n"
            "Alasan: [Penjelasan singkat]"
        )
        user_prompt = f"Klasifikasikan sentimen dan buat ringkasan dari review berikut:\n\n{review_text}"
    else:
        system_prompt = (
            "You are an AI assistant that classifies and summarizes movie reviews. "
            "Respond in English using this format:\n"
            "Sentiment: [Positive/Negative]\n"
            "Summary: [One-sentence summary]\n"
            "Reason: [Short explanation]"
        )
        user_prompt = f"Classify and summarize the following review:\n\n{review_text}"

    return system_prompt, user_prompt

def process_review(rag, query_text):
    lang = detect_language(query_text)
    docs = rag.retrieve(query_text)
    context = "\n\n".join(docs)
    system_prompt, user_prompt = build_prompt(context, lang)
    
    result = rag.query_model(
        prompt=user_prompt,
        system_prompt=system_prompt
    )

    print("\n--- RESULT ---")
    print(result)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Sentiment + Summary")
    parser.add_argument("--query", type=str, help="Review or query text to analyze")
    args = parser.parse_args()

    rag = RAGPipeline(csv_path="data/IMDB_Dataset.csv", top_k=3)

    if args.query:
        process_review(rag, args.query)
    else:
        query = input("Masukkan review atau topik analisis: ")
        process_review(rag, query)