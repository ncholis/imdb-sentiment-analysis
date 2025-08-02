import pandas as pd

# Ambil 5 review Bahasa Inggris dari IMDb dataset
df = pd.read_csv("data/IMDB_Dataset.csv")
english_sample = df.sample(n=5, random_state=42).reset_index(drop=True)
english_sample.insert(0, "review_id", range(1, 6))

# 5 review Bahasa Indonesia (bisa disesuaikan nanti)
indonesian_reviews = [
    "Film ini sangat menyentuh dan membuat saya menangis di akhir.",
    "Cerita film terlalu lambat dan tidak punya klimaks yang memuaskan.",
    "Aktor utamanya luar biasa, tapi karakter pendukung tidak dikembangkan dengan baik.",
    "Saya suka sinematografinya, tapi plotnya terlalu mudah ditebak.",
    "Film ini sangat lucu dan cocok ditonton bareng keluarga."
]

indonesian_sample = pd.DataFrame({
    "review_id": range(6, 11),
    "review": indonesian_reviews
})

# Gabungkan keduanya
final_df = pd.concat([english_sample[["review_id", "review"]], indonesian_sample], ignore_index=True)

# Simpan ke file
final_df.to_csv("data/input_reviews.csv", index=False, quoting=1)

print("✅ Multilingual input_reviews.csv saved (5 English + 5 Indonesian reviews)")