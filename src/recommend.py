# Movie Recommender System
# Author: Francisco Nunez

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('data/movies.csv')
print(df[['title', 'overview']].head())

# Fill missing overviews
df['overview'] = df['overview'].fillna('')

# Convert overviews to TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build reverse mapping of movie titles to DataFrame indices
title_to_index = pd.Series(df.index, index=df['title']).drop_duplicates()

# Recommendation function
def recommend(title, top_n=10):
    idx = title_to_index.get(title)
    if idx is None:
        return f"Movie '{title}' not found in the database."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[recommended_indices].tolist()

import os
import matplotlib.pyplot as plt

# Prompt user for movie title
input_title = input("Enter a movie title: ").strip()
recs = recommend(input_title)
print(f"\nRecommendations for '{input_title}':")
print(recs)

# Save similarity scores to CSV and plot chart
movie_idx = title_to_index.get(input_title)
if movie_idx is not None:
    sim_scores = list(enumerate(cosine_sim[movie_idx]))
    os.makedirs('output', exist_ok=True)
    sim_df = pd.DataFrame(sim_scores, columns=['movie_index', 'similarity_score'])
    sim_df['movie_title'] = sim_df['movie_index'].apply(lambda i: df['title'].iloc[i])
    sim_df = sim_df.sort_values(by='similarity_score', ascending=False)
    safe_title = input_title.replace(" ", "_").replace(":", "").lower()
    sim_df.to_csv(f'output/similarity_scores_{safe_title}.csv', index=False)

    # Plot top 10 similar movies
    top_sim_df = sim_df[1:11]  # Skip the first one (the movie itself)
    plt.figure(figsize=(10, 6))
    plt.barh(top_sim_df['movie_title'], top_sim_df['similarity_score'], color='skyblue')
    plt.xlabel("Similarity Score")
    plt.title(f"Top 10 Similar Movies to '{input_title}'")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'output/similarity_chart_{safe_title}.png')
    plt.show()
else:
    print(f"Movie '{input_title}' not found in the database.")