import os
import json
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
import matplotlib.pyplot as plt 
import seaborn as sns 
def load_and_preprocess_data():
    print("Loading data...")
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Goodreads_books_with_genres.csv')
    books = pd.read_csv(csv_path)
    books.dropna(inplace=True)
    df = books[['Title', 'Author', 'language_code', 'publication_date', 'publisher', 'genres', 'ratings_count']].copy()
    df['Title'] = df['Title'].str.split('(').str[0].str.strip()
    df['Author'] = df['Author'].str.split('/')
    df['publication_date'] = df['publication_date'].astype(str).str.split('/').str[-1]
    df['publication_date'] = df['publication_date'].apply(lambda x: [x])
    others = df['language_code'].value_counts()[df['language_code'].value_counts() < 100]
    df['language_code'] = df['language_code'].apply(lambda x: ['other' if x in others else x])
    df['publisher'] = df['publisher'].apply(lambda x: [str(x).replace(' ', '')])
    df['genres'] = df['genres'].apply(lambda x: list(set(str(x).replace(" ", ";").replace(",", ";").split(";"))))
    df['tags'] = df['Author'] + df['language_code'] + df['publication_date'] + df['publisher'] + df['genres']
    new_df = df[['Title', 'tags', 'ratings_count']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    return new_df.iloc[0:1000].copy()
def compute_metrics():
    df = load_and_preprocess_data()
    print("Computing feature vectors and similarity matrix...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    similarity = cosine_similarity(vectors)
    print("Generating recommendations for all items...")
    top_k = 5
    recommendations_idx = []
    for idx in range(len(df)):
        sim_scores = list(enumerate(similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_items = [i[0] for i in sim_scores[1:top_k+1]]
        recommendations_idx.append(top_items)
    print("Calculating metrics...")
    unique_recommended_items = set([item for sublist in recommendations_idx for item in sublist])
    catalog_coverage = len(unique_recommended_items) / len(df)
    ils_scores = []
    for rec_list in recommendations_idx:
        if len(rec_list) > 1:
            rec_vectors = vectors[rec_list]
            rec_sim = cosine_similarity(rec_vectors)
            upper_tri = rec_sim[np.triu_indices(len(rec_list), k=1)]
            ils_scores.append(np.mean(upper_tri))
        else:
            ils_scores.append(0)
    avg_ils = np.mean(ils_scores)
    jaccard_sims = []
    num_samples = 1000
    for _ in range(num_samples):
        i, j = np.random.choice(len(recommendations_idx), 2, replace=False)
        set_i = set(recommendations_idx[i])
        set_j = set(recommendations_idx[j])
        intersection = len(set_i.intersection(set_j))
        union = len(set_i.union(set_j))
        if union > 0:
            jaccard_sims.append(intersection / union)
    personalization = 1.0 - np.mean(jaccard_sims)
    log_pops = np.log1p(df['ratings_count'].values)
    max_log_pop = np.max(log_pops)
    min_log_pop = np.min(log_pops)
    list_novelties = []
    for rec_list in recommendations_idx:
        rec_pops = log_pops[rec_list]
        norm_novelty = 1.0 - ((np.mean(rec_pops) - min_log_pop) / (max_log_pop - min_log_pop + 1e-9))
        list_novelties.append(norm_novelty)
    avg_novelty = np.mean(list_novelties)
    metrics = {
        "catalog_coverage": round(catalog_coverage, 4),
        "intra_list_similarity": round(avg_ils, 4),
        "personalization": round(personalization, 4),
        "novelty": round(avg_novelty, 4)
    }
    print("Saving metrics...")
    output_dir = os.path.dirname(__file__)
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plot_data = {
        'Metric': ['Catalog Coverage', 'Diversity (1 - ILS)', 'Personalization', 'Novelty'],
        'Score': [catalog_coverage, 1.0 - avg_ils, personalization, avg_novelty]
    }
    plot_df = pd.DataFrame(plot_data)
    ax = sns.barplot(x='Score', y='Metric', data=plot_df, palette="viridis")
    plt.xlim(0, 1.0)
    plt.title('Content-Based Recommender Evaluation Metrics (0 to 1 scale)', fontsize=16)
    for i, v in enumerate(plot_df['Score']):
        ax.text(v + 0.01, i, str(round(v, 2)), color='black', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'), dpi=300)
    print("Done! Check evaluation_metrics.json and evaluation_metrics.png")

if __name__ == "__main__":
    compute_metrics()
