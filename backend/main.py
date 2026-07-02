import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
import uvicorn 
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading data...")
csv_path = os.path.join(os.path.dirname(__file__), 'Goodreads_books_with_genres.csv')
books = pd.read_csv(csv_path)
books.dropna(inplace=True)

df = books[['Title', 'Author', 'language_code', 'publication_date', 'publisher', 'genres']].copy()
df['Title'] = df['Title'].str.split('(').str[0].str.strip()
df['Author'] = df['Author'].str.split('/')
df['publication_date'] = df['publication_date'].astype(str).str.split('/').str[-1]
df['publication_date'] = df['publication_date'].apply(lambda x: [x])

others = df['language_code'].value_counts()[df['language_code'].value_counts() < 100]
df['language_code'] = df['language_code'].apply(lambda x: ['other' if x in others else x])

df['publisher'] = df['publisher'].apply(lambda x: [str(x).replace(' ', '')])
df['genres'] = df['genres'].apply(lambda x: list(set(str(x).replace(" ", ";").replace(",", ";").split(";"))))

df['tags'] = df['Author'] + df['language_code'] + df['publication_date'] + df['publisher'] + df['genres']

new_df = df[['Title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df1 = new_df.iloc[0:1000].copy()

print("Computing similarity...")
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df1['tags']).toarray()
similarity = cosine_similarity(vectors)

details_df = books.iloc[0:1000].copy()
details_df['Title_clean'] = new_df1['Title'].values

@app.get("/books")
def get_books():
    return {"books": new_df1['Title'].tolist()}

@app.get("/recommend")
def recommend(title: str = Query(..., description="Title of the book")):
    try:
        idx = new_df1[new_df1['Title'] == title].index[0]
        book_similarity = list(enumerate(similarity[idx]))
        sorted_similarity = sorted(book_similarity, reverse=True, key=lambda x: x[1])[1:6]
        
        recommendations = []
        for i in sorted_similarity:
            row = details_df.iloc[i[0]]
            recommendations.append({
                "title": row['Title'],
                "author": row['Author'],
                "rating": row['average_rating'],
                "genres": str(row['genres']).replace(';', ', ')
            })
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": "Book not found or error in processing.", "details": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
