# Book Recommendation System

Deployment link : https://book-recommendation-system-six-theta.vercel.app/

A content-based book recommendation system that helps users discover their next read. The system analyzes book metadata using Natural Language Processing techniques to compute similarity between titles and surface relevant recommendations through a clean, interactive web interface backed by a FastAPI server.

---

## Overview

The Book Recommendation System is a full-stack machine learning application that takes a book title as input and returns a ranked list of similar books based on content similarity. Unlike collaborative filtering approaches that require user interaction history, this system relies entirely on book metadata — including genre, author, publisher, and ratings — making it effective even without prior user behavior data.

The project is composed of three parts: a data science pipeline built in a Jupyter Notebook, a FastAPI backend that serves the recommendation engine, and a plain HTML/CSS/JavaScript frontend that provides a user-friendly search interface.

---

## Features

- Content-based recommendation using `CountVectorizer` and cosine similarity
- FastAPI backend with search and recommendation endpoints
- Autocomplete-enabled search interface with a responsive frontend
- Custom unsupervised evaluation engine measuring Diversity, Catalog Coverage, Personalization, and Novelty
- Step-by-step exploratory Jupyter Notebook documenting the data preprocessing and model logic
- No build tools or bundlers required for the frontend — open `index.html` directly in a browser

---

## Project Structure

```
Book_recommendation_system/
|
|-- Book-Recommendation-System/
|   |
|   |-- backend/
|   |   |-- main.py                        # FastAPI application entry point
|   |   |-- recommend.py                   # Recommendation logic and similarity computation
|   |   |-- evaluate_model.py              # Unsupervised evaluation script
|   |   |-- requirements.txt               # Python dependencies
|   |   |-- evaluation_metrics.json        # Output: raw evaluation scores (generated)
|   |   |-- evaluation_metrics.png         # Output: evaluation bar chart (generated)
|   |
|   |-- frontend/
|   |   |-- index.html                     # Main UI entry point
|   |   |-- style.css                      # Stylesheet
|   |   |-- script.js                      # Frontend logic and API calls
|   |
|   |-- Book-Recommender-System-Content-Based.ipynb   # Jupyter Notebook
|
|-- LICENSE                                # MIT License
|-- README.md                              # Project documentation
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Science | Python, Pandas, NumPy, Scikit-Learn |
| NLP | CountVectorizer, Cosine Similarity |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter Notebook / JupyterLab |

---

**Dataset**: [Goodreads Books Dataset on Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks)

Key columns used by the system include:

| Column | Description |
|---|---|
| `title` | Book title |
| `authors` | Author name(s) |
| `genres` | Genre tags associated with the book |
| `average_rating` | Mean reader rating on Goodreads |
| `ratings_count` | Total number of ratings received |
| `publisher` | Publishing house |

## How It Works

The recommendation engine follows a standard content-based filtering pipeline:

1. **Tag Construction**: For each book, relevant metadata fields (genre, author, publisher) are combined into a single text representation called a "tag."

2. **Vectorization**: The tags are transformed into numerical vectors using `CountVectorizer` from Scikit-Learn, which builds a vocabulary from all tokens across the corpus and represents each book as a sparse word-count vector.

3. **Similarity Computation**: Cosine similarity is computed between all pairs of book vectors. Cosine similarity measures the angle between two vectors in high-dimensional space, making it effective for sparse text representations regardless of document length.

4. **Recommendation Retrieval**: Given a query book title, the system retrieves the top-N most similar books by ranking all other books by their cosine similarity score against the query.

5. **API Serving**: The FastAPI backend loads the precomputed similarity matrix and exposes REST endpoints for search and recommendation queries, which the frontend consumes.

---

## Evaluation Metrics

Since this is an unsupervised recommendation system with no explicit user feedback, a custom evaluation framework is used to assess recommendation quality across four dimensions:

| Metric | Description |
|---|---|
| **Diversity** | Measures how different the recommended books are from one another within a single recommendation list. Higher diversity indicates the system avoids recommending overly similar titles. |
| **Catalog Coverage** | The proportion of the total book catalog that appears in recommendation lists across all queries. Higher coverage means more of the catalog is being surfaced. |
| **Personalization** | Measures how different recommendation lists are across different query books. A high personalization score means the system tailors results to the specific input rather than returning a generic popular list. |
| **Novelty** | Assesses how unexpected or non-obvious the recommendations are, typically penalizing very popular books that would appear regardless of the query. |

Running `evaluate_model.py` produces both a JSON file with raw scores and a bar chart PNG for visual inspection.

---


## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software with attribution.

See the [LICENSE](./LICENSE) file for full details.

---
