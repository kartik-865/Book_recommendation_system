# Book Recommendation System

A content-based book recommendation system that helps users discover their next read. The system analyzes book metadata using Natural Language Processing techniques to compute similarity between titles and surface relevant recommendations through a clean, interactive web interface backed by a FastAPI server.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Running the Project](#running-the-project)
- [Exploring the Notebook](#exploring-the-notebook)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

The Book Recommendation System is a full-stack machine learning application that takes a book title as input and returns a ranked list of similar books based on content similarity. Unlike collaborative filtering approaches that require user interaction history, this system relies entirely on book metadata — including genre, author, publisher, and ratings — making it effective even without prior user behavior data.

The project is composed of three parts: a data science pipeline built in a Jupyter Notebook, a FastAPI backend that serves the recommendation engine, and a plain HTML/CSS/JavaScript frontend that provides a user-friendly search interface.

---

## Motivation

The volume of published books available today makes manual discovery increasingly difficult. Generic bestseller lists and curated picks do not account for individual taste, and full collaborative filtering systems require large amounts of user interaction data that is rarely available for cold-start scenarios.

Content-based filtering offers a practical and interpretable alternative. By representing books as vectors derived from their metadata and computing cosine similarity between them, the system can reliably surface books that share thematic, stylistic, or authorial characteristics — without requiring any user history.

This project demonstrates how a complete, deployable recommendation system can be built from a publicly available dataset using standard machine learning tools and a lightweight web stack.

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

## Dataset

The dataset used to build and train the recommendation model is sourced from Kaggle and contains metadata for thousands of books listed on Goodreads.

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

Ensure the dataset file is placed in the appropriate directory as expected by the preprocessing and backend scripts before running the application.

---

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

## Requirements

- Python 3.8 or higher
- pip or Anaconda package manager

Python dependencies (install via `requirements.txt`):

- fastapi
- uvicorn
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Installation and Setup

1. **Clone the repository:**

```bash
git clone https://github.com/kartik-865/Book_recommendation_system.git
cd Book_recommendation_system
```

2. **Set up a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install backend dependencies:**

```bash
cd Book-Recommendation-System/backend
pip install -r requirements.txt
```

4. **Download the dataset** from the [Kaggle link above](#dataset) and place it in the expected directory as referenced by the preprocessing scripts.

---

## Running the Project

### Step 1 — Start the Backend API

Navigate to the backend directory and start the FastAPI server:

```bash
cd Book-Recommendation-System/backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. You can also explore the auto-generated interactive API documentation at `http://localhost:8000/docs`.

### Step 2 — Open the Frontend

No build step is required. Navigate to the `frontend` folder and open `index.html` directly in your web browser:

```
Book-Recommendation-System/frontend/index.html
```

The interface will connect to the locally running FastAPI server. You can type any book title into the search bar to receive recommendations instantly.

### Step 3 — Run the Evaluation Script

To generate quality metrics for the recommendation engine:

```bash
cd Book-Recommendation-System/backend
python evaluate_model.py
```

This produces two output files in the `backend/` directory:

- `evaluation_metrics.json` — raw numerical scores for all four metrics
- `evaluation_metrics.png` — a bar chart visualization of the scores

---

## Exploring the Notebook

The Jupyter Notebook (`Book-Recommender-System-Content-Based.ipynb`) documents the entire data science workflow in an interactive, step-by-step format. It covers:

- Loading and inspecting the raw dataset
- Cleaning and preprocessing book metadata
- Constructing tag representations for each book
- Applying CountVectorizer and computing the cosine similarity matrix
- Testing and visualizing recommendations for sample queries

To open the notebook:

```bash
cd Book-Recommendation-System
jupyter notebook Book-Recommender-System-Content-Based.ipynb
```

---

## Future Work

Several directions are available to extend and improve this project:

- **Collaborative Filtering**: Incorporate user rating data to build a hybrid model that combines content-based and collaborative signals for improved personalization
- **TF-IDF Vectorization**: Replace CountVectorizer with TF-IDF to down-weight overly common terms and improve the quality of tag representations
- **Advanced NLP**: Use sentence embeddings (e.g., via Sentence-BERT) instead of bag-of-words vectors to capture semantic meaning more accurately
- **User Profiles**: Add a session-based user profile that tracks previously viewed or liked books to refine recommendations over time
- **Containerization**: Package the backend with Docker for consistent and portable deployment across environments
- **Cloud Deployment**: Deploy the FastAPI backend to a cloud platform (e.g., Render, Railway, or AWS) and host the frontend on GitHub Pages for public access
- **Extended Dataset**: Incorporate additional metadata fields such as plot summaries or user reviews using text mining to enrich the tag representations

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit with descriptive messages
4. Push to your fork and open a Pull Request against the `main` branch

For substantial changes, please open an issue first to discuss the proposal before submitting a pull request.

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software with attribution.

See the [LICENSE](./LICENSE) file for full details.

---

## Acknowledgements

- [Goodreads Books Dataset on Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) for providing the book metadata used in this project
- The Scikit-Learn development team for the vectorization and similarity utilities that form the core of the recommendation engine
- The FastAPI project for providing a high-performance, easy-to-use Python web framework

---

*Developed by Kartik Gahlot. For questions or collaboration, please open an issue on this repository.*
