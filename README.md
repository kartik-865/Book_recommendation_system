# 📚 Book Recommendation System

A complete **Content-Based Book Recommendation System** that helps users discover what to read next. The system analyzes book metadata (like genres, authors, publishers, and publication dates) using Natural Language Processing (NLP) techniques to find and recommend similar books.

## ✨ Features

- **Machine Learning Core**: Uses `CountVectorizer` and `cosine_similarity` to calculate the mathematical closeness between different books based on their tags.
- **FastAPI Backend**: A blazing-fast Python server that computes similarities and exposes search and recommendation endpoints.
- **Dynamic Frontend**: A beautiful, humanized, and responsive user interface built with Vanilla HTML/CSS/JS, featuring autocomplete search and micro-animations.
- **Evaluation Engine**: A custom unsupervised evaluation script that calculates Diversity, Catalog Coverage, Personalization, and Novelty to ensure high-quality recommendations.

## 📊 Dataset

The original dataset used to train this model was sourced from Kaggle. 
It contains metadata (Title, Author, Genres, Rating, etc.) for thousands of Goodreads books. 
- **Dataset Link**: [Goodreads Books Dataset on Kaggle](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) *(Note: Ensure the dataset contains the necessary columns like `genres` and `ratings_count` as expected by the script)*.

## 🛠️ Tech Stack

- **Data Science**: Python, Pandas, NumPy, Scikit-Learn
- **Backend API**: FastAPI, Uvicorn
- **Frontend**: HTML5, Vanilla CSS3, Vanilla JavaScript
- **Visualization**: Matplotlib, Seaborn

## 🚀 How to Run the Project

### 1. Start the Backend API
1. Open your Anaconda Prompt (or any Python terminal).
2. Navigate to the `backend` directory:
   ```bash
   cd Book-Recommendation-System/backend
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn main:app --reload
   ```
   *The API will start running locally at `http://localhost:8000`.*

### 2. View the Frontend
There is no complex frontend build step required! 
Simply go to the `Book-Recommendation-System/frontend` folder and double-click the **`index.html`** file to open it in your web browser. You can immediately start searching for books and viewing recommendations.

### 3. Generate Evaluation Metrics
To see how well the recommendation engine is performing across the catalog, you can run the evaluation script:
1. In your Anaconda Prompt, navigate to the `backend` directory:
   ```bash
   cd Book-Recommendation-System/backend
   ```
2. Run the evaluation script:
   ```bash
   python evaluate_model.py
   ```
3. This will generate two files in the `backend` folder:
   - `evaluation_metrics.json`: The raw numerical scores.
   - `evaluation_metrics.png`: A beautiful bar chart visualizing the model's Diversity, Coverage, Personalization, and Novelty.

## 📓 Jupyter Notebooks
If you want to explore the data preprocessing steps and the logic behind the recommendation algorithm step-by-step, you can open the `Book-Recommender-System-Content-Based.ipynb` file inside the `Book-Recommendation-System` folder using Jupyter Notebook or JupyterLab.