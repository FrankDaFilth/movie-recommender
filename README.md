

# 🎬 Movie Recommender System

A content-based movie recommendation system that suggests similar movies based on textual descriptions (overviews). Built using Python and the TMDB 5000 Movie Dataset.

## 📦 Overview

- **Goal**: Recommend movies similar to a given title using plot overviews
- **Dataset**: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- **Tech Stack**: Python, pandas, scikit-learn (TF-IDF + cosine similarity)

## 🔍 How It Works

- Loads movie titles and descriptions
- Converts overviews into TF-IDF vectors
- Computes similarity scores between movies
- Returns the top N most similar movies

## 🧠 Example

```python
recommend('The Dark Knight')
```

**Output:**
- The Dark Knight Rises
- Batman Returns
- Batman Begins
- Batman Forever
- Batman & Robin
...

## 🗃️ Project Structure

```
movie-recommender/
├── data/               # Contains movies.csv
├── output/             # Placeholder for any future visualizations
├── src/
│   └── recommend.py    # Main recommendation logic
├── README.md
```

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas scikit-learn
```

2. Download the dataset and place `movies.csv` in the `data/` folder.

3. Run the script:

```bash
python src/recommend.py
```

## 📈 Future Enhancements

- Add genre or keyword filters
- Build a collaborative filtering engine (in progress)
- Create a web interface using Streamlit

## 🙌 Author

Francisco Nunez — developed as part of a portfolio data science project sprint.