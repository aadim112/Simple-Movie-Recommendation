import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🍿", layout="wide")

# Custom CSS for modern, premium look with dark theme accents
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Text */
    p, span, div, label {
        color: #cbd5e1 !important;
    }

    /* Cards for recommendations */
    .movie-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        transition: transform 0.2s;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.2);
        border-color: #38bdf8;
    }

    .movie-title {
        color: #38bdf8 !important;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 12px;
        line-height: 1.3;
    }

    .movie-metric {
        display: inline-block;
        background-color: #0f172a;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.875rem;
        margin-right: 8px;
        border: 1px solid #334155;
    }
    
    .metric-value {
        color: #e2e8f0 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Function to load data with caching to prevent reloading on every interaction
@st.cache_resource
def load_data():
    try:
        movies_dict = pickle.load(open('movie_list.pkl', 'rb'))
        movies = pd.DataFrame(movies_dict)
        count_matrix = pickle.load(open('count_matrix.pkl', 'rb'))
        return movies, count_matrix
    except FileNotFoundError:
        return None, None

movies, count_matrix = load_data()

def get_recommendations(movie_title, movies, count_matrix):
    try:
        # Find index of the movie
        movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
        
        # Calculate similarity dynamically
        similarity_scores = cosine_similarity(count_matrix[movie_index], count_matrix).flatten()
        
        # Get top 6 similar movies (excluding the movie itself)
        # enumerate gives (index, similarity)
        movie_list_indices = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])[1:7]
        
        recommended_movies = []
        for i in movie_list_indices:
            idx = i[0]
            movie_info = {
                'title': movies.iloc[idx].title,
                'popularity': float(movies.iloc[idx].popularity),
                'vote_average': float(movies.iloc[idx].vote_average)
            }
            recommended_movies.append(movie_info)
        return recommended_movies
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return []

# App Header
st.title("🍿 Content-Based Movie Recommender")
st.markdown("Discover movies similar to your favorites! It dynamically calculates recommendations instantly.")

if movies is not None and not movies.empty:
    st.markdown("---")
    
    # Search and selection
    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Search or select a movie from the list below",
        movie_list
    )

    if st.button('Show Recommendations', type='primary'):
        with st.spinner('Finding the best matches...'):
            recommendations = get_recommendations(selected_movie, movies, count_matrix)
            
            if recommendations:
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"Top 6 movies similar to **{selected_movie}**:")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display in grid format (2 rows, 3 columns)
                for i in range(0, 6, 3):
                    cols = st.columns(3)
                    for j in range(3):
                        if i + j < len(recommendations):
                            movie = recommendations[i + j]
                            with cols[j]:
                                st.markdown(f"""
                                <div class="movie-card">
                                    <div class="movie-title">{movie['title']}</div>
                                    <div>
                                        <span class="movie-metric">⭐ <span class="metric-value">{round(movie['vote_average'], 1)}</span></span>
                                        <span class="movie-metric">📈 <span class="metric-value">{round(movie['popularity'], 1)}</span></span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found.")
else:
    st.error("Data files not found. Please ensure 'movie_list.pkl' and 'count_matrix.pkl' exist in the directory.")
