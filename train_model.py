import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def train_model():
    print("Loading dataset...")
    df = pd.read_csv('movie_dataset.csv')
    
    # Selecting relevant features
    features = ['genres', 'keywords', 'cast', 'director', 'tagline']
    
    # Filling null values
    for feature in features:
        df[feature] = df[feature].fillna('')
    
    def combine_features(row):
        return row['genres'] + " " + row['keywords'] + " " + row['cast'] + " " + row['director'] + " " + row['tagline']
    
    print("Pre-processing data...")
    df['combined_features'] = df.apply(combine_features, axis=1)
    
    # Vectorizing the text
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['combined_features'])
    
    # Selecting essential columns for the web app to save memory
    # We mainly need title and id (and index to map similarity)
    movie_list = df[['index', 'title', 'id', 'popularity', 'vote_average']].copy()
    
    # Save the processed data and the sparse count matrix
    # The sparse matrix takes tremendously less space than the N*N dense similarity matrix
    print("Saving model files...")
    pickle.dump(movie_list.to_dict(), open('movie_list.pkl', 'wb'))
    pickle.dump(count_matrix, open('count_matrix.pkl', 'wb'))
    
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
