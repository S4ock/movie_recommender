import numpy as np
import pandas as pd
import ast
import nltk
import pickle
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

# Merging the 2 datasets
movies = movies.merge(credit, on='title')

# Deleting the unnecessary columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Getting rid of extra parentheses
def convert(text):
    l = []
    for i in ast.literal_eval(text):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

def convert_cast(text):
    l = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            l.append(i['name'])
        counter += 1
    return l

movies['cast'] = movies['cast'].apply(convert_cast)

def fetch_director(text):
    l = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

def remove_space(word):
    l = []
    for i in word:
        l.append(i.replace(" ", ""))
    return l

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()

def stems(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stems)

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)

recommend("Spectre")
pickle.dump(new_df,open('artifacts/movie_list.pkl','wb'))
pickle.dump(similarity,open('artifacts/similarity.pkl','wb'))