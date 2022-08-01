# Simple Recommender System

import pandas as pd

movies = pd.read_csv("movies.csv")
print(movies.info())
print(movies.head())

ratings = pd.read_csv("ratings.csv")
print(ratings.info())
print(ratings.head())

movies_data = pd.read_csv("movies_metadata.csv")
print(movies_data.head())

"""
Weighted Rating - 

(v/v+m)*R + (m/v+m)*C

v is the number of votes (vote_count)
m is the minimum votes required to be listed in chart
R is the average rating for the movie (vote_average)
C is the mean vote across the whole report
"""

C = movies_data["vote_average"].mean()
print(C)

m = movies_data["vote_count"].quantile(0.90)
print(m)

# Filter out all the qualified movies into a new DataFrame

q_movies = movies_data.copy().loc[movies_data["vote_count"] >= m]
print(q_movies.shape)

def weighted_rating(x, m = m, C = C):
  v = x["vote_count"]
  R = x["vote_average"]

  return (v/(v+m)*R) + (m/(m+v)*C)

q_movies["score"] = q_movies.apply(weighted_rating, axis = 1)
q_movies = q_movies.sort_values('score', ascending = False)

# Printing the Top 20 movies 
print(q_movies[["title", "vote_count", "vote_average", "score"]].head(20))
