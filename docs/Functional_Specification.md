# Functional Specifications

## Background
A recommendation engine is a type of machine learning which offers relevant suggestions to customers. It finds applications in various sectors such as e-commerce, media and entertainment (ex. Netflix, Amazon). 
There are two major approaches for recommendation systems: 
* Content-based: recommend items to a user that are similar to the ones the user preferred in the past
* Collaborative: predict users’ preferences by analyzing relationships between users and interdependencies among items and extrapolate new associations. 
Consequently, books are ideal candidates for recommendation system, by the nature of their data and usage. Although there are packages that exist for recommenders, there are few that are targeted towards specific applications, such as website development- web developers need simplified ML solutions. This package simplifies the process of creating a recommender system for books by integrating multiple algorithms and methods to create a uniform, easy interface for implementation for both content-based and collaborative recommendation systems, depending on the user’s preference.

## User profile
* Web developers
* Students
* Researchers

## Data sources

Kaggle: [Books dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv)

## Use cases
1. Content-based: get recommendations based on a book\
User: Input the ISBN and number of desired recommendations to the recommendation system. \
System: Instantiate a KNN recommendation system. Pre-process data, build a KNN model and return a dataframe of predictions of books similar to the ISBN of the input book.
2. Collaborative: get recommendations based on a user\
User: Input the User ID and number of desired recommendations to the recommendation system. \
System: Instantiate a Matrix Factorization recommendation system. Pre-process data, build a Matrix Factorization model and return a dataframe of predictions of books based on users similar to that of the input user.

