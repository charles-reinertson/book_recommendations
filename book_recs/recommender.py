from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import sys
import pandas as pd
from utils import config
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD, NMF
import warnings


class KNN():
    def __init__(self, bookData):
        self.data = bookData
        self.data_pivot = None
        self.model = None
        
    def knn_fit(self):
        """
        Fit the KNN model to the dataset.
        """
        self._filter_data()
        self.book_pivot = self.data.pivot_table(columns='User-ID', index='ISBN', values="Book-Rating")
        self.book_pivot.fillna(0, inplace=True)
        book_sparse = csr_matrix(self.book_pivot)
        self.model = NearestNeighbors(algorithm='brute')
        self.model.fit(book_sparse)
    
    def knn_predict(self, book, num_recommendations):
        """
        Get the book recommendations based on a provided book.

        book: Specify the ISBN of the book to base recommendations on
        num_recommendations: Number of recommended book to return
        """
        distances, suggestions = self.model.kneighbors(self.book_pivot.loc[book, :].values.reshape(1, -1))
        recommend = self.data[self.data['ISBN'].isin(self.book_pivot.index[suggestions[0]].values)].drop_duplicates(['ISBN'])
        recommend = recommend.loc[:, ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication']]
        return recommend
    
    def _filter_data(self):
        """
        Filter the data to only include users who have rated at least 200 books 
        and books that have at least 10 ratings.
        """
        cols = ['Book-Rating', 'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'User-ID']
        self.data = self.data.get_dataframe().loc[:, cols]
        users = self.data['User-ID'].value_counts() >= 200
        users_filt = users[users].index
        self.data = self.data[self.data['User-ID'].isin(users_filt)]
        books = self.data['ISBN'].value_counts() >= 10
        books_filt = books[books].index
        self.data = self.data[self.data['ISBN'].isin(books_filt)]

class Matrix_Factorization():
    
    def __init__(self, bookData):
        self.data = bookData
        self.nmf = None
        self.nmf_X = None
        self.user_id_to_num_dict = {}
        self.book_title = None
        
    def _filter_data(self):
        """
        Filter the data to only include users who have rated at least 200 books 
        and books that have at least 10 ratings.
        """
        users = self.data['User-ID'].value_counts() >= 200
        users_filt = users[users].index
        self.data = self.data[self.data['User-ID'].isin(users_filt)]
        books = self.data['ISBN'].value_counts() >= 10
        books_filt = books[books].index
        self.data = self.data[self.data['ISBN'].isin(books_filt)]   
        
    def  matrix_factorization_fit(self):
        
        self._filter_data()
        df_mat = self.data.sample(frac = 0.3)
        df_mat= df_mat.drop(columns= ['Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'Location'])
        df_mat = df_mat.dropna(axis = 0, subset = ['Book-Title'])
        df_mat_ratingCount = (df_mat.
             groupby(by = ['Book-Title'])['Book-Rating'].
             count().
             reset_index().
             rename(columns = {'Book-Rating': 'totalRatingCount'})
             [['Book-Title', 'totalRatingCount']]
            )
        rating_mat_totalRatingCount = df_mat.merge(df_mat_ratingCount, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
        user_rating = rating_mat_totalRatingCount.drop_duplicates(['User-ID','Book-Title'])
        book_user_rating_pivot = user_rating.pivot(index = 'User-ID', columns = 'Book-Title', values = 'Book-Rating').fillna(0)
        
        user_id_list = list(book_user_rating_pivot.index)
        user_id_to_num_dict = {}
        for i,item in enumerate(user_id_list):
            user_id_to_num_dict[item]= i
        self.user_id_to_num_dict = user_id_to_num_dict
        X = book_user_rating_pivot.values
        SVD = TruncatedSVD(n_components=12, random_state=17)
        matrix = SVD.fit_transform(X)
        nmf_model = NMF(n_components=20)
        self.nmf = nmf_model
        self.nmf_X = X #or should i return?
        self.nmf_model.fit(self.X)
        
    def matrix_factorization_predict(self, user_idx, num_recommendations):
        Theta = self.nmf_model.transform(self.nmf_X)       
        M = self.nmf_model.components_.T         
        X_pred = M.dot(Theta.T)             
        X_pred = X_pred.T
        
        rated_items_df_user = pd.DataFrame(self.nmf_X).iloc[self.user_id_to_num_dict[user_idx], :]                 # get the list of actual ratings of user_idx (seen movies)
        user_prediction_df_user = pd.DataFrame(X_pred).iloc[self.user_id_to_num_dict[user_idx],:]     # get the list of predicted ratings of user_idx (unseen movies)
        reco_df = pd.concat([rated_items_df_user, user_prediction_df_user, pd.DataFrame(self.book_title)], axis=1)   # merge both lists with the movie's title
        reco_df.columns = ['rating','prediction','title']
        reco_df = reco_df[ reco_df['rating'] == 0 ]
        res= reco_df.sort_values(by='prediction', ascending=False)[:num_recommendations]
        return list(res['title'])

        
        
        
        
        
        
        
        
        
        
        
        
        

    