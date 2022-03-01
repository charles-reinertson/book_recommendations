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
        
    def  matrix_factorization_fit(self, book_title_inp):
        df_mat = self.data.sample(frac = 0.01)
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
        # X = book_user_rating_pivot.values.T
        # SVD = TruncatedSVD(n_components=12, random_state=17)
        # matrix = SVD.fit_transform(X)
        # warnings.filterwarnings("ignore",category =RuntimeWarning)
        # corr = np.corrcoef(matrix)
        # book_title = book_user_rating_pivot.columns
        # book_title_list = list(book_title)
        # rec = book_title_list.index(book_title_inp)
        # corr_rec = corr[rec]
        # res = list(book_title[(corr_rec >= 0.9)])
        
        X = book_user_rating_pivot.values.T
        SVD = TruncatedSVD(n_components=12, random_state=17)
        matrix = SVD.fit_transform(X)
        nmf_model = NMF(n_components=20)
        self.nmf = nmf_model
        self.nmf_X = X #or should i return?
        
    def matrix_factorization_predict(self):
        Theta = self.nmf_model.transform(self.X)       
        M = self.nmf_model.components_.T         
        X_pred = M.dot(Theta.T)             
        X_pred = X_pred.T
        return X_pred
        
        
        
        
        
        
        
        
        
        
        
        

    