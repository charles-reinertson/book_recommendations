import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import sys
import pandas as pd
from utils import config
from sklearn.decomposition import TruncatedSVD, NMF
import warnings


class KNN():
    def __init__(self, bookData):
        self.data = bookData
        self.knn = None
        self.nmf = None
        self.nmf_X = None
        
    def knn_fit(self, neighbors=5):
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        # will need to be updated using function from BookDataset
        X = self.data.pd.loc[:, self.df.columns != 'Book-Rating']
        y = self.data.pd.loc[:, 'Book-Rating']
        neigh.fit(X, y)
        self.knn = neigh
    
    def knn_predict(self, X):
        return self.knn.predict(X)

class Matrix_Factorization():
    def __init__(self, bookData):
        self.data = bookData
        
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
        
        
        
        
        
        
        
        
        
        
        
        

    