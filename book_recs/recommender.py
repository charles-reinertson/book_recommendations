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

class System():
    def __init__(self, bookData):
        self.data = bookData
        self._filter_data()

    def fit(self):
        "Interface for fitting the recommender system"
        pass

    def predict(self, user_input, num_recommendations):
        "Interface for predicting from the recommender system"
        pass


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

class KNN(System):
    def __init__(self, bookData):
        """
        bookData: BookDataset object (gets changed to a dataframe object)
        """
        super().__init__(bookData)

        self.data_pivot = None
        self.model = None
        
    def fit(self):
        """
        Fit the KNN model to the dataset.
        """
        self.book_pivot = self.data.pivot_table(columns='User-ID', index='ISBN', values="Book-Rating")
        self.book_pivot.fillna(0, inplace=True)
        book_sparse = csr_matrix(self.book_pivot)
        self.model = NearestNeighbors(algorithm='brute')
        self.model.fit(book_sparse)
    
    def predict(self, book, num_recommendations):
        """
        Get the book recommendations based on a provided book.

        book: String of the ISBN of the book to base recommendations on
        num_recommendations: Number of recommended book to return (not implemented)
        
        recommend: Dataframe of recommended book titles, ISBN, book author, and year of publication
        """
        distances, suggestions = self.model.kneighbors(self.book_pivot.loc[book, :].values.reshape(1, -1))
        recommend = self.data[self.data['ISBN'].isin(self.book_pivot.index[suggestions[0]].values)].drop_duplicates(['ISBN'])
        recommend = recommend.loc[:, ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication']]
        return recommend

class Matrix_Factorization(System):
    
    def __init__(self, bookData):
        """

        Parameters
        ----------
        bookData : TYPE
        
        Description
        ----------
        Initialize with bookDataset object.

        Returns
        -------
        None.

        """
        super().__init__(bookData)

        self.nmf = None
        self.nmf_X = None
        self.user_id_to_num_dict = {}
        self.book_title = None
        
        
    def fit(self):
        """
        Description
        ----------
        Clean data, convert to sparse matrix of User-ID, Book-Title and Book-Rating. Fit a model on this data.

        Returns
        -------
        None.

        """
        df_mat = self.data.sample(frac = 0.3)
        # df_mat= df_mat.drop(columns= ['Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'Location'])
        # df_mat = df_mat.dropna(axis = 0, subset = ['Book-Title'])
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
        # SVD = TruncatedSVD(n_components=12, random_state=17)
        # matrix = SVD.fit_transform(X)
        nmf_model = NMF(n_components=20)
        self.nmf = nmf_model
        self.nmf_X = X 
        self.nmf.fit(self.nmf_X)
        
    def predict(self, user_idx, num_recommendations):
        """
        

        Parameters
        ----------
        user_idx : int
            Input User-ID to get recommendations.
        num_recommendations : int
            Number of recommendations returned to the user.
            
        Description
        ----------
        Predict recommendations for a user.
        
        Returns
        -------
        bookrecs: list
            List of generated recommendations.

        """
        Theta = self.nmf.transform(self.nmf_X)       
        M = self.nmf.components_.T         
        X_pred = M.dot(Theta.T)             
        X_pred = X_pred.T
        
        rated_items_df_user = pd.DataFrame(self.nmf_X).iloc[self.user_id_to_num_dict[user_idx], :]                
        user_prediction_df_user = pd.DataFrame(X_pred).iloc[self.user_id_to_num_dict[user_idx],:]    
        reco_df = pd.concat([rated_items_df_user, user_prediction_df_user, pd.DataFrame(self.book_title)], axis=1)  
        reco_df.columns = ['rating','prediction','title']
        reco_df = reco_df[ reco_df['rating'] == 0 ]
        res= reco_df.sort_values(by='prediction', ascending=False)[:num_recommendations]
        return list(res['title'])

        
        
        
        
        
        
        
        
        
        
        
        
        

    
