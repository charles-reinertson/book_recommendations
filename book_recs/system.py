import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier

from process_data import BookDataset
from recommender import KNN, Matrix_Factorization
from utils import config

class RSystem(KNN, Matrix_Factorization):
    def __init__(self, x_cols, y_col='Book-Rating', clean_data=True, recommender_type='KNN'):
        # read in, clean (depending on flag), and merge data
        self.data_object = BookDataset(clean_data)
        # get recommender type, either KNN or Matrix Factorization
        self.recommender_type = recommender_type
        # get the features and dependent variable
        self.x_df, self.y_df = self.data_object.get_x_y(x_cols, 'Book-Rating')
        # train our model
        self._train_system()



    def add_data(self, clean_data=True):
        """
        Add more data to the recommender system and retrain the model with this new data.
        """
        self.data_object.append_data(clean_data)
        self._train_system()

    def get_recommendations(self, user_id='5', num_recommendations=10):
        """
        Get the book recommendations for a specific user.

        user_id: Specify the desired user to recieve their recommendations
        num_recommendations: Number of recommended book to return
        """
        if self.recommender_type == 'KNN':
            predictions = self.knn_predict(user_id)
        else:
            pass 

        return predictions[:10]
    
    def _train_system(self):
        """
        Train our data on either KNN or Matrix factorization recommender system
        """
        if self.recommender_type == 'KNN':
            self.knn_fit()
        else:
            pass
            # self.matrix_fact_fit

