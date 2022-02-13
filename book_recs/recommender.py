import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import os
import sys
import pandas as pd
from utils import config



class KNN():
    def __init__(self, bookData):
        self.data = bookData
        self.knn = None
        
    def knn_fit(self, neighbors=5):
        neigh = KNeighborsClassifier(n_neighbors=neighbors)
        # will need to be updated using function from BookDataset
        X = self.data.pd.loc[:, self.df.columns != 'Book-Rating']
        y = self.data.pd.loc[:, 'Book-Rating']
        neigh.fit(X, y)
        self.knn = neigh
    
    def knn_predict(self, X):
        return self.knn.predict(X)
