import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from utils import config

class BookDataset():
    def __init__(self):
        """
        Read df_books, df_ratings and df_users, clean df_books, df_ratings and df_users,
        as well as join them on the appropriate features.
        """
        self.df = None
        self.df_books = None
        self.df_ratings = None
        self.df_users = None
        
    
    def read_data(self):
        """
        Read in all csv files into a pandas dataframe and return dataframes.
        """
        self.df_books = pd.read_csv(config('csv_file_books'))
        self.df_ratings = pd.read_csv(config('csv_file_ratings'))
        self.df_users = pd.read_csv(config('csv_file_users')) 

        return self
    
    def clean_data(self):
        """
        Clean self.df_books, self.df_ratings and self.df_users after their data has been read in by the read_data function
        """
        self.validate_proper_usage()
        # drop the column age because there is 40% missing data
        self.df_users.drop('Age', axis=1, inplace=True)
        # drop rows with any missing data
        self.df_books = self.df_books.dropna()

        # remove non-numeric items from the column "Year-Of-Publication"
        self.df_books = self.df_books[pd.to_numeric(self.df_books["Year-Of-Publication"], errors='coerce').notnull()]
        # change "Year-Of-Publication" to numeric
        self.df_books["Year-Of-Publication"] = pd.to_numeric(self.df_books["Year-Of-Publication"])
        # only keep rows of dataframe where "Year-Of-Publication" is between 1900 and 2022
        self.df_books = self.df_books.loc[(self.df_books["Year-Of-Publication"] >= 1900) & 
                                (self.df_books["Year-Of-Publication"] <= 2022)]
        
        return self

    def join_data(self):
        """
        Join self.df_books, self.df_ratings and self.df_users.
        """
        self.validate_proper_usage()
        # Inner join df_book and df_ratings on ISBN
        self.df = pd.merge(self.df_books, self.df_ratings, how="inner", on=["ISBN"])
        # Inner join df_users and the new dataframe on "User-ID"
        self.df = pd.merge(self.df, self.df_users, how="inner", on=["User-ID"])
        self.df.reset_index(drop=True, inplace=True)
        self._save_memory()

        return self
    
    def simple_split_data(self, size=0.2):
        """
        Perform a simple train-test split on the joined data. Return split data.
        """
        return train_test_split(self.df, test_size=size, random_state=42)
    
    def KFold_split_data(self, k=10):
        """
        Perform k-fold cross validation on the joined data. Return split data.
        """
        kf= KFold(n_splits=k, shuffle= False)
        result = next(kf.split(self.df), None)
        train= self.df.iloc[result[0]]
        test= self.df.iloc[result[1]]
        return train, test
        
    
    def validate_proper_usage(self):
        """
        Validate that the user has read in the dataframes before calling clean_data or join_data.
        """
        # Check that self.df_books, self.df_ratings and self.df_users are not null, else throw error
        if (self.df_books or self.df_ratings or self.df_users) is None:
            sys.exit("Must read data first using the function 'read_data'.") 
    
    def _save_memory(self):
        """
        Remove self.df_books, self.df_ratings and self.df_users after they have been merged.
        """
        self.df_books = None
        self.df_ratings = None
        self.df_users = None


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx]


