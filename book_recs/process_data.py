import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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

        # Inner join df_book and df_ratings on ISBN
        self.df = pd.merge(self.df_books, self.df_ratings, how="inner", on=["ISBN"])
        # Inner join df_users and the new dataframe on "User-ID"
        self.df = pd.merge(self.df, self.df_users, how="inner", on=["User-ID"])
        self.df.reset_index(drop=True, inplace=True)

        return self


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx]


