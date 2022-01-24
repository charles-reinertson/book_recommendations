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
        df_books, df_ratings, df_users = self.read_data()
        df_books, df_ratings, df_users = self.clean_data(df_books, df_ratings, df_users)
        self.join_data(df_books, df_ratings, df_users)
    
    @staticmethod
    def read_data():
        """
        Read in all csv files into a pandas dataframe and return dataframes.
        """
        df_books = pd.read_csv(config('csv_file_books'))
        df_ratings = pd.read_csv(config('csv_file_ratings'))
        df_users = pd.read_csv(config('csv_file_users')) 

        return df_books, df_ratings, df_users
    
    @staticmethod
    def clean_data(df_books, df_ratings, df_users):
        # drop rows with any missing data
        df_books = df_books.dropna()
        # drop the column age because there is 40% missing data
        df_users.drop('Age', axis=1, inplace=True)

        # remove non-numeric items from the column "Year-Of-Publication"
        df_books = df_books[pd.to_numeric(df_books["Year-Of-Publication"], errors='coerce').notnull()]
        # change "Year-Of-Publication" to numeric
        df_books["Year-Of-Publication"] = pd.to_numeric(df_books["Year-Of-Publication"])
        # only keep rows of dataframe where "Year-Of-Publication" is between 1900 and 2022
        df_books = df_books.loc[(df_books["Year-Of-Publication"] >= 1900) & 
                                (df_books["Year-Of-Publication"] <= 2022)]
        
        return df_books, df_ratings, df_users

    def join_data(self, df_books, df_ratings, df_users):

        # Inner join df_book and df_ratings on ISBN
        self.df = pd.merge(df_books, df_ratings, how="inner", on=["ISBN"])
        # Inner join df_users and the new dataframe on "User-ID"
        self.df = pd.merge(self.df, df_users, how="inner", on=["User-ID"])
        self.df.reset_index(drop=True, inplace=True)


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[[idx]]


def process_data():
    """
    Read in, join, and clean all relevant book data for knn.
    Return a pandas dataframe of relevant book data.
    """
    data = BookDataset()

    bob = len(data)
    item = data[2]
    bob

