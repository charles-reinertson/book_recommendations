import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from utils import config

class BookDataset():
    def __init__(self, clean_data=True, new_data=False):
        """
        Read df_books, df_ratings and df_users. Clean df_books, df_ratings and df_users,
        as well as join them on the appropriate features.
        """
        df_books, df_ratings, df_users = self._read_data(new_data)
        # clean the dataframes if the flag is specified
        if clean_data:
            self._clean_data(df_books, df_ratings, df_users)
        # join df_books, df_ratings and df_users
        self.df = self._join_data(df_books, df_ratings, df_users)


    def _read_data(self, new_data):
        """
        Read in all csv files into a pandas dataframe and return dataframes.
        """
        if not new_data:
            df_books = pd.read_csv(config('csv_file_books'))
            df_ratings = pd.read_csv(config('csv_file_ratings'))
            df_users = pd.read_csv(config('csv_file_users')) 
        else:
            df_books = pd.read_csv(config('csv_file_new_books'))
            df_ratings = pd.read_csv(config('csv_file_new_ratings'))
            df_users = pd.read_csv(config('csv_file_new_users')) 

        return df_books, df_ratings, df_users
    
    def _clean_data(self, df_books, df_ratings, df_users):
        """
        Clean self.df_books, self.df_ratings and self.df_users after their data has been read in by the read_data function
        """
        # self.validate_proper_usage()
        # drop the column age because there is 40% missing data
        df_users.drop('Age', axis=1, inplace=True)
        # drop rows with any missing data
        df_books = df_books.dropna()

        # remove non-numeric items from the column "Year-Of-Publication"
        df_books = df_books[pd.to_numeric(df_books["Year-Of-Publication"], errors='coerce').notnull()]
        # change "Year-Of-Publication" to numeric
        df_books["Year-Of-Publication"] = pd.to_numeric(df_books["Year-Of-Publication"])
        # only keep rows of dataframe where "Year-Of-Publication" is between 1900 and 2022
        df_books = df_books.loc[(df_books["Year-Of-Publication"] >= 1900) & 
                                    (df_books["Year-Of-Publication"] <= 2022)]
        

    def get_dataframe(self):
        """
        Return the merged dataframe.
        """
        return self.df

    def get_x_y(self, columns_x, column_y):
        """
        Return two dataframes. The first of specified x columns and 
        the second dataframe as a single y column for prediction.

        columns_x: a list of strings of desired columns
        column_y: a string of column to predict
        """
        return self.df[columns_x], self.df[column_y]

    @staticmethod
    def _join_data(df_books, df_ratings, df_users):
        """
        Join df_books, df_ratings and df_users and return
        the resulting dataframe.
        """
        # Inner join df_book and df_ratings on ISBN
        df = pd.merge(df_books, df_ratings, how="inner", on=["ISBN"])
        # Inner join df_users and the new dataframe on "User-ID"
        df = pd.merge(df, df_users, how="inner", on=["User-ID"])
        df.reset_index(drop=True, inplace=True)

        # drop incorrect year of publication in df_books
        df = df[df['Year-Of-Publication'] != 'Gallimard']
        df = df[df['Year-Of-Publication'] != 'DK Publishing Inc']

        return df

    def append_data(self, clean_data=True):
        """
        Read in, clean (depending on flag), and merge newly added data.
        Append it to self.df
        """
        df_books, df_ratings, df_users = self._read_data(new_data=True)
        # clean the dataframes if the flag is specified
        if clean_data:
            self._clean_data(df_books, df_ratings, df_users)
        # join df_books, df_ratings and df_users
        df = self._join_data(df_books, df_ratings, df_users)

        self.df.append(df, ignore_index=True)
    
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


