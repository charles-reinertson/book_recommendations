import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import _mypath
import book_recs

def process_data():
    """
    Read in, join, and clean all relevant book data for knn.
    Return a pandas dataframe of relevant book data.
    """
    bookData = book_recs.BookDataset()
    return bookData.read_data().clean_data().join_data()

if __name__ == '__main__':
    # STEP 1: PROCESS AND CLEAN THE DATA
    data_object = process_data()

    x_columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 
                'Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'User-ID', 'Location']

    x_df, y_df = data_object.get_x_y(x_columns, 'Book-Rating')

    print(x_df.head())
    print(y_df.head())