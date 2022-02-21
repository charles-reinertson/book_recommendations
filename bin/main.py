import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import _mypath
from book_recs import RSystem


if __name__ == '__main__':
    x_columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 
                'Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'User-ID', 'Location']
    y_col = 'Book-Rating'
    # Instantiate a recommender system with specified features, predicted variable, and recommender type.
    recommender_system = RSystem(x_columns, y_col, clean_data=True, recommender_type='KNN')
    # add data at the specified data location. Must be formatted like original input data
    recommender_system.add_data(clean_data=True)
    # get recommendations for a specific user
    recommendations = recommender_system.get_recommendations(user_id='5', num_recommendations=10)

    print(recommendations)