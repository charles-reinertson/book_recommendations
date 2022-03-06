import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import _mypath
from book_recs import RSystem


if __name__ == '__main__':

    # OPTION 1 - RECOMMENDER SYSTEM ON SIMILAR USERS
    
    # Instantiate a recommender system with specified features, predicted variable, and recommender type.
    recommender_system = RSystem(clean_data=True, recommender_type='similar_user')
    # add data at the specified data location. Must be formatted like original input data
    recommender_system.add_data(clean_data=True)
    # get recommendations for a specific user (also 11676)
    recommendations = recommender_system.get_recommendations(16795, num_recommendations=10)

    print(recommendations)

    # OPTION 2 - RECOMMENDER SYSTEM ON SIMILAR BOOKS
    
    # # Instantiate a recommender system with specified features, predicted variable, and recommender type.
    # recommender_system = RSystem(clean_data=True, recommender_type='similar_book')
    # # add data at the specified data location. Must be formatted like original input data
    # recommender_system.add_data(clean_data=True)
    # # get recommendations for a specific user (here at examples ['0971880107', '0316666343', '0385504209', '0060928336', '0312195516'])
    # recommendations = recommender_system.get_recommendations('0971880107', num_recommendations=10)

    # print(recommendations)