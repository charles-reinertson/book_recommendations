import numpy as np
import os
import sys
import pandas as pd
import _mypath
from book_recs import RSystem


if __name__ == '__main__':

    # OPTION 1 - RECOMMENDER SYSTEM ON SIMILAR USERS

    # Instantiate a collaborative recommender system with specified features.
    rec_system = RSystem(clean_data=True, recommender_type='similar_user')
    # add data at the specified location.
    # Must be formatted like original input data.
    rec_system.add_data(clean_data=True)
    # get recommendations for a specific user (also 11676)
    recommendations = rec_system.get_recommendations(user_input=16795,
                                                     num_recommendations=5)

    print(recommendations)

    # OPTION 2 - RECOMMENDER SYSTEM ON SIMILAR BOOKS

    # Instantiate a content-based recommender system with specified features.
    rec_system = RSystem(clean_data=True, recommender_type='similar_book')
    # Add data at the specified location.
    # Must be formatted like original  input data.
    rec_system.add_data(clean_data=True)
    # get recommendations for a specific user:
    # (here at examples ['0971880107', '0316666343', '0385504209'])
    recommendations = rec_system.get_recommendations('0971880107',
                                                     num_recommendations=5)

    print(recommendations)
