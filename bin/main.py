import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import _mypath
from book_recs import RSystem


if __name__ == '__main__':

    # Instantiate a recommender system with specified features, predicted variable, and recommender type.
    recommender_system = RSystem(clean_data=True, recommender_type='similar_book')
    # add data at the specified data location. Must be formatted like original input data
    recommender_system.add_data(clean_data=True)
    # get recommendations for a specific user
    recommendations = recommender_system.get_recommendations('Classical Mythology', num_recommendations=10)

    print(recommendations)