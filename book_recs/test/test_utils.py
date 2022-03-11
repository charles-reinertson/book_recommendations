import _mypath
from utils import config
import pytest


def test_config_old_data():

    assert 'data/Books.csv' == config('csv_file_books')
    assert 'data/Ratings.csv' == config('csv_file_ratings')
    assert 'data/Users.csv' == config('csv_file_users')


def test_config_new_data():

    assert 'new_data/Books.csv' == config('csv_file_new_books')
    assert 'new_data/Ratings.csv' == config('csv_file_new_ratings')
    assert 'new_data/Users.csv' == config('csv_file_new_users')


def test_config_bad_input():

    worked = False
    try:
        config('bob')
    except KeyError:
        worked = True
        print('This worked as expected with a key error')

    assert worked is True
