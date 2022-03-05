import _mypath
from utils import config
import pytest

def test_config():
    assert 'data/Books.csv' == config('csv_file_books')
    assert 'data/Ratings.csv' == config('csv_file_ratings')
    assert 'data/Users.csv' == config('csv_file_users')
    assert 'new_data/Books.csv' == config('csv_file_new_books') 
    assert 'new_data/Ratings.csv' == config('csv_file_new_ratings')
    assert 'new_data/Users.csv' == config('csv_file_new_users')
