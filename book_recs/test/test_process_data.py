import pandas
import _mypath
import pandas as pd
from process_data import BookDataset
import pytest

@pytest.fixture
def book_data_clean():
    '''Returns a BookDataset instance with clean data'''
    return BookDataset(clean_data=True)

@pytest.fixture
def book_data_unclean():
    '''Returns a BookDataset instance with clean data'''
    return BookDataset(clean_data=False)

@pytest.fixture
def book_data_new_append():
    '''
    Returns a BookDataset instance with clean data to append
    to existing data.
    '''
    return BookDataset(new_data=True)


def test_bookdataset_clean(book_data_clean, book_data_unclean):
    assert book_data_clean.df.shape[1] < book_data_unclean.df.shape[1]
    book_data_clean.df["Year-Of-Publication"] = pd.to_numeric(book_data_clean.df["Year-Of-Publication"])
    count_errors = book_data_clean.df.loc[(book_data_clean.df["Year-Of-Publication"] < 1900)].shape[0]
    assert count_errors == 0

    count_errors = book_data_clean.df.loc[(book_data_clean.df["Year-Of-Publication"] > 2022)].shape[0]
    assert count_errors == 0

def test_bookdataset_unclean(book_data_unclean):
    book_data_unclean.df["Year-Of-Publication"] = pd.to_numeric(book_data_unclean.df["Year-Of-Publication"])
    count_errors = book_data_unclean.df.loc[(book_data_unclean.df["Year-Of-Publication"] < 1900)].shape[0]
    assert count_errors > 0

    count_errors = book_data_unclean.df.loc[(book_data_unclean.df["Year-Of-Publication"] > 2022)].shape[0]
    assert count_errors > 0

# def test_bookdataset_new_data_append(book_data_new_append):
#     pass

    
data_object = BookDataset()
print(data_object)