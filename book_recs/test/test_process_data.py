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
    # assert book_data_clean.df.shape[1] < book_data_unclean.df.shape[1]
    book = BookDataset(clean_data=True)
    count_errors = book_data_clean.df.loc[book_data_clean.df["Year-Of-Publication"] < 1900].shape[0]
    assert count_errors == 0

    count_errors = book_data_clean.df.loc[book_data_clean.df["Year-Of-Publication"] > 2022].shape[0]
    assert count_errors == 0

def test_bookdataset_unclean(book_data_unclean):
    book_data_unclean.df["Year-Of-Publication"] = pd.to_numeric(book_data_unclean.df["Year-Of-Publication"])
    count_errors = book_data_unclean.df.loc[(book_data_unclean.df["Year-Of-Publication"] < 1900)].shape[0]
    assert count_errors > 0

    count_errors = book_data_unclean.df.loc[(book_data_unclean.df["Year-Of-Publication"] > 2022)].shape[0]
    assert count_errors > 0

def test_bookdataset_new_data_append():
    book = BookDataset(new_data=True)
    size_reg = book.df.shape[0]
    book.append_data(clean_data=True)
    size_append = book.df.shape[0]
    assert size_reg < size_append

def test_simple_split_data(book_data_clean):

    x, y = book_data_clean.simple_split_data()
    assert x.shape[0] > y.shape[0]   

def test_KFold_split_data(book_data_clean):

    train, test = book_data_clean.KFold_split_data()
    assert train.shape[0] > test.shape[0]   

def test_get_dataframe(book_data_clean):
    assert isinstance(book_data_clean.get_dataframe(), pd.DataFrame)

def test_get_x_y(book_data_clean):
    x, y = book_data_clean.get_x_y(["Year-Of-Publication", "ISBN"], ["User-ID"])
    assert x.columns[0] == "Year-Of-Publication"
    assert x.columns[1] =="ISBN"
    assert y.columns[0] == "User-ID"