import _mypath
from process_data import BookDataset
from recommender import Matrix_Factorization
import pytest

@pytest.fixture
def book_data():
    '''Returns a BookDataset instance with clean data'''
    return BookDataset(clean_data=True)

@pytest.fixture
def nmf():
    '''Returns an NMF instance'''
    return Matrix_Factorization(bookData=book_data())

@pytest.fixture
def nmf_fit():
    '''Returns a fitted NMF instance'''
    return Matrix_Factorization(bookData=book_data()).fit()

@pytest.fixture
def isbn_correct():
    '''Returns a string of an ISBN in the dataset with more than 10 reviews'''
    return '0451202856'

@pytest.fixture
def isbn_incorrect():
    '''Returns a string of an ISBN in the dataset with less than 10 reviews'''
    return '0609804618'


def test_nmf_filtered(nmf):
    assert min(nmf.data['User-ID'].value_counts()) >= 200
    assert min(nmf.data['ISBN'].value_counts()) >= 10

def test_nmf_fit_attributes_exist(nmf_fit):
    assert nmf_fit.data_pivot != None
    assert nmf_fit.model != None

def test_predict_does_not_raise_error(nmf_fit, isbn_correct):
    try:
        nmf_fit.predict(isbn_correct, 5)
    except KeyError as exc:
        assert False, f"ISBN '0451202856' raised an exception {exc}"
        
def test_predict_raises_error(nmf_fit, isbn_incorrect):
    with pytest.raises(KeyError):
        nmf_fit.predict(isbn_incorrect, 5)

def test_predict_num_recs(nmf_fit, isbn_correct):
    assert len(nmf_fit.predict(isbn_correct, 5)) == 5
    assert len(nmf_fit.predict(isbn_correct, 10)) == 10
    assert len(nmf_fit.predict(isbn_correct, 2)) == 2
    assert len(nmf_fit.predict(isbn_correct, 50)) == 50
        