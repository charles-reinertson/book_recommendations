import _mypath
from process_data import BookDataset
from recommender import KNN, System
import pytest

@pytest.fixture
def book_data():
    '''Returns a BookDataset instance with clean data'''
    return BookDataset(clean_data=True)

@pytest.fixture
def knn(book_data):
    '''Returns a KNN instance'''
    return KNN(bookData=book_data)

@pytest.fixture
def knn_fit(book_data):
    '''Returns a fitted KNN instance'''
    return KNN(bookData=book_data).fit()

@pytest.fixture
def isbn_correct():
    '''Returns a string of an ISBN in the dataset with more than 10 reviews'''
    return '0451202856'

@pytest.fixture
def isbn_incorrect():
    '''Returns a string of an ISBN in the dataset with less than 10 reviews'''
    return '0609804618'



assert min(KNN(bookData=BookDataset(clean_data=True)).data['User-ID'].value_counts()) >= 200
assert min(KNN(bookData=book_data).data['ISBN'].value_counts()) >= 10

def test_knn_fit_attributes_exist(knn_fit):
    assert knn_fit.data_pivot != None
    assert knn_fit.model != None

def test_predict_does_not_raise_error(knn_fit, isbn_correct):
    try:
        knn_fit.predict(isbn_correct, 5)
    except KeyError as exc:
        assert False, f"ISBN '0451202856' raised an exception {exc}"
        
def test_predict_raises_error(knn_fit, isbn_incorrect):
    with pytest.raises(KeyError):
        knn_fit.predict(isbn_incorrect, 5)

def test_predict_num_recs(knn_fit, isbn_correct):
    assert len(knn_fit.predict(isbn_correct, 5)) == 5
    assert len(knn_fit.predict(isbn_correct, 10)) == 10
    assert len(knn_fit.predict(isbn_correct, 2)) == 2
    assert len(knn_fit.predict(isbn_correct, 50)) == 50
        