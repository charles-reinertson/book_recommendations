import _mypath
from process_data import BookDataset
from recommender import KNN, System
import pytest

@pytest.fixture
def book_data():
    '''Returns a BookDataset instance with clean data'''
    return BookDataset(clean_data=True)
book_data = BookDataset(clean_data=True)

@pytest.fixture
def knn(book_data):
    '''Returns a KNN instance'''
    return KNN(bookData=book_data)
knn = KNN(bookData=BookDataset(clean_data=True))

@pytest.fixture
def knn_fit(book_data):
    '''Returns a fitted KNN instance'''
    knn_fit = KNN(bookData=book_data)
    knn_fit.fit()
    return knn_fit
knn_fit = KNN(bookData=book_data)
knn_fit.fit()

@pytest.fixture
def isbn_correct():
    '''Returns a string of an ISBN in the dataset with more than 10 reviews'''
    return '0451202856'
isbn_correct = '0451202856'

@pytest.fixture
def isbn_incorrect():
    '''Returns a string of an ISBN in the dataset with less than 10 reviews'''
    return '0609804618'
isbn_incorrect = '0609804618'

def test_knn_filtered():
    assert min(knn.data['ISBN'].value_counts()) >= 10

def test_knn_fit_attributes_exist():
    assert knn_fit.data_pivot.empty == False
    assert knn_fit.model != None

def test_predict_does_not_raise_error():
    try:
        knn_fit.predict(isbn_correct, 5)
    except KeyError as exc:
        assert False, f"ISBN '0451202856' raised an exception {exc}"
        
def test_predict_raises_error():
    with pytest.raises(KeyError):
        knn_fit.predict(isbn_incorrect, 5)

def test_predict_num_recs():
    assert len(knn_fit.predict(isbn_correct, 5)) == 5
    assert len(knn_fit.predict(isbn_correct, 10)) == 10
    assert len(knn_fit.predict(isbn_correct, 2)) == 2
    assert len(knn_fit.predict(isbn_correct, 50)) == 50
        
