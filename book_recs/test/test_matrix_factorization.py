import _mypath
from process_data import BookDataset
from recommender import Matrix_Factorization
import pytest
from numpy import testing

#@pytest.fixture
def book_data():
    '''Returns a BookDataset instance with clean data'''
    return BookDataset(clean_data=True)
book_data = BookDataset(clean_data=True)

@pytest.fixture
def nmf():
    '''Returns an NMF instance'''
    return Matrix_Factorization(bookData=book_data())
nmf = Matrix_Factorization(bookData=BookDataset(clean_data=True))

@pytest.fixture
def nmf_fit():
    '''Returns a fitted NMF instance'''
    nmf_fit = Matrix_Factorization(bookData=book_data)
    nmf_fit.fit()
    return nmf_fit
nmf_fit = Matrix_Factorization(bookData=book_data)
nmf_fit.fit()

@pytest.fixture
def user_correct():
    '''Returns a user in the dataset with more than 200 reviews'''
    return '16795'
user_correct = 16795

@pytest.fixture
def user_incorrect():
    '''Returns a user in the dataset with less than 10 reviews'''
    return '1'
user_incorrect = 1


def test_nmf_filtered():
    assert min(nmf.data['ISBN'].value_counts()) >= 10

def test_nmf_fit_attributes_exist():
    assert isinstance(nmf_fit.model, type(None)) == False
    assert isinstance(nmf_fit.model, type(None)) == False



def test_predict_does_not_raise_error():
    # try:
    #     nmf_fit.predict(user_correct, 5)
    # except KeyError as exc:
    #     assert False, f"User '16795' raised an exception {exc}"
    df = nmf_fit.predict(user_correct, 5)
    assert df.shape[0] == 5
        
def test_predict_raises_error():
    with pytest.raises(KeyError):
        nmf_fit.predict(int(user_incorrect), 5)

def test_predict_num_recs():
    assert len(nmf_fit.predict(user_correct, 5)) == 5
    assert len(nmf_fit.predict(user_correct, 10)) == 10
    assert len(nmf_fit.predict(user_correct, 2)) == 2
    assert len(nmf_fit.predict(user_correct, 50)) == 50
        
