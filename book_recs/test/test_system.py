import _mypath
from system import RSystem
import pytest

@pytest.fixture
def system_similiar_book():
    '''
    Returns a RSystem instance with similar_book to 
    get book recommendations based on a provided similar book. 
    '''
    return RSystem(clean_data=True, recommender_type='similar_book')

@pytest.fixture
def system_similar_user():
    '''
    Returns a RSystem instance with similar_user to 
    get book recommendations based on similar users. 
    '''
    return RSystem(clean_data=True, recommender_type='similar_user')


def test_add_data_sim_book(system_similiar_book):
    before = system_similiar_book.data_object.df.shape[0]
    system_similiar_book.add_data()
    after  = system_similiar_book.data_object.df.shape[0]
    assert before < after

def test_get_recommendations_sim_book(system_similiar_book):
    
    recs = system_similiar_book.get_recommendations("12432", 10)
    assert isinstance(recs, list)
    assert len(recs) == 10

def test_add_data_sim_user(system_similar_user):
    before = system_similar_user.data_object.df.shape[0]
    system_similar_user.add_data()
    after  = system_similar_user.data_object.df.shape[0]
    assert before < after

def test_get_recommendations_sim_user(system_similar_user):
    
    recs = system_similar_user.get_recommendations(12432, 10)
    assert isinstance(recs, list)
    assert len(recs) == 10