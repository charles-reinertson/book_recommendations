from process_data import BookDataset
from recommender import KNN, Matrix_Factorization
from utils import config

class RSystem(object):
    def __init__(self, clean_data=True, recommender_type='similar_book'):
        """
        Initialize recommender system.

        clean_data: flag for whether the user wants us to clean his data for him or if it
                    is pre-cleaned
        recommender_type: if similar_book then get book recommendations based on a provided similar book. 
                          If similar_user then get book recommendations based on similar users.
        """
        assert(recommender_type == 'similar_book' or recommender_type == 'similar_user')
        # read in, clean (depending on flag), and merge data
        self.data_object = BookDataset(clean_data)
        # get recommender type, either KNN or Matrix Factorization
        self.recommender_type = recommender_type
        # train our model and return the specified recommender system object
        self.system = None
        self._train_system()



    def add_data(self, clean_data=True):
        """
        Add more data to the recommender system and retrain the model with this new data.
        """
        self.data_object.append_data(clean_data)
        self._train_system()

    def get_recommendations(self, user_input, num_recommendations=10):
        """
        Get the book recommendations for a specific user.

        user_input: Specify the desired user (if similar user)  or desired book ISBN
        (if similar book) to recieve their recommendations
        num_recommendations: Number of recommended book to return
        """
        predictions = self.system.predict(user_input, num_recommendations)

        return predictions[:10]
    
    def _train_system(self):
        """
        Train our data on either KNN or Matrix factorization recommender system.
        """
        if self.recommender_type == 'similar_book':
            self.system = KNN(self.data_object)
        else:
            self.system = Matrix_Factorization(self.data_object)
        
        self.system.fit()


