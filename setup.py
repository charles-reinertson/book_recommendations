from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Easy recommender system for the https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv dataset'
LONG_DESCRIPTION = 'Implement a recommender system for the https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv dataset that includes data cleaning, feature engineering, and modeling in a few lines of code'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="book_recs", 
        version=VERSION,
        author="Charles Reinertson, Urmika, Rebecca",
        author_email="<crein1@uw.edu, edu, edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        # add any additional packages that 
        # needs to be installed along with your package
        install_requires=['matplotlib', 'numpy', 'pandas', 'sklearn', 'warnings'], 
        
        keywords=['python', 'book recommendation dataset'],
)