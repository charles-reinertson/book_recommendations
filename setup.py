from setuptools import setup, find_packages


VERSION = '0.0.1'
DESCRIPTION = 'Easy recommender system for the https://www.kaggle.com/arashnic/book-recommendation-dataset?select=Books.csv dataset'
LONG_DESCRIPTION = '''Python package that allows users to easily clean and add data with predefined categories to a recommendation engine for books.
Users can retrieve their recommendations based upon past books they have reviewed. The package will allow web developers to create a website for book
recommendations. The package contains different configuration settings such as whether to auto-clean the data or recommend books
based on similar users or similar books. The similar books model uses the K-Nearest Neighbor algorithm whereas the similar users
specification uses the Matrix Factorization algorithm.'''

# Setting up
setup(
      # the name must match the folder name 'verysimplemodule'
      name="book_recs",
      version=VERSION,
      author="Charles Reinertson, Urmika Kasi, Rebecca Klein",
      author_email="<crein1@uw.edu, ukasi@uw.edu, klein324@uw.edu>",
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      packages=find_packages(),
      # add any additional packages that
      # needs to be installed along with your package
      install_requires=['numpy', 'pandas', 'sklearn', 'setuptools', 'pytest', 'pytest-cov', 'scipy'],
      keywords=['python', 'book recommendation dataset'],
     )
