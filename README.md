# RecommendARead
## Background
Python package that allows users to easily clean and add data with predefined categories to a recommendation engine for books. Users can retrieve their recommendations based upon past books they have reviewed. The package will allow web developers to create a website for book recommendations. The package contains different configuration settings such as whether to auto-clean the data or recommend books based on similar users or similar books. The similar books model uses the K-Nearest Neighbor algorithm whereas the similar users specification uses the Matrix Factorization algorithm. 
## Team Members
Charles Reinertson\
Rebecca Klein\
Urmika Kasi
## Data
[Books Kaggle Dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset?select=movies_metadata.csv)\
Features selected from the dataset:\
Books.csv\
ISBN: Books are identified by their respective ISBN\
Book-Title: The title of the book\
Book-Author: The author of the book\
Year-Of-Publication: The year of publication of the book\
Publisher: The publisher of the book\
Image-URL-S: URLs linking to cover images small size\
Image-URL-M: URLs linking to cover images medium size\
Image-URL-L: URLs linking to cover images large size\
Ratings.csv\
Book-Rating: The rating of the book expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by 0.\
USER-ID: ID of user who wrote the rating\
ISBN: Books are identified by their respective ISBN\
Users.csv\
Location: location of the user\
Age: age of the user\
Software\
Programming Languages\
Python\
Python Packages\
numpy >= 1.22.0\
pandas >= 1.3.5\
pytest >= 7.0.1\
setuptools >= 60.5.0\
scipy >= 1.7.3\
scikit-learn >= 1.0.2
## Structure
This package has the following structure.\
```
book_recommendations/
  |- bin/
     |- _mypath.py
     |- main.py
  |- book_recs/
     |- test
       |- _mypath.py
       |- test_knn.py
       |- test_matrix_factorization.py
       |- test_process_data.py
       |- test_system.py
       |- test_utils.py
     |- __init__.py
     |- process_data.py
     |- recommender.py
     |- system.py
     |- utils.py
  |- data/
     |- Books.csv
     |- Ratings.csv
     |- Users.csv
  |- docs/
     |- Design_Specification.md
     |- Final_Presentation.pdf
     |- Functional_Specification.md
  |- new_data/
     |- Books.csv
     |- Ratings.csv
     |- Users.csv
  |- README.md
  |- config.json
  |- setup.py

```
## Installation
pip install betas
## Usage
### STEP 1:
From the book_recs library import the RSystem class
### STEP 2:
Create a config.json file (copy example screenshot) that contains the locations of the three csv files that are formatted exactly as the Kaggle dataset format. These are the csv files which the recommendation system will be built on. You may modify the "csv_file_new_books", "csv_file_new_ratings" and "csv_file_new_users" to hold the location of new data to be appended on to the existing recommendation system data (see step 4).
### STEP 3: 
Instantiate a recommender system with specified features. Choose auto-clean by setting the flag clean_data=True or turn it off with clean_data=False. The recommender_type flag determines if the user wants to have a system that retrieves books based on similar books or a system that retrieves similar books based on books like users have read. recommender_type='similar_user' or recommender_type=’similar_book’ 
### STEP 4: 
Use the recommender system object to call the add_data function. The new data is located in a folder specified in your config.json file. The new data must be in the same format as the Kaggle book dataset data. There is a flag clean_data that can be set to true if the user wants the data auto-cleaned.
### STEP 5:
#### a)
 If recommender_type='similar_user':  Use the recommender system object to call the function get_recommendations with an integer USER-ID as well as the number of recommendations to return to the user (between zero and max). This will return a pandas dataframe of similar books based on books like users have read.
#### b)
If recommender_type='similar_book':  Use the recommendation system object to call all the function get_recommendations with string ISBN as well as the number of recommendations to return to the user (between zero and max). This will return a pandas dataframe of similar books to the book corresponding to the ISBN passed into the function.
