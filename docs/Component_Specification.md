# Component Specification
## Process Data
Input: dataset\
Creates a class (BookDataset) for the dataset with various operations. Includes methods to read, clean, split and return the dataset as well as add new data. 

## Recommender
* System\
Input: BookDataset\
A template class that reads a BookDataset object containing cleaned data, defines standard interfaces for implementing KNN and Matrix Factorization and filters the data further making it suitable for prediction.
* KNN\
Input: book, num_recommendations\
A wrapper class that uses KNN to get a list of book recommendations based on a particular book. Includes methods to create and fit a KNN model, and predict recommendations.
* Matrix Factorization\
Input: user, num_recommendations\
A wrapper class that uses Matrix Factorization (NMF) to get a list of book recommendations based on user preferences. Includes methods to create and fit an NMF model, and predict recommendations.

## System
Input: dataset\
Integrates the recommenders and datasets into a single interface for operation (class RSystem). Here are instructions for how a web developer would interact with the interface.

## Sample Implementation
Input: recommender_type
Includes code that loads data, builds models and provides recommendations depending on recommendation preference (similar_user or similar_book)

### STEP 1:
From the book_recs library import the RSystem class\
### STEP 2:
Create a config.json file (copy example screenshot) that contains the locations of the three csv files that are formatted exactly as the Kaggle dataset format. These are the csv files which the recommendation system will be built on. You may modify the "csv_file_new_books", "csv_file_new_ratings" and "csv_file_new_users" to hold the location of new data to be appended on to the existing recommendation system data (see step 4).\
### STEP 3: 
Instantiate a recommender system with specified features. Choose auto-clean by setting the flag clean_data=True or turn it off with clean_data=False. The recommender_type flag determines if the user wants to have a system that retrieves books based on similar books or a system that retrieves similar books based on books like users have read. recommender_type='similar_user' or recommender_type=’similar_book’ \ 
### STEP 4: 
Use the recommender system object to call the add_data function. The new data is located in a folder specified in your config.json file. The new data must be in the same format as the Kaggle book dataset data. There is a flag clean_data that can be set to true if the user wants the data auto-cleaned.\
### STEP 5:
#### a)
 If recommender_type='similar_user':  Use the recommender system object to call the function get_recommendations with an integer USER-ID as well as the number of recommendations to return to the user (between zero and max). This will return a pandas dataframe of similar books based on books like users have read.\
#### b)
If recommender_type='similar_book':  Use the recommendation system object to call all the function get_recommendations with string ISBN as well as the number of recommendations to return to the user (between zero and max). This will return a pandas dataframe of similar books to the book corresponding to the ISBN passed into the function.

