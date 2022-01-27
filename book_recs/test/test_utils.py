from utils import config


assert('data/Books.csv' == config('csv_file_books'))
assert('data/Ratings.csv' == config('csv_file_ratings'))
assert('data/Users.csv' == config('csv_file_users'))
assert('I will fill out this section when I know how knn works better' == config('knn_hyperparameters.TODO'))