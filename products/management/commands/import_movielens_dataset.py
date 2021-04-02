import pandas as pd
import shutil
import os
import zipfile
from urllib.request import urlretrieve

from django.contrib.auth.hashers import make_password
from django.core.management import BaseCommand

from products.models import Product, ProductRating
from users.models import User


class Command(BaseCommand):
    dataset_dir = 'recommenders/movielens-dataset/'

    def download_and_move_dataset(self):
        zip_file_name = 'movielens.zip'
        if not os.path.exists(zip_file_name):
            print("Downloading movielens dataset...")
            try:
                urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", "movielens.zip")
            except Exception as e:
                print("Download failed. Please check connection")
                return False
            print("Download done")
        else:
            print('No need to download since file already exists')

        zip_ref = zipfile.ZipFile('movielens.zip', "r")
        zip_ref.extractall()

        source_dir = 'ml-100k/'
        target_dir = self.dataset_dir

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        file_names = os.listdir(source_dir)

        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

        os.rmdir(source_dir)

        return True

    def prepare_dataframes(self):
        users_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        # The movies file contains a binary feature for each genre.
        genre_cols = [
            "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        movies_cols = ['movie_id', 'title', 'release_date', "video_release_date", "imdb_url"] + genre_cols

        ratings = pd.read_csv(self.dataset_dir + 'u.data', sep='\t', names=ratings_cols, encoding='latin-1')
        users = pd.read_csv(self.dataset_dir + 'u.user', sep='|', names=users_cols, encoding='latin-1')
        movies = pd.read_csv(self.dataset_dir + 'u.item', sep='|', names=movies_cols, encoding='latin-1')

        movielens = ratings.merge(movies, on='movie_id').merge(users, on='user_id')

        return movielens, users, movies, ratings

    def import_movies_as_products(self, movies_df):
        products = []
        for i, movie in movies_df.iterrows():
            movie_id = movie.movie_id
            movie_name = movie.title

            try:
                Product.objects.get(id=movie_id)
            except Product.DoesNotExist:
                products.append(
                    Product(name=movie_name)
                )

        Product.objects.bulk_create(products)

    def create_users(self, users_df):
        users = []
        current_user_count = User.objects.count()
        num_users_needed = len(users_df.index)
        num_users_to_create = num_users_needed - current_user_count

        default_password = make_password('1111')
        for i in range(num_users_to_create):
            users.append(
                User(
                    username=User.generate_random_username(),
                    password=default_password
                )
            )
        User.objects.bulk_create(users)

    def import_ratings(self, ratings_df):
        ratings = []
        for i, rating in ratings_df.iterrows():
            ratings.append(
                ProductRating(
                    user_id=rating.user_id,
                    product_id=rating.movie_id,
                    rating=rating.rating
                )
            )
        ProductRating.objects.bulk_create(ratings, ignore_conflicts=True)

    def handle(self, *args, **options):
        download_result = self.download_and_move_dataset()
        if not download_result:
            return False

        movielens, users, movies, ratings = self.prepare_dataframes()

        self.import_movies_as_products(movies)
        self.create_users(users)
        self.import_ratings(ratings)
