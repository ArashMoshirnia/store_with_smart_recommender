import shutil
import os
import zipfile
from urllib.request import urlretrieve

from django.core.management import BaseCommand


class Command(BaseCommand):
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
        target_dir = 'recommenders/movielens-dataset/'

        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        file_names = os.listdir(source_dir)

        for file_name in file_names:
            shutil.move(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

        os.rmdir(source_dir)

        return True

    def handle(self, *args, **options):
        self.download_and_move_dataset()
