import random
import string

from django.contrib.auth.models import AbstractUser


class User(AbstractUser):

    @staticmethod
    def generate_random_username(length=6):
        lowercase_letters = string.ascii_lowercase
        digits = string.digits
        return ''.join(random.choice(lowercase_letters + digits) for i in range(length))
