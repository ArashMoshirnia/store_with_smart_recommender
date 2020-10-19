from django.urls import path

from users.api.views import UserRegistration

urlpatterns = [
    path('register/', UserRegistration.as_view(), name='user_register')
]
