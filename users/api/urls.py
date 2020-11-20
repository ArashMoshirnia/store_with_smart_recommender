from django.urls import path

from users.api.views import UserRegistration, UserLoginView

urlpatterns = [
    path('register/', UserRegistration.as_view(), name='user_register'),
    path('login/', UserLoginView.as_view(), name='user_login')
]
