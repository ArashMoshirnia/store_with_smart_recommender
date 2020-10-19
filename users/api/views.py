from rest_framework.generics import CreateAPIView

from users.api.serializers import UserSerializer
from users.models import User


class UserRegistration(CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
