from rest_framework.authtoken.models import Token
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response

from users.api.serializers import UserSerializer
from users.models import User


class UserRegistration(CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer


class UserLoginView(ObtainAuthToken):

    def post(self, request, *args, **kwargs):
        """
        API view that logs a user in and returns an authorization token. Request body should be as follows:
        { 'username': username, 'password': password
        }

        """
        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        user_serializer = UserSerializer(user).data
        return Response({
            'token': token.key,
            'user': user_serializer,
        })
