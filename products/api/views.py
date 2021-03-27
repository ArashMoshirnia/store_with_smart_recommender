from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from rest_framework.mixins import CreateModelMixin, RetrieveModelMixin

from products.api.serializers import ProductSerializer, ProductRatingSerializer
from products.models import Product, ProductRating


class ProductViewSet(ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer


class ProductRatingViewSet(RetrieveModelMixin, CreateModelMixin, GenericViewSet):
    authentication_classes = (TokenAuthentication, )
    permission_classes = (IsAuthenticated, )
    queryset = ProductRating.objects.all()
    serializer_class = ProductRatingSerializer

    def create(self, request, *args, **kwargs):
        user = request.user
        product_id = self.kwargs.get('pk')
        product = get_object_or_404(Product.objects.all(), pk=product_id)

        data = {
            'user': user.id,
            'product': product_id,
            'rating': request.data.get('rating')
        }

        serializer = ProductRatingSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response(serializer.data)

    def retrieve(self, request, *args, **kwargs):
        user = request.user
        product_id = self.kwargs.get('pk')
        product = get_object_or_404(Product.objects.all(), pk=product_id)

        product_rating = get_object_or_404(ProductRating.objects.all(), user=user, product=product)
        serializer = ProductRatingSerializer(product_rating)

        return Response(serializer.data)


product_list_create = ProductViewSet.as_view({
    'get': 'list',
    'post': 'create'
})
product_retrieve_update_destroy = ProductViewSet.as_view({
    'get': 'retrieve',
    'put': 'update',
    'delete': 'destroy'
})
product_rating_viewset = ProductRatingViewSet.as_view({
    'get': 'retrieve',
    'post': 'create'
})
