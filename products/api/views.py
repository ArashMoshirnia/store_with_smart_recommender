from constance import config
from django.db.models import Avg
from rest_framework.authentication import TokenAuthentication
from rest_framework.generics import get_object_or_404
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from rest_framework.mixins import CreateModelMixin, RetrieveModelMixin

from products.api.serializers import ProductSerializer, ProductRatingSerializer
from products.models import Product, ProductRating
from recommenders.neural_network import get_recommendations_for_user

class ProductViewSet(ModelViewSet):
    authentication_classes = (TokenAuthentication, )
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

    def get_recommendations(self, request, *args, **kwargs):
        user = request.user
        smart_recommendations = []
        if user.is_authenticated:
            try:
                smart_recommendations = get_recommendations_for_user(user.id)
            except:
                pass

        recommendation_count = config.RECOMMENDATION_COUNT

        if len(smart_recommendations) < recommendation_count:
            count_diff = recommendation_count - len(smart_recommendations)

            newest_highest_products = Product.objects.filter(
                is_enabled=True
            ).annotate(
                avg_rating=Avg('ratings__rating')
            ).order_by(
                '-created_time__date', '-avg_rating'
            ).values_list(
                'id', flat=True
            )[:count_diff]

            smart_recommendations.extend(list(newest_highest_products))

        products = Product.objects.filter(id__in=smart_recommendations)
        serializer = ProductSerializer(products, many=True)

        return Response(serializer.data)


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
product_recommendation_list = ProductViewSet.as_view({
    'get': 'get_recommendations'
})
