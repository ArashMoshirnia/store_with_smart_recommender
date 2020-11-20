from rest_framework import serializers
from rest_framework.exceptions import ValidationError

from products.models import Category, Product, ProductRating


class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ('id', 'name')


class ProductSerializer(serializers.ModelSerializer):
    category = CategorySerializer()
    avatar = serializers.SerializerMethodField()

    class Meta:
        model = Product
        fields = ('id', 'name', 'quantity', 'sold_count', 'description', 'avatar', 'category')

    def get_avatar(self, obj):
        avatar_url = obj.avatar.url
        request = self.context.get('request')
        return request.build_absolute_uri(avatar_url)


class ProductRatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProductRating
        fields = ('user', 'product', 'rating')

    def validate_rating(self, value):
        if value < 0 or value > 5:
            raise ValidationError('Rating must be a value between 0 and 5')
        return value
