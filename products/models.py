from django.conf import settings
from django.db import models
from django.db.models import Avg


class Category(models.Model):
    name = models.CharField(max_length=100)
    is_enabled = models.BooleanField('is enabled', default=True)
    created_time = models.DateTimeField(auto_now_add=True)
    modified_time = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'category'
        verbose_name_plural = 'categories'

    def __str__(self):
        return self.name


class Product(models.Model):
    name = models.CharField(max_length=100)
    category = models.ForeignKey(Category, related_name='products', on_delete=models.PROTECT, null=True)
    quantity = models.PositiveIntegerField(default=0)
    sold_count = models.PositiveIntegerField(default=0)
    price = models.PositiveIntegerField(default=0)
    description = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/products', null=True, blank=True)
    is_enabled = models.BooleanField(default=True)
    created_time = models.DateTimeField(auto_now_add=True)
    modified_time = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-id']

    def __str__(self):
        return self.name

    @property
    def average_rating(self):
        avg_rating = self.ratings.aggregate(average=Avg('rating'))['average']
        if avg_rating is None:
            avg_rating = 0
        return avg_rating


class ProductRating(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='product_ratings', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, related_name='ratings', on_delete=models.CASCADE)
    rating = models.PositiveSmallIntegerField()

    class Meta:
        unique_together = ('user', 'product')

    def __str__(self):
        return '{} - {}: {}'.format(self.user, self.product, self.rating)


class PurchaseProduct(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='product_purchases', on_delete=models.CASCADE)
    product = models.ForeignKey(Product, related_name='purchases', on_delete=models.CASCADE)
    purchase_count = models.PositiveSmallIntegerField()
    created_time = models.DateTimeField(auto_now_add=True)
    modified_time = models.DateTimeField(auto_now=True)

    def __str__(self):
        return '{} - {}: {}'.format(self.user, self.product, self.created_time)
