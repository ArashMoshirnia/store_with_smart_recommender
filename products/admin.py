from django.contrib import admin

from products.models import Product, Category, ProductRating


class ProductAdmin(admin.ModelAdmin):
    list_display = ('id', 'name', 'is_enabled')


class CategoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'name')


class RatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'product', 'rating')


admin.site.register(Product, ProductAdmin)
admin.site.register(Category, CategoryAdmin)
admin.site.register(ProductRating, RatingAdmin)
