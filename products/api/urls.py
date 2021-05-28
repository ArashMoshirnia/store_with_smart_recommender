from django.urls import path

from products.api.views import product_list_create, product_rating_viewset, product_retrieve_update_destroy, \
    product_recommendation_list

urlpatterns = [
    path('', product_list_create, name='product_list_create'),
    path('<int:pk>/', product_retrieve_update_destroy, name='product_retrieve_update_destroy'),
    path('<int:pk>/rate/', product_rating_viewset, name='product_rating'),
    path('recommendations/', product_recommendation_list, name='product_recommendation')
]
