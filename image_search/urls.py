from django.urls import path
from . import views

app_name = 'image_search'

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search_images, name='search_images'),
] 