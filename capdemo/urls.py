from django.urls import path

from . import views

app_name = 'capdemo'
urlpatterns = [
    path('', views.index, name='index'),
    path('caption/', views.caption_all, name='results')
]