# your_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('', views.index, name='home'),
    path('restaurant/', views.restaurant, name='restaurant'),
    path('about/', views.about, name='about'),
    path('logout/', views.logout_view, name='logout'),
    path('restaurant/<int:location_id>//', views.detail, name='detail'),
    path('add_review/', views.add_review, name='add_review'),
    path('recommendations/', views.recommendation, name='recommendations'),

    # Add more URL patterns as needed
]
