from django.urls import path
from .views import homePage, healthPrediction

urlpatterns = [
    path('', homePage),
    path('prediction', healthPrediction)
]
