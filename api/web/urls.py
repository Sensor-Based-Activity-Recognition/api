from django.urls import path

from . import views

urlpatterns = [
    path("CNN", views.CNN, name="CNN"),
]
