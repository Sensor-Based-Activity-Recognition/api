from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("CNN", views.CNN, name="CNN"),
    path("HGBC", views.HGBC, name="HGBC"),
]
