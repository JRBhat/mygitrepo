from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    # path("custom", views.custom, name="custom"),
    # path("jesus", views.jesus, name="jesus"),
    # path("<str:name>", views.greet, name="greet")
]

