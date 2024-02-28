# APP url file

from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("jb", views.jb, name="jb"),
    path("kkk", views.kkk, name="kkk"),
    path("<str:name>", views.greet_better, name="greet")
    ]
