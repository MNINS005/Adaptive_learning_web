from django.urls import path

from . import views

app_name = "attempts"

urlpatterns = [
    path("log/", views.log_attempt, name="log"),
]
