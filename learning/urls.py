from django.urls import path

from . import views

app_name = "learning"

urlpatterns = [
    path("", views.home, name="home"),
    path("progress/<uuid:user_id>/", views.progress, name="progress"),
    path("knowledge/<uuid:user_id>/", views.knowledge, name="knowledge"),
    path("retrain/", views.retrain, name="retrain"),
]
