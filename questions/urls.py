from django.urls import path

from . import views

app_name = "questions"

urlpatterns = [
    path("", views.question_list, name="list"),
    path("next/<uuid:user_id>/", views.next_question, name="next"),
]
