from django.urls import path
from . import views

urlpatterns = [
    # auth
    path("auth/register/",  views.RegisterView.as_view(),    name="register"),
    path("",     views.LoginView.as_view(),       name="login"),

    # questions
    path("questions/all/",               views.AllQuestionsView.as_view(),  name="questions-all"),
    path("questions/next/<uuid:user_id>/", views.NextQuestionView.as_view(), name="next-question"),

    # attempts
    path("attempts/", views.LogAttemptView.as_view(), name="log-attempt"),

    # users
    path("users/all/",                        views.AllUsersView.as_view(),    name="all-users"),
    path("users/my_profile/<uuid:user_id>/",  views.MyProfileView.as_view(),  name="my-profile"),
    path("users/my_progress/<uuid:user_id>/", views.MyProgressView.as_view(), name="my-progress"),
    path("users/my_knowledge/<uuid:user_id>/",views.MyKnowledgeView.as_view(),name="my-knowledge"),

    # training
    path("train/retrain/", views.RetrainView.as_view(), name="retrain"),
]