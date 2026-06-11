from django.urls import path

from . import views

app_name = "users"

urlpatterns = [
    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("", views.user_list, name="list"),
    path("<uuid:user_id>/profile/", views.profile, name="profile"),
]
