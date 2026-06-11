from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from passlib.hash import bcrypt

from attempts.models import Attempt
from learning.models import KnowledgeState

from .forms import LoginForm, RegisterForm
from .models import User


def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            email = form.cleaned_data["email"]

            if User.objects.filter(email=email).exists():
                form.add_error("email", "Email already registered.")
            elif User.objects.filter(username=username).exists():
                form.add_error("username", "Username already taken.")
            else:
                user = User.objects.create(
                    username=username,
                    email=email,
                    password=bcrypt.hash(form.cleaned_data["password"]),
                )
                request.session["user_id"] = str(user.id)
                messages.success(request, "Account created successfully.")
                return redirect("users:profile", user_id=user.id)
    else:
        form = RegisterForm()

    return render(request, "users/register.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            password = form.cleaned_data["password"]
            user = User.objects.filter(email=email).first()

            if user and bcrypt.verify(password, user.password):
                request.session["user_id"] = str(user.id)
                messages.success(request, f"Welcome back, {user.username}.")
                return redirect("users:profile", user_id=user.id)

            form.add_error(None, "Invalid email or password.")
    else:
        form = LoginForm()

    return render(request, "users/login.html", {"form": form})


def logout_view(request):
    request.session.pop("user_id", None)
    messages.info(request, "You have been logged out.")
    return redirect("learning:home")


def user_list(request):
    users = User.objects.order_by("username")[:50]
    return render(request, "users/user_list.html", {"users": users})


def profile(request, user_id):
    user = get_object_or_404(User, id=user_id)
    attempts = Attempt.objects.filter(user=user)
    total = attempts.count()
    correct = attempts.filter(is_correct=True).count()
    states = KnowledgeState.objects.filter(user=user)
    topic_skills = {ks.topic: round(ks.skill_score, 4) for ks in states}

    context = {
        "profile_user": user,
        "total_attempts": total,
        "correct_attempts": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "strongest_topic": max(topic_skills, key=topic_skills.get) if topic_skills else None,
        "weakest_topic": min(topic_skills, key=topic_skills.get) if topic_skills else None,
        "topic_skills": topic_skills,
    }
    return render(request, "users/profile.html", context)
