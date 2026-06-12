import threading

from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from attempts.models import Attempt
from users.models import User

from .models import KnowledgeState


def home(request):
    return render(request, "learning/index.html")


def progress(request, user_id):
    user = get_object_or_404(User, id=user_id)
    recent_attempts = (
        Attempt.objects.filter(user=user)
        .select_related("question")
        .order_by("-attempted_at")[:20]
    )
    topic_progress = KnowledgeState.objects.filter(user=user).order_by("-skill_score")
    return render(
        request,
        "learning/progress.html",
        {
            "learner": user,
            "recent_attempts": recent_attempts,
            "topic_progress": topic_progress,
        },
    )


def knowledge(request, user_id):
    user = get_object_or_404(User, id=user_id)
    states = KnowledgeState.objects.filter(user=user).order_by("-skill_score")
    return render(request, "learning/knowledge.html", {"learner": user, "states": states})


def retrain(request):
    if request.method == "POST":
        def run_training():
            from src.pipeline.training_pipeline import TrainingPipeline

            TrainingPipeline().run()

        threading.Thread(target=run_training, daemon=True).start()
        messages.success(request, "Retraining started in the background.")
        return redirect("learning:home")

    return render(request, "learning/retrain.html")
