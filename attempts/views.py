import threading

from django.contrib import messages
from django.shortcuts import redirect
from django.utils.http import url_has_allowed_host_and_scheme

from learning.models import KnowledgeState
from src.logger import logger

from .forms import AttemptForm
from .models import Attempt


def compute_reward(is_correct, question_difficulty, user_skill):
    difficulty = question_difficulty or 0.5
    reward = 1.0 if is_correct else -0.3
    reward += max(0, difficulty - user_skill) * (0.5 if is_correct else 0.2)
    if difficulty < user_skill - 0.3:
        reward -= 0.5
    return float(reward)


def log_attempt(request):
    if request.method == "POST":
        form = AttemptForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data["user"]
            question = form.cleaned_data["question"]
            is_correct = form.cleaned_data["result"] == "correct"

            attempt = Attempt.objects.create(
                user=user,
                question=question,
                is_correct=is_correct,
            )

            knowledge_state, _ = KnowledgeState.objects.get_or_create(
                user=user,
                topic=question.topic,
                defaults={"skill_score": 0.3},
            )
            current_skill = knowledge_state.skill_score

            if is_correct:
                delta = 0.05
                knowledge_state.skill_score = min(1.0, current_skill + delta)
            else:
                delta = 0.03
                knowledge_state.skill_score = max(0.0, current_skill - delta)
            knowledge_state.save()

            reward = compute_reward(is_correct, question.difficulty, current_skill)
            logger.info(f"Attempt user={user.username} correct={is_correct} reward={round(reward, 3)}")

            if Attempt.objects.count() % 500 == 0:
                def retrain():
                    from src.pipeline.training_pipeline import TrainingPipeline

                    TrainingPipeline().run()

                threading.Thread(target=retrain, daemon=True).start()

            if is_correct:
                messages.success(request, "Nice work. Your progress was updated.")
            else:
                messages.info(request, "Attempt saved. Try another problem and keep going.")
            next_url = request.POST.get("next")
            if next_url and url_has_allowed_host_and_scheme(
                next_url,
                allowed_hosts={request.get_host()},
                require_https=request.is_secure(),
            ):
                return redirect(next_url)
            return redirect("users:profile", user_id=attempt.user_id)

        messages.error(request, "Please select an attempt result before saving.")
        next_url = request.POST.get("next")
        if next_url and url_has_allowed_host_and_scheme(
            next_url,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            return redirect(next_url)

    return redirect("questions:list")
