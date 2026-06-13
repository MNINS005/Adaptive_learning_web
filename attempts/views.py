import threading

from django.contrib import messages
from django.shortcuts import redirect, render

from learning.models import KnowledgeState
from src.logger import logger

from .forms import AttemptForm
from .models import Attempt


def compute_reward(is_correct, question_difficulty, user_skill, answer_time):
    difficulty = question_difficulty or 0.5
    reward = 1.0 if is_correct else -0.3
    reward += max(0, difficulty - user_skill) * (0.5 if is_correct else 0.2)
    if answer_time <= 30 and is_correct:
        reward += 0.1
    if difficulty < user_skill - 0.3:
        reward -= 0.5
    return float(reward)


def log_attempt(request):
    if request.method == "POST":
        form = AttemptForm(request.POST)
        if form.is_valid():
            user = form.cleaned_data["user"]
            question = form.cleaned_data["question"]
            is_correct = form.cleaned_data["is_correct"]
            time_taken = form.cleaned_data["time_taken"]

            attempt = Attempt.objects.create(
                user=user,
                question=question,
                is_correct=is_correct,
                time_taken=time_taken,
            )

            knowledge_state, _ = KnowledgeState.objects.get_or_create(
                user=user,
                topic=question.topic,
                defaults={"skill_score": 0.3},
            )
            current_skill = knowledge_state.skill_score
            answer_time = time_taken or 60

            if is_correct:
                delta = 0.08 if answer_time <= 30 else 0.05 if answer_time <= 150 else 0.02
                knowledge_state.skill_score = min(1.0, current_skill + delta)
            else:
                delta = 0.01 if answer_time >= 150 else 0.03
                knowledge_state.skill_score = max(0.0, current_skill - delta)
            knowledge_state.save()

            reward = compute_reward(is_correct, question.difficulty, current_skill, answer_time)
            logger.info(f"Attempt user={user.username} correct={is_correct} reward={round(reward, 3)}")

            if Attempt.objects.count() % 500 == 0:
                def retrain():
                    from src.pipeline.training_pipeline import TrainingPipeline

                    TrainingPipeline().run()

                threading.Thread(target=retrain, daemon=True).start()

            messages.success(request, "Attempt logged successfully.")
            return redirect("users:profile", user_id=attempt.user_id)
    else:
        form = AttemptForm()

    return render(request, "attempts/log_attempt.html", {"form": form})
