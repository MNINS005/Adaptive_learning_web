import os

from django.shortcuts import get_object_or_404, render

from attempts.models import Attempt
from learning.models import KnowledgeState
from src.logger import logger
from users.models import User

from .models import Question

TOPIC_MAP = {
    "arrays": 0,
    "linked_lists": 1,
    "trees": 2,
    "graphs": 3,
    "dynamic_programming": 4,
    "sorting": 5,
    "searching": 6,
    "recursion": 7,
    "general_cs": 8,
}

DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

ENABLE_ML_RECOMMENDER = os.getenv("ENABLE_ML_RECOMMENDER") == "1"
dkt_model = rl_agent = None
question_index = {}
num_questions = 0

if ENABLE_ML_RECOMMENDER:
    try:
        from src.constants import ARTIFACT_DIR, DKT_MODEL_PATH, RL_POLICY_PATH
        from src.utils.main_utils import load_dkt_model, load_json, load_rl_agent

        dkt_model = load_dkt_model(DKT_MODEL_PATH)
        rl_agent = load_rl_agent(RL_POLICY_PATH)
        question_index = load_json(os.path.join(ARTIFACT_DIR, "transformed", "question_index.json"))
        num_questions = len(question_index)
        logger.info(f"Models loaded - {num_questions} questions indexed")
    except Exception as exc:
        logger.warning(f"Models not loaded: {exc}")


def difficulty_level_for_skill(skill_score):
    if skill_score < 0.4:
        return "Easy"
    if skill_score < 0.7:
        return "Medium"
    return "Hard"


def fallback_question_for(user):
    attempted_question_ids = Attempt.objects.filter(user=user).values_list("question_id", flat=True)
    weakest_state = KnowledgeState.objects.filter(user=user).order_by("skill_score").first()
    questions = Question.objects.exclude(id__in=attempted_question_ids)

    if not questions.exists():
        return Question.objects.order_by("topic", "difficulty", "title").first()

    if weakest_state:
        target_level = difficulty_level_for_skill(weakest_state.skill_score)
        topic_match = (
            questions
            .filter(topic=weakest_state.topic)
            .filter(difficulty_level=target_level)
            .order_by("difficulty", "title")
            .first()
        )
        if topic_match:
            return topic_match

        topic_fallback = (
            questions
            .filter(topic=weakest_state.topic)
            .order_by("difficulty", "title")
            .first()
        )
        if topic_fallback:
            return topic_fallback

    return questions.order_by("topic", "difficulty", "title").first()


def question_list(request):
    topic = request.GET.get("topic")
    difficulty_level = request.GET.get("difficulty_level")
    limit = int(request.GET.get("limit", 20))
    questions = Question.objects.all()
    topics = (
        Question.objects
        .exclude(topic__isnull=True)
        .exclude(topic="")
        .order_by("topic")
        .values_list("topic", flat=True)
        .distinct()
    )
    if topic:
        questions = questions.filter(topic=topic)
    if difficulty_level:
        questions = questions.filter(difficulty_level=difficulty_level)

    questions = list(questions[:limit])
    current_user = None
    user_id = request.session.get("user_id")
    if user_id:
        current_user = User.objects.filter(id=user_id).first()

    if current_user:
        attempts = (
            Attempt.objects
            .filter(user=current_user, question__in=questions)
            .order_by("question_id", "-attempted_at")
        )
        latest_by_question = {}
        for attempt in attempts:
            latest_by_question.setdefault(attempt.question_id, attempt)

        for question in questions:
            attempt = latest_by_question.get(question.id)
            if not attempt:
                question.practice_status = "Not Attempted"
            elif attempt.is_correct:
                question.practice_status = "Solved"
            else:
                question.practice_status = "Attempted"
    else:
        for question in questions:
            question.practice_status = "Login to track"

    return render(
        request,
        "questions/question_list.html",
        {
            "questions": questions,
            "topic": topic,
            "difficulty_level": difficulty_level,
            "topics": topics,
            "difficulty_levels": DIFFICULTY_LEVELS,
            "limit": limit,
        },
    )


def next_question(request, user_id):
    user = get_object_or_404(User, id=user_id)
    source = "rl_agent"
    user_skill = 0.0

    if not dkt_model or not rl_agent:
        question = fallback_question_for(user)
        source = "fallback"
    else:
        import numpy as np
        from src.utils.main_utils import (
            encode_user_sequence,
            filter_candidate_questions,
            get_knowledge_state,
        )

        attempts = Attempt.objects.filter(user=user).order_by("attempted_at")
        attempt_dicts = [
            {"question_id": str(attempt.question_id), "is_correct": int(attempt.is_correct)}
            for attempt in attempts
        ]
        user_sequence = encode_user_sequence(attempt_dicts, question_index, num_questions)
        state = get_knowledge_state(dkt_model, user_sequence)
        user_skill = float(np.mean(state))

        weakest_state = KnowledgeState.objects.filter(user=user).order_by("skill_score").first()
        target_topic = weakest_state.topic if weakest_state else None
        target_level = difficulty_level_for_skill(weakest_state.skill_score) if weakest_state else None
        attempted_question_ids = set(
            Attempt.objects.filter(user=user).values_list("question_id", flat=True)
        )
        available_questions = [
            question
            for question in Question.objects.all()
            if question.id not in attempted_question_ids
        ]

        candidates = filter_candidate_questions(available_questions, user_skill, target_topic)
        if target_level:
            level_candidates = [
                question for question in candidates
                if question.difficulty_level == target_level
            ]
            candidates = level_candidates or candidates
        if not candidates:
            candidates = available_questions[:10]

        scores = [
            rl_agent.forward(state, question.difficulty, TOPIC_MAP.get(question.topic, 0))
            for question in candidates
        ]
        question = candidates[int(np.argmax(scores))] if candidates else None

    return render(
        request,
        "questions/next_question.html",
        {
            "learner": user,
            "question": question,
            "user_skill": round(user_skill, 4),
            "source": source,
        },
    )
