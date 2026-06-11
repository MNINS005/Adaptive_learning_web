import os

import numpy as np
from django.shortcuts import get_object_or_404, render

from attempts.models import Attempt
from learning.models import KnowledgeState
from src.constants import ARTIFACT_DIR, DKT_MODEL_PATH, RL_POLICY_PATH
from src.logger import logger
from src.utils.main_utils import (
    encode_user_sequence,
    filter_candidate_questions,
    get_knowledge_state,
    load_dkt_model,
    load_json,
    load_rl_agent,
)
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

try:
    dkt_model = load_dkt_model(DKT_MODEL_PATH)
    rl_agent = load_rl_agent(RL_POLICY_PATH)
    question_index = load_json(os.path.join(ARTIFACT_DIR, "transformed", "question_index.json"))
    num_questions = len(question_index)
    logger.info(f"Models loaded - {num_questions} questions indexed")
except Exception as exc:
    dkt_model = rl_agent = None
    question_index = {}
    num_questions = 0
    logger.warning(f"Models not loaded: {exc}")


def question_list(request):
    topic = request.GET.get("topic")
    limit = int(request.GET.get("limit", 20))
    questions = Question.objects.all()
    if topic:
        questions = questions.filter(topic=topic)

    return render(
        request,
        "questions/question_list.html",
        {"questions": questions[:limit], "topic": topic, "limit": limit},
    )


def next_question(request, user_id):
    user = get_object_or_404(User, id=user_id)
    source = "rl_agent"
    user_skill = 0.0

    if not dkt_model or not rl_agent:
        question = Question.objects.first()
        source = "fallback"
    else:
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
        candidates = filter_candidate_questions(list(Question.objects.all()), user_skill, target_topic)
        if not candidates:
            candidates = list(Question.objects.all()[:10])

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
