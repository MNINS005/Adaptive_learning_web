import os
import sys
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from passlib.hash import bcrypt

from .models import User, Question, Attempt, KnowledgeState
from .serializers import (
    UserCreateSerializer, UserOutSerializer,
    AttemptCreateSerializer, AttemptOutSerializer,
    QuestionOutSerializer
)
from src.utils.main_utils import (
    load_dkt_model, load_rl_agent, load_json,
    encode_user_sequence, get_knowledge_state,
    filter_candidate_questions, compute_reward
)
from src.constants import DKT_MODEL_PATH, RL_POLICY_PATH, ARTIFACT_DIR
from src.logger import logger

# ── load models once ───────────────────────────────────────────────────
TOPIC_MAP = {
    "arrays": 0, "linked_lists": 1, "trees": 2,
    "graphs": 3, "dynamic_programming": 4,
    "sorting": 5, "searching": 6,
    "recursion": 7, "general_cs": 8,
}

try:
    dkt_model      = load_dkt_model(DKT_MODEL_PATH)
    rl_agent       = load_rl_agent(RL_POLICY_PATH)
    question_index = load_json(
        os.path.join(ARTIFACT_DIR, "transformed", "question_index.json")
    )
    num_questions  = len(question_index)
    logger.info(f"Models loaded — {num_questions} questions indexed")
except Exception as e:
    dkt_model = rl_agent = None
    question_index = {}
    num_questions  = 0
    logger.warning(f"Models not loaded: {e}")


# ── Auth Views ─────────────────────────────────────────────────────────

class RegisterView(APIView):
    def post(self, request):
        ser = UserCreateSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=400)

        d = ser.validated_data
        if User.objects.filter(email=d["email"]).exists():
            return Response({"error": "Email already registered"}, status=400)
        if User.objects.filter(username=d["username"]).exists():
            return Response({"error": "Username taken"}, status=400)

        user = User.objects.create(
            username = d["username"],
            email    = d["email"],
            password = bcrypt.hash(d["password"])
        )
        logger.info(f"Registered: {user.username}")
        return Response(UserOutSerializer(user).data, status=201)


class LoginView(APIView):

    def get(self, request):
        return Response({
            "message": "Send a POST request with email and password."
        })

    def post(self, request):
        serializer = LoginSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]

        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {"error": "Invalid email or password"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        if not bcrypt.verify(password, user.password):
            return Response(
                {"error": "Invalid email or password"},
                status=status.HTTP_401_UNAUTHORIZED
            )

        logger.info(f"Login successful: {user.username}")

        return Response(
            {
                "message": "Login successful",
                "user": {
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email
                }
            },
            status=status.HTTP_200_OK
        )


# ── Question Views ─────────────────────────────────────────────────────

class NextQuestionView(APIView):
    def get(self, request, user_id):
        try:
            # fallback if models not loaded
            if not dkt_model or not rl_agent:
                q = Question.objects.first()
                if not q:
                    return Response({"error": "No questions found"}, status=404)
                return Response({
                    "question_id": str(q.id),
                    "content"    : q.content,
                    "topic"      : q.topic,
                    "difficulty" : q.difficulty,
                    "user_skill" : 0.0,
                    "source"     : "random_fallback"
                })

            # get attempts
            attempts = Attempt.objects.filter(
                user_id=user_id
            ).order_by("attempted_at")

            attempt_dicts = [
                {
                    "question_id": str(a.question_id),
                    "is_correct" : int(a.is_correct)
                }
                for a in attempts
            ]

            user_seq   = encode_user_sequence(
                attempt_dicts, question_index, num_questions
            )
            state      = get_knowledge_state(dkt_model, user_seq)
            user_skill = float(np.mean(state))

            # weakest topic
            ks = KnowledgeState.objects.filter(
                user_id=user_id
            ).order_by("skill_score").first()
            target_topic = ks.topic if ks else None

            # candidates
            all_q      = list(Question.objects.all())
            candidates = filter_candidate_questions(
                all_q, user_skill, target_topic
            )
            if not candidates:
                candidates = all_q[:10]

            scores = [
                rl_agent.forward(
                    state,
                    q.difficulty,
                    TOPIC_MAP.get(q.topic, 0)
                )
                for q in candidates
            ]
            best_q = candidates[int(np.argmax(scores))]

            return Response({
                "question_id"  : str(best_q.id),
                "content"      : best_q.content,
                "topic"        : best_q.topic,
                "difficulty"   : best_q.difficulty,
                "user_skill"   : round(user_skill, 4),
                "leetcode_url" : best_q.leetcode_url,
                "source"       : "rl_agent"
            })

        except Exception as e:
            logger.error(f"Next question error: {e}")
            return Response({"error": str(e)}, status=500)


class AllQuestionsView(APIView):
    def get(self, request):
        topic = request.query_params.get("topic")
        limit = int(request.query_params.get("limit", 20))
        qs    = Question.objects.all()
        if topic:
            qs = qs.filter(topic=topic)
        return Response(
            QuestionOutSerializer(qs[:limit], many=True).data
        )


# ── Attempt Views ──────────────────────────────────────────────────────

class LogAttemptView(APIView):
    def post(self, request):
        ser = AttemptCreateSerializer(data=request.data)
        if not ser.is_valid():
            return Response(ser.errors, status=400)

        d = ser.validated_data
        try:
            user     = User.objects.get(id=d["user_id"])
            question = Question.objects.get(id=d["question_id"])
        except (User.DoesNotExist, Question.DoesNotExist) as e:
            return Response({"error": str(e)}, status=404)

        attempt = Attempt.objects.create(
            user       = user,
            question   = question,
            is_correct = d["is_correct"],
            time_taken = d.get("time_taken")
        )

        # update knowledge state
        ks, _         = KnowledgeState.objects.get_or_create(
            user  = user,
            topic = question.topic,
            defaults={"skill_score": 0.3}
        )
        current_skill = ks.skill_score
        time          = d.get("time_taken") or 60

        if d["is_correct"]:
            delta = 0.08 if time <= 30 else 0.05 if time <= 150 else 0.02
            ks.skill_score = min(1.0, current_skill + delta)
        else:
            delta = 0.01 if time >= 150 else 0.03
            ks.skill_score = max(0.0, current_skill - delta)
        ks.save()

        reward = compute_reward(
            d["is_correct"], question.difficulty, current_skill, time
        )
        logger.info(
            f"Attempt → user={user.username} "
            f"correct={d['is_correct']} reward={round(reward,3)}"
        )

        # auto retrain trigger
        if Attempt.objects.count() % 500 == 0:
            import threading
            def retrain():
                from src.pipeline.training_pipeline import TrainingPipeline
                TrainingPipeline().run()
            threading.Thread(target=retrain, daemon=True).start()
            logger.info("Auto retrain triggered")

        return Response(AttemptOutSerializer(attempt).data, status=201)


# ── User Views ─────────────────────────────────────────────────────────

class MyProfileView(APIView):
    def get(self, request, user_id):
        try:
            user    = User.objects.get(id=user_id)
            total   = Attempt.objects.filter(user=user).count()
            correct = Attempt.objects.filter(user=user, is_correct=True).count()
            states  = KnowledgeState.objects.filter(user=user)

            topic_skills = {ks.topic: round(ks.skill_score, 4) for ks in states}
            strongest    = max(topic_skills, key=topic_skills.get) if topic_skills else None
            weakest      = min(topic_skills, key=topic_skills.get) if topic_skills else None

            return Response({
                "user_id"         : str(user.id),
                "username"        : user.username,
                "email"           : user.email,
                "total_attempts"  : total,
                "correct_attempts": correct,
                "accuracy"        : round(correct/total, 4) if total else 0.0,
                "strongest_topic" : strongest,
                "weakest_topic"   : weakest,
                "topic_skills"    : topic_skills,
                "member_since"    : str(user.created_at)
            })
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class MyProgressView(APIView):
    def get(self, request, user_id):
        try:
            user     = User.objects.get(id=user_id)
            attempts = Attempt.objects.filter(
                user=user
            ).select_related("question").order_by("-attempted_at")[:20]

            history = [
                {
                    "question"    : a.question.content[:60],
                    "topic"       : a.question.topic,
                    "difficulty"  : a.question.difficulty,
                    "is_correct"  : a.is_correct,
                    "time_taken"  : a.time_taken,
                    "attempted_at": str(a.attempted_at)
                }
                for a in attempts
            ]

            states = KnowledgeState.objects.filter(
                user=user
            ).order_by("-skill_score")

            topic_progress = [
                {
                    "topic"      : ks.topic,
                    "skill_score": round(ks.skill_score, 4),
                    "level"      : "beginner"      if ks.skill_score < 0.4
                                   else "intermediate" if ks.skill_score < 0.7
                                   else "advanced",
                    "updated_at" : str(ks.updated_at)
                }
                for ks in states
            ]

            return Response({
                "user_id"        : str(user.id),
                "recent_attempts": history,
                "topic_progress" : topic_progress
            })
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)


class MyKnowledgeView(APIView):
    def get(self, request, user_id):
        try:
            user   = User.objects.get(id=user_id)
            states = KnowledgeState.objects.filter(
                user=user
            ).order_by("-skill_score")

            if not states:
                return Response({
                    "user_id": str(user.id),
                    "message": "No knowledge state yet — answer some questions!",
                    "topics" : []
                })

            return Response({
                "user_id": str(user.id),
                "topics" : [
                    {
                        "topic"      : ks.topic,
                        "skill_score": round(ks.skill_score, 4),
                        "level"      : "beginner"      if ks.skill_score < 0.4
                                       else "intermediate" if ks.skill_score < 0.7
                                       else "advanced",
                        "updated_at" : str(ks.updated_at)
                    }
                    for ks in states
                ]
            })
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=404)


class AllUsersView(APIView):
    def get(self, request):
        users = User.objects.all()[:20]
        return Response([
            {"user_id": str(u.id), "username": u.username, "email": u.email}
            for u in users
        ])


# ── Training View ──────────────────────────────────────────────────────

class RetrainView(APIView):
    def post(self, request):
        import threading
        def run():
            from src.pipeline.training_pipeline import TrainingPipeline
            TrainingPipeline().run()
        threading.Thread(target=run, daemon=True).start()
        return Response({
            "status" : "started",
            "message": "Retraining running in background"
        })