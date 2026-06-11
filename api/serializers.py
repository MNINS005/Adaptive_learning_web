from rest_framework import serializers
from .models import User, Question, Attempt, KnowledgeState


class UserCreateSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=100)
    email    = serializers.EmailField()
    password = serializers.CharField(max_length=255)


class LoginSerializer(serializers.Serializer):
    email    = serializers.EmailField()
    password = serializers.CharField(max_length=255)


class UserOutSerializer(serializers.ModelSerializer):
    class Meta:
        model  = User
        fields = ["id", "username", "email", "created_at"]


class AttemptCreateSerializer(serializers.Serializer):
    user_id     = serializers.UUIDField()
    question_id = serializers.UUIDField()
    is_correct  = serializers.BooleanField()
    time_taken  = serializers.IntegerField(required=False, allow_null=True)


class AttemptOutSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Attempt
        fields = ["id", "is_correct", "time_taken", "attempted_at"]


class QuestionOutSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Question
        fields = ["id", "content", "topic", "difficulty",
                  "source", "leetcode_url"]


class KnowledgeStateSerializer(serializers.ModelSerializer):
    class Meta:
        model  = KnowledgeState
        fields = ["topic", "skill_score", "updated_at"]