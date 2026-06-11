import uuid
from django.db import models


class User(models.Model):
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    username   = models.CharField(max_length=100, unique=True)
    email      = models.EmailField(unique=True)
    password   = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "users"

    def __str__(self):
        return self.username


class Question(models.Model):
    id         = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content    = models.TextField()
    topic      = models.CharField(max_length=100, blank=True, null=True)
    difficulty = models.FloatField(blank=True, null=True)
    source     = models.CharField(max_length=100, blank=True, null=True)
    #leetcode_slug = models.CharField(max_length=200, blank=True, null=True)
    leetcode_url  = models.CharField(max_length=300, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "questions"

    def __str__(self):
        return f"[{self.topic}] {self.content[:50]}"


class Attempt(models.Model):
    id           = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user         = models.ForeignKey(User,     on_delete=models.CASCADE, related_name="attempts")
    question     = models.ForeignKey(Question, on_delete=models.CASCADE, related_name="attempts")
    is_correct   = models.BooleanField()
    time_taken   = models.IntegerField(blank=True, null=True)
    attempted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "attempts"

    def __str__(self):
        return f"{self.user.username} → {self.question.content[:30]} → {self.is_correct}"


class KnowledgeState(models.Model):
    id          = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name="knowledge_states")
    topic       = models.CharField(max_length=100)
    skill_score = models.FloatField(default=0.3)
    updated_at  = models.DateTimeField(auto_now=True)

    class Meta:
        db_table        = "knowledge_states"
        unique_together = ("user", "topic")

    def __str__(self):
        return f"{self.user.username} | {self.topic} | {self.skill_score}"
