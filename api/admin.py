from django.contrib import admin
from .models import User, Question, Attempt, KnowledgeState


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display  = ["username", "email", "created_at"]
    search_fields = ["username", "email"]


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display  = ["content", "topic", "difficulty", "source"]
    list_filter   = ["topic", "difficulty"]
    search_fields = ["content", "topic"]


@admin.register(Attempt)
class AttemptAdmin(admin.ModelAdmin):
    list_display = ["user", "question", "is_correct", "time_taken", "attempted_at"]
    list_filter  = ["is_correct"]


@admin.register(KnowledgeState)
class KnowledgeStateAdmin(admin.ModelAdmin):
    list_display = ["user", "topic", "skill_score", "updated_at"]
    list_filter  = ["topic"]