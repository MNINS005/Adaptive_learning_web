from django.contrib import admin

from .models import Question


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("title", "topic", "difficulty_level", "difficulty", "source", "leetcode_url", "created_at")
    list_filter = ("topic", "source")
    search_fields = ("title", "topic", "leetcode_url")
