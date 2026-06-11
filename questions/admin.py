from django.contrib import admin

from .models import Question


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("content_preview", "topic", "difficulty", "source", "created_at")
    list_filter = ("topic", "source")
    search_fields = ("content", "topic")

    @admin.display(description="question")
    def content_preview(self, obj):
        return obj.content[:80]
