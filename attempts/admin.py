from django.contrib import admin

from .models import Attempt


@admin.register(Attempt)
class AttemptAdmin(admin.ModelAdmin):
    list_display = ("user", "question", "is_correct", "attempted_at")
    list_filter = ("is_correct", "attempted_at")
    search_fields = ("user__username", "question__content")
