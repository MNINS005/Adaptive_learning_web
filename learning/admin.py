from django.contrib import admin

from .models import KnowledgeState


@admin.register(KnowledgeState)
class KnowledgeStateAdmin(admin.ModelAdmin):
    list_display = ("user", "topic", "skill_score", "updated_at")
    list_filter = ("topic",)
    search_fields = ("user__username", "topic")
