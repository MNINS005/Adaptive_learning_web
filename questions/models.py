import uuid

from django.db import models


class Question(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    content = models.TextField()
    topic = models.CharField(max_length=100, blank=True, null=True)
    difficulty = models.FloatField(blank=True, null=True)
    source = models.CharField(max_length=100, blank=True, null=True)
    leetcode_url = models.CharField(max_length=300, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "questions"
        ordering = ["topic", "difficulty"]

    def __str__(self):
        return f"[{self.topic}] {self.content[:50]}"
