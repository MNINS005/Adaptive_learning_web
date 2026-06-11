import uuid

from django.db import models


class KnowledgeState(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey("users.User", on_delete=models.CASCADE, related_name="knowledge_states")
    topic = models.CharField(max_length=100)
    skill_score = models.FloatField(default=0.3)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "knowledge_states"
        unique_together = ("user", "topic")
        ordering = ["-skill_score"]

    def __str__(self):
        return f"{self.user.username} | {self.topic} | {self.skill_score}"
