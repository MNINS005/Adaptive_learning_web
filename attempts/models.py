import uuid

from django.db import models


class Attempt(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey("users.User", on_delete=models.CASCADE, related_name="attempts")
    question = models.ForeignKey("questions.Question", on_delete=models.CASCADE, related_name="attempts")
    is_correct = models.BooleanField()
    time_taken = models.IntegerField(blank=True, null=True)
    attempted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "attempts"
        ordering = ["-attempted_at"]

    def __str__(self):
        return f"{self.user.username} -> {self.question.content[:30]} -> {self.is_correct}"
