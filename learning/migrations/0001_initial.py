import uuid

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("users", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="KnowledgeState",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("topic", models.CharField(max_length=100)),
                ("skill_score", models.FloatField(default=0.3)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="knowledge_states",
                        to="users.user",
                    ),
                ),
            ],
            options={
                "db_table": "knowledge_states",
                "ordering": ["-skill_score"],
                "unique_together": {("user", "topic")},
            },
        ),
    ]
