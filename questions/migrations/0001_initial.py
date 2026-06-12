import uuid

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Question",
            fields=[
                ("id", models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ("content", models.TextField()),
                ("topic", models.CharField(blank=True, max_length=100, null=True)),
                ("difficulty", models.FloatField(blank=True, null=True)),
                ("source", models.CharField(blank=True, max_length=100, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "db_table": "questions",
                "ordering": ["topic", "difficulty"],
            },
        ),
    ]
