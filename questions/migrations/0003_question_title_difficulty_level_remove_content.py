from django.db import migrations, models


def copy_content_to_title(apps, schema_editor):
    Question = apps.get_model("questions", "Question")
    for question in Question.objects.all():
        old_content = getattr(question, "content", "") or ""
        first_line = old_content.strip().splitlines()[0] if old_content.strip() else ""
        question.title = first_line[:255] or "Untitled Question"
        question.save(update_fields=["title"])


class Migration(migrations.Migration):
    dependencies = [
        ("questions", "0002_question_leetcode_url"),
    ]

    operations = [
        migrations.AddField(
            model_name="question",
            name="title",
            field=models.CharField(blank=True, max_length=255),
        ),
        migrations.AddField(
            model_name="question",
            name="difficulty_level",
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
        migrations.RunPython(copy_content_to_title, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="question",
            name="title",
            field=models.CharField(max_length=255),
        ),
        migrations.RemoveField(
            model_name="question",
            name="content",
        ),
    ]
