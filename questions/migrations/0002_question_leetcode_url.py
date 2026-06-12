from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("questions", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="question",
            name="leetcode_url",
            field=models.CharField(blank=True, max_length=300, null=True),
        ),
    ]
