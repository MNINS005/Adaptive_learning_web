import csv
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from questions.models import Question


DIFFICULTY_MAP = {
    "easy": 0.25,
    "medium": 0.55,
    "hard": 0.85,
}

TOPIC_MAP = {
    "array": "arrays",
    "hash table": "arrays",
    "linked list": "linked_lists",
    "tree": "trees",
    "binary tree": "trees",
    "graph": "graphs",
    "dynamic programming": "dynamic_programming",
    "sorting": "sorting",
    "binary search": "searching",
    "recursion": "recursion",
}


class Command(BaseCommand):
    help = "Import or update LeetCode questions from the local CSV dataset."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default="notebook/leetcode_dataset - lc.csv",
            help="Path to the LeetCode CSV file.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=300,
            help="Maximum rows to import. Use 0 to import all rows.",
        )

    def handle(self, *args, **options):
        csv_path = Path(options["path"])
        if not csv_path.exists():
            raise CommandError(f"CSV file not found: {csv_path}")

        limit = options["limit"]
        created = 0
        updated = 0

        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader, start=1):
                if limit and index > limit:
                    break

                title = (row.get("title") or "").strip()
                url = (row.get("url") or "").strip()
                description = (row.get("description") or "").strip()
                difficulty_label = (row.get("difficulty") or "").strip().lower()
                related_topics = row.get("related_topics") or ""

                if not title:
                    continue

                topic = self.get_topic(related_topics)
                difficulty = DIFFICULTY_MAP.get(difficulty_label, 0.5)
                content = f"{title}\n\n{description}" if description else title

                question, was_created = Question.objects.update_or_create(
                    leetcode_url=url or None,
                    defaults={
                        "content": content,
                        "topic": topic,
                        "difficulty": difficulty,
                        "source": "leetcode",
                    },
                )

                if was_created:
                    created += 1
                else:
                    updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Import complete. Created {created} questions, updated {updated} questions."
            )
        )

    def get_topic(self, related_topics):
        topics = [topic.strip().lower() for topic in related_topics.split(",")]
        for topic in topics:
            if topic in TOPIC_MAP:
                return TOPIC_MAP[topic]
        return "general_cs"
