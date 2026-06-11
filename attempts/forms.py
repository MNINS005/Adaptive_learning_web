from django import forms

from questions.models import Question
from users.models import User


class AttemptForm(forms.Form):
    user = forms.ModelChoiceField(queryset=User.objects.order_by("username"))
    question = forms.ModelChoiceField(queryset=Question.objects.order_by("topic", "difficulty"))
    is_correct = forms.BooleanField(required=False)
    time_taken = forms.IntegerField(required=False, min_value=0)
