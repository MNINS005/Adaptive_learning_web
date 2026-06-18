from .models import User


def current_user(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return {"current_user": None}

    try:
        return {"current_user": User.objects.get(id=user_id)}
    except User.DoesNotExist:
        request.session.pop("user_id", None)
        return {"current_user": None}
