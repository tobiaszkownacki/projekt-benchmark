import httpx

from core.config import get_recaptcha_min_score, get_recaptcha_secret_key


def verify_recaptcha(token: str, *, action: str = "register") -> bool:
    secret = get_recaptcha_secret_key()
    if not secret or secret == "CHANGE_ME" or not token:
        return False

    response = httpx.post(
        "https://www.google.com/recaptcha/api/siteverify",
        data={"secret": secret, "response": token},
        timeout=10.0,
    )
    response.raise_for_status()
    payload = response.json()

    if not payload.get("success"):
        return False
    if payload.get("action") and payload["action"] != action:
        return False
    return payload.get("score", 0.0) >= get_recaptcha_min_score()
