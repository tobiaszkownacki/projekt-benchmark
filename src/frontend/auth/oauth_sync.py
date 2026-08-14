import streamlit as st

from auth import repository


def _detect_provider() -> str:
    iss = str(getattr(st.user, "iss", "") or "")
    if "google" in iss:
        return "google"
    if "microsoft" in iss:
        return "microsoft"
    return "unknown"


def sync_oauth_user() -> repository.User | None:
    if not st.user.is_logged_in:
        return None

    email = getattr(st.user, "email", None)
    if not email:
        st.error("Failed to retrieve email address from OAuth account.")
        st.logout()
        return None

    user = repository.upsert_oauth_user(
        email=email,
        oauth_sub=str(getattr(st.user, "sub", "") or email),
        auth_provider=_detect_provider(),
        display_name=getattr(st.user, "name", None),
    )

    return user
