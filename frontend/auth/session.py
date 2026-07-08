from typing import Optional
from uuid import UUID

import streamlit as st

from auth import repository
from auth.constants import ACCOUNT_DISABLED_MESSAGE
from auth.oauth_sync import sync_oauth_user
from auth.passwords import verify_password


def is_logged_in() -> bool:
    return bool(st.user.is_logged_in or st.session_state.get("auth_user_id"))


def get_current_user() -> Optional[repository.User]:
    if st.user.is_logged_in:
        return sync_oauth_user()

    user_id = st.session_state.get("auth_user_id")
    if not user_id:
        return None
    return repository.get_by_id(UUID(user_id))


def login_with_email(email: str, password: str) -> tuple[bool, str]:
    user = repository.get_by_email(email)
    if not user:
        return False, "Invalid email or password."
    if user.auth_provider != "email":
        provider = user.auth_provider.capitalize()
        return False, f"This account uses {provider} login. Use the corresponding OAuth button."
    if not user.is_active:
        return False, ACCOUNT_DISABLED_MESSAGE
    if not user.password_hash or not verify_password(password, user.password_hash):
        return False, "Invalid email or password."

    repository.update_last_login(user.id)
    st.session_state["auth_user_id"] = str(user.id)
    return True, ""


def logout() -> None:
    st.session_state.pop("auth_user_id", None)
    if st.user.is_logged_in:
        st.logout()
