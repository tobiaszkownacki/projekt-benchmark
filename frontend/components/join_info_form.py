from dataclasses import dataclass
from typing import Optional

import streamlit as st
from email_validator import EmailNotValidError, validate_email

_MODE_ORGANISATION = "Associated organisation"
_MODE_REASON = "Reason to join"


@dataclass
class JoinInfo:
    associated_organisation: Optional[str] = None
    associated_org_email: Optional[str] = None
    join_reason: Optional[str] = None


def validate_join_info(info: JoinInfo, mode: str) -> Optional[str]:
    if mode == _MODE_ORGANISATION:
        if not (info.associated_organisation or "").strip():
            return "Please enter the associated organisation."
        if not (info.associated_org_email or "").strip():
            return "Please enter the associated e-mail."
        try:
            validate_email(info.associated_org_email, check_deliverability=False)
        except EmailNotValidError as exc:
            return str(exc)
    else:
        if not (info.join_reason or "").strip():
            return "Please enter your reason to join."
    return None


def render_join_info_inputs(key_prefix: str) -> tuple[JoinInfo, str]:
    mode = st.radio(
        "How would you like to introduce yourself?",
        [_MODE_ORGANISATION, _MODE_REASON],
        key=f"{key_prefix}_join_mode",
    )

    info = JoinInfo()
    if mode == _MODE_ORGANISATION:
        info.associated_organisation = st.text_input(
            "Associated organisation", key=f"{key_prefix}_organisation"
        )
        info.associated_org_email = st.text_input(
            "Associated e-mail", key=f"{key_prefix}_org_email"
        )
    else:
        info.join_reason = st.text_area(
            "Reason to join", key=f"{key_prefix}_reason"
        )

    return info, mode
