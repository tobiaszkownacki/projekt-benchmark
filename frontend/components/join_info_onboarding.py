import streamlit as st

from auth import repository
from auth.session import logout
from components.join_info_form import render_join_info_inputs, validate_join_info


def render_join_info_onboarding(user: repository.User) -> None:
    st.title("Tell us a bit more")
    st.info(
        "Before an administrator can review your account, please provide "
        "the information below."
    )

    join_info, join_mode = render_join_info_inputs("onboarding")

    if st.button("Submit", use_container_width=True, key="onboarding_submit"):
        join_error = validate_join_info(join_info, join_mode)
        if join_error:
            st.error(join_error)
            return

        repository.set_join_info(
            user.id,
            associated_organisation=join_info.associated_organisation,
            associated_org_email=join_info.associated_org_email,
            join_reason=join_info.join_reason,
        )
        st.rerun()

    st.button("Log out", on_click=logout, key="onboarding_logout")
