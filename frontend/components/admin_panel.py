import streamlit as st

from auth import repository


def render_admin_panel() -> None:
    st.subheader("Admin panel")
    st.caption("Users awaiting approval")

    pending = repository.list_unverified()
    if not pending:
        st.success("No users awaiting approval.")
        return

    for user in pending:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            st.write(user.email)
        with col2:
            st.write(user.auth_provider.capitalize())
        with col3:
            st.write(user.created_at.strftime("%Y-%m-%d %H:%M"))
        with col4:
            if st.button("Approve", key=f"approve_{user.id}"):
                repository.approve_user(user.id)
                st.success(f"Approved: {user.email}")
                st.rerun()

        with st.expander("Registration details"):
            if user.join_reason:
                st.markdown("**Reason to join:**")
                st.write(user.join_reason)
            elif user.associated_organisation:
                st.markdown("**Associated organisation:**")
                st.write(user.associated_organisation)
                st.markdown("**Associated e-mail:**")
                st.write(user.associated_org_email or "-")
            else:
                st.write("No additional information provided.")
