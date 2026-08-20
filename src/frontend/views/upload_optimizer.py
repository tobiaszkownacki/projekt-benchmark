import streamlit as st
from src.frontend.core.validator import run_synchronous_validation

st.title("Upload your optimizer")

uploaded_file = st.file_uploader("Choose file with optimizer", type=["py"])

if uploaded_file is not None:
    user_code = uploaded_file.getvalue().decode("utf-8")

    if st.button("Validate and start benchmark"):
        with st.spinner("Validation of optimizer in safe sandbox container..."):

            result = run_synchronous_validation(user_code)

            if result["status"] == "success":
                st.success("Validation succeeded!")
                st.code(result["logs"], language="text")
                # Here need to send user_code do RabbitMQ
            elif result["status"] == "timeout":
                st.error("Error: " + result["logs"])
            else:
                st.error("Validation failed. Correct the code and retrieve it again.")
                st.code(result["logs"], language="text")
