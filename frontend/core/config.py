import streamlit as st


def get_database_url() -> str:
    db = st.secrets["database"]
    host = db["host"]
    port = db["port"]
    dbname = db["dbname"]
    user = db["user"]
    password = db["password"]
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_recaptcha_site_key() -> str:
    return st.secrets["recaptcha"]["site_key"]


def get_recaptcha_secret_key() -> str:
    return st.secrets["recaptcha"]["secret_key"]


def get_recaptcha_min_score() -> float:
    return float(st.secrets["recaptcha"].get("min_score", 0.5))
