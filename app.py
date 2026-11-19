"""PawPal+ entry point. Run with:  streamlit run app.py

Sets up shared state and styling, then routes to one of three pages in views/:
  home.py          - welcome page: choose the AI planner or build it yourself
  ai_assistant.py  - natural-language planning with the PawPal AI agent
  manual.py        - the original owner / pets / tasks / schedule workflow
"""

import streamlit as st

import ui_shared as ui
from pawpal_ai import load_env, setup_logging

load_env()
setup_logging()

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

ui.init_state()
ui.inject_style()

page = st.navigation(
    [
        st.Page(ui.HOME_PAGE, title="Home", icon="🏠", default=True),
        st.Page(ui.AI_PAGE, title="AI Planner", icon="✨", url_path="ai"),
        st.Page(ui.MANUAL_PAGE, title="Build it myself", icon="📝", url_path="build"),
    ],
    position="hidden",  # we draw our own nav row instead of the sidebar
)
ui.top_nav()
page.run()
