"""Welcome page: pick between the AI planner and building a schedule by hand."""

import streamlit as st

import ui_shared as ui

owner = ui.get_owner()

ui.hero(
    "Pet care, planned around your day",
    'PawPal<span class="pp-accent">+</span>',
    "Tell us about your pets and your time, and we'll build a daily plan you can actually stick to.",
)

st.write("")
col1, col2 = st.columns(2, gap="medium")
with col1:
    with st.container(border=True, height="stretch"):
        st.markdown("### ✨ Plan with AI")
        st.markdown(
            "Describe your day in plain English. PawPal AI sets up your pets and tasks, "
            "checks the plan, and explains why."
        )
        st.caption("Best for: getting started fast.")
        if st.button("Start with AI", type="primary", icon="✨", width="stretch"):
            st.switch_page(ui.AI_PAGE)
with col2:
    with st.container(border=True, height="stretch"):
        st.markdown("### 📝 Build it myself")
        st.markdown(
            "Add your pets and their care tasks step by step, set priorities and preferred times, "
            "then generate your schedule."
        )
        st.caption("Best for: full control over every detail.")
        if st.button("Start building", icon="📝", width="stretch"):
            st.switch_page(ui.MANUAL_PAGE)

st.write("")
with st.container(border=True):
    st.markdown("#### 🐾 How PawPal AI works")
    steps = st.columns(5)
    for col, (icon, title, text) in zip(steps, [
        ("💬", "You ask", "Describe pets, needs, and free time"),
        ("📚", "Looks it up", "Finds matching care guidelines"),
        ("📝", "AI proposes", "Turns it into pets and tasks"),
        ("🔍", "Rules check", "No clashes, fits your time"),
        ("✅", "You approve", "Nothing changes until you apply"),
    ]):
        with col:
            st.markdown(f"<div style='text-align:center;font-size:1.8rem'>{icon}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center'><b>{title}</b><br>"
                        f"<span class='pp-muted'>{text}</span></div>", unsafe_allow_html=True)

if owner.get_pets():
    st.write("")
    with st.container(border=True):
        st.markdown("#### 🏡 Your household")
        ui.household_summary(owner)
