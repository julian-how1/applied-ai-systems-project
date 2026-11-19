"""AI planner page: natural-language request -> checked proposal -> preview -> apply."""

import streamlit as st

import ui_shared as ui
from pawpal_ai import PawPalAgent, apply_proposal

owner = ui.get_owner()

EXAMPLES = {
    "🐶 New dog": "I have a 3 year old dog named Rex. I'm free for 90 minutes starting at 7am. "
                 "He needs a 30 minute walk, breakfast at 7:30, and his allergy pill with breakfast.",
    "🐱🐶 Two pets": "Plan my day. I've got about two hours. My cat Luna (2) needs feeding twice a day and "
                     "her litter cleaned. My dog Ollie (6) needs a long walk and some play time.",
    "⏰ Short on time": "I only have 45 minutes today. Make sure the important stuff still happens.",
    "🔧 Fix clashes": "My schedule feels crammed. Can you fix any clashes?",
}
STAGE_ICONS = {"interpret": "🧠", "retrieve": "📚", "propose": "📝", "check": "🔍", "explain": "💬"}


def use_example(text: str) -> None:
    st.session_state.ai_request = text


def apply_ai_changes() -> None:
    """Button callback: re-validate and apply the AI's proposal, then sync the owner widgets."""
    result = st.session_state.get("ai_result")
    if result is None:
        return
    report = apply_proposal(owner, result.actions)
    if report.ok:
        st.session_state.owner_minutes = owner.available_minutes
        st.session_state.owner_start = owner.day_start
        st.session_state.ai_flash = ("success", f"Applied {len(report.applied)} change(s) to your household!")
        st.session_state.ai_result = None
    else:
        st.session_state.ai_flash = ("error", "Your data changed since the AI ran, so the proposal no longer fits: "
                                     + "; ".join(report.errors))


def discard_ai_changes() -> None:
    st.session_state.ai_result = None


ui.hero(
    "AI Planner",
    '✨ Plan with <span class="pp-accent">PawPal AI</span>',
    "Describe your pets and your day. I'll set things up and double-check the plan.",
)

llm, llm_error = ui.get_llm()
if llm is None:
    st.info(f"The AI planner is offline: {llm_error}", icon="🔌")

with st.container(border=True):
    ui.household_summary(owner)

    st.markdown("**Try an example:**")
    with st.container(horizontal=True, gap="small"):
        for label, text in EXAMPLES.items():
            st.button(label, on_click=use_example, args=(text,), type="tertiary")

    request = st.text_area(
        "What do you need?",
        key="ai_request",
        height=110,
        placeholder="e.g. I have a 3 year old dog named Rex. I'm free for 90 minutes starting at 7am...",
        label_visibility="collapsed",
    )
    if st.button("Ask PawPal AI", type="primary", icon="✨", width="stretch",
                 disabled=llm is None or not request.strip()):
        with st.spinner("Reading your request, drafting a plan, and checking it against the rules..."):
            st.session_state.ai_result = PawPalAgent(llm).run(owner, request)
        st.session_state.pop("ai_flash", None)

flash = st.session_state.pop("ai_flash", None)
if flash:
    kind, message = flash
    if kind == "success":
        st.success(message, icon="🎉")
        st.page_link(ui.MANUAL_PAGE, label="See your pets, tasks, and schedule", icon="📝")
    else:
        st.error(message)

result = st.session_state.get("ai_result")
if result is not None:
    with st.container(border=True):
        if result.error:
            st.error(result.error, icon="😿")
        else:
            with st.container(horizontal=True, gap="small"):
                if result.accepted:
                    st.badge("Passed all schedule checks", icon="✅", color="green")
                else:
                    st.badge("Needs attention", icon="⚠️", color="orange")
                st.badge(f"{result.mode.title()} mode", icon="🧭", color="gray")
                st.badge(f"{result.rounds} round{'s' if result.rounds != 1 else ''}", icon="🔁", color="gray")

            st.markdown(f"#### 🧠 What I understood\n{result.proposal.understanding or '(no summary)'}")
            if result.proposal.notes:
                st.caption(f"📌 {result.proposal.notes}")

            if not result.accepted:
                st.error("I couldn't make a plan that passes every check:\n\n"
                         + "\n".join(f"- {err}" for err in result.report.errors))

            st.markdown("#### 📝 Proposed changes")
            if result.report.applied:
                st.markdown("\n".join(f"- {line}" for line in result.report.applied))
            else:
                st.caption("No changes needed. Your current setup already handles this.")

            if result.scheduler is not None:
                st.markdown("#### 📅 Schedule preview")
                ui.render_plan(result.scheduler, show_issues=False)  # the validator's warnings cover these
            for warning in result.report.warnings:
                st.warning(warning, icon="⚠️")

            st.markdown("#### 💬 Why this plan")
            if result.explanation.startswith("Daily plan"):  # built-in fallback text is preformatted
                st.text(result.explanation)
            else:
                st.markdown(result.explanation)

            st.write("")
            col1, col2 = st.columns(2)
            with col1:
                st.button("Apply changes", type="primary", icon="✅", width="stretch",
                          on_click=apply_ai_changes, disabled=not (result.accepted and result.actions))
            with col2:
                st.button("Discard", icon="🗑️", width="stretch", on_click=discard_ai_changes)

        if result.guidelines:
            with st.expander(f"📚 Care guidelines used ({len(result.guidelines)})"):
                st.caption("Looked up in PawPal's knowledge base (data/pet_care_guidelines.csv) "
                           "before the AI answered.")
                st.dataframe(
                    [
                        {
                            "ID": g.id,
                            "For": ("any pet" if g.species == "any" else g.species)
                                   + ("" if g.life_stage == "any" else f" ({g.life_stage})"),
                            "Task": f"{ui.CATEGORY_ICONS.get(g.category, '🐾')} {g.task}",
                            "Minutes": g.duration_min,
                            "Priority": ui.PRIORITY_LABELS[g.priority],
                            "Timing": g.timing,
                            "Note": g.notes,
                        }
                        for g in result.guidelines
                    ],
                    hide_index=True,
                )

        with st.expander("🔍 How the AI got here"):
            for step in result.steps:
                st.markdown(f"{STAGE_ICONS.get(step.stage, '•')} **{step.stage.title()}**: {step.detail}")
