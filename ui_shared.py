"""Shared UI pieces for the PawPal+ Streamlit pages: state, styling, navigation, schedule display."""

from __future__ import annotations

import base64
from datetime import date, datetime, time
from pathlib import Path

import streamlit as st

from pawpal_ai import GeminiClient
from pawpal_system import Owner, Scheduler, Task

ASSETS = Path(__file__).with_name("assets")

HOME_PAGE = "views/home.py"
AI_PAGE = "views/ai_assistant.py"
MANUAL_PAGE = "views/manual.py"

CATEGORY_ICONS = {"exercise": "🦮", "feeding": "🍖", "medication": "💊", "grooming": "🛁", "play": "🎾", "other": "🐾"}
PRIORITY_LABELS = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}
SPECIES_ICONS = {"dog": "🐕", "cat": "🐈"}


# --- State ------------------------------------------------------------------------------


def init_state() -> Owner:
    """Create the shared Owner once per session and keep the owner widgets in sync with it."""
    if "owner" not in st.session_state:
        st.session_state.owner = Owner(name="Jordan", available_minutes=120)
    owner: Owner = st.session_state.owner
    # Widget values are dropped when their page isn't shown, so refill them from the Owner each run.
    st.session_state.setdefault("owner_name", owner.name)
    st.session_state.setdefault("owner_minutes", owner.available_minutes)
    st.session_state.setdefault("owner_start", owner.day_start)
    return owner


def get_owner() -> Owner:
    return st.session_state.owner


@st.cache_resource
def get_llm():
    """Create the Gemini client once per server; return (client, error message)."""
    try:
        return GeminiClient(), None
    except Exception as exc:  # missing key or SDK
        return None, str(exc)


# --- Styling ----------------------------------------------------------------------------


@st.cache_data
def _svg_data_uri(name: str) -> str:
    data = (ASSETS / name).read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode()


def inject_style() -> None:
    """Add the warm background glow, the side pet illustrations, and small style touches."""
    dog, cat = _svg_data_uri("dog.svg"), _svg_data_uri("cat.svg")
    st.html(f"""
<style>
  .stApp {{
    background:
      radial-gradient(ellipse at 50% -10%, rgba(255,154,92,0.20), transparent 55%),
      radial-gradient(ellipse at 50% 120%, rgba(255,209,102,0.08), transparent 50%),
      #1E1714;
  }}
  header, .stAppHeader, [data-testid="stHeader"] {{ background: transparent !important; }}
  [data-testid^="stBaseButton-primary"],
  [data-testid^="stBaseButton-primary"] * {{ color: #2A1A12 !important; font-weight: 700 !important; }}
  .pp-pet {{
    position: fixed; top: 50%; width: min(210px, 13vw); z-index: 0;
    pointer-events: none; transform: translateY(-40%);
    filter: drop-shadow(0 14px 22px rgba(0,0,0,0.45));
    animation: pp-bob 4.5s ease-in-out infinite;
  }}
  .pp-pet.left {{ left: 4vw; }}
  .pp-pet.right {{ right: 4vw; animation-delay: -2.25s; }}
  @keyframes pp-bob {{
    0%, 100% {{ transform: translateY(-40%); }}
    50% {{ transform: translateY(calc(-40% - 12px)); }}
  }}
  @media (max-width: 1200px) {{ .pp-pet {{ display: none; }} }}
  @media (prefers-reduced-motion: reduce) {{ .pp-pet {{ animation: none; }} }}

  .pp-hero {{ text-align: center; padding: 0.5rem 0 0.75rem; }}
  .pp-hero .pp-eyebrow {{
    color: #FF9A5C; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; font-size: 0.8rem;
  }}
  .pp-hero h1 {{ font-family: 'Fredoka', sans-serif; font-size: 3.2rem; margin: 0.1rem 0 0.2rem; padding: 0; }}
  .pp-hero p {{ color: #E6CDBD; font-size: 1.15rem; margin: 0; }}
  .pp-accent {{ color: #FF9A5C; }}
  .pp-muted {{ color: #BFA697; font-size: 0.95rem; }}
</style>
<img class="pp-pet left" src="{dog}" alt="">
<img class="pp-pet right" src="{cat}" alt="">
""")


def hero(eyebrow: str, title_html: str, subtitle: str) -> None:
    """Big centered page header."""
    st.html(f"""
<div class="pp-hero">
  <div class="pp-eyebrow">{eyebrow}</div>
  <h1>{title_html}</h1>
  <p>{subtitle}</p>
</div>
""")


def top_nav() -> None:
    """Small navigation row shown on every page."""
    with st.container(horizontal=True, horizontal_alignment="center", gap="small"):
        st.page_link(HOME_PAGE, label="Home", icon="🏠")
        st.page_link(AI_PAGE, label="AI Planner", icon="✨")
        st.page_link(MANUAL_PAGE, label="Build it myself", icon="📝")


# --- Display helpers --------------------------------------------------------------------------


def pet_label(pet) -> str:
    return f"{SPECIES_ICONS.get(pet.species, '🐾')} {pet.name}"


def task_status(task: Task) -> str:
    """Short label: when this task was done, or when it's next due."""
    if task.last_completed:
        return f"✅ done {task.last_completed:%b %d}"
    if task.due_date and task.due_date > date.today():
        return f"🗓️ due {task.due_date:%b %d}"
    return "⏳ due now"


def fmt_clock(t: time | datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def household_summary(owner: Owner) -> None:
    """One friendly line describing the current household and time budget."""
    pets = owner.get_pets()
    open_tasks = sum(1 for p in pets for t in p.tasks if t.last_completed is None)
    if pets:
        who = " · ".join(pet_label(p) for p in pets)
        st.markdown(
            f"**{who}** &nbsp;|&nbsp; ⏱️ {owner.available_minutes} min from {fmt_clock(owner.day_start)}"
            f" &nbsp;|&nbsp; 📋 {open_tasks} task{'s' if open_tasks != 1 else ''}"
        )
    else:
        st.markdown('<span class="pp-muted">No pets yet. Starting fresh.</span>', unsafe_allow_html=True)


def render_plan(scheduler: Scheduler, show_issues: bool = True) -> None:
    """Show a generated plan: time-budget bar, the timetable, then (optionally) skipped tasks and conflicts."""
    owner, plan = scheduler.owner, scheduler.plan
    if not plan and not scheduler.skipped:
        st.info("No tasks are due today. 🎉")
        return

    used = sum(e.task.duration_minutes for e in plan)
    budget = owner.available_minutes
    st.progress(min(used / budget, 1.0) if budget else 1.0,
                text=f"⏱️ {used} of {budget} minutes used · {scheduler.day:%A, %B %d}")

    if plan:
        st.dataframe(
            [
                {
                    "Time": f"{fmt_clock(e.start)} – {fmt_clock(e.end)}",
                    "Pet": pet_label(e.pet),
                    "Task": f"{CATEGORY_ICONS.get(e.task.category, '🐾')} {e.task.title}",
                    "Priority": PRIORITY_LABELS[e.task.priority],
                    "Why": e.reason,
                }
                for e in plan
            ],
            hide_index=True,
        )
    if not show_issues:
        return
    for pet, task, reason in scheduler.skipped:
        st.warning(f"Skipped **{task.title}** for {pet.name} ({task.priority}): {reason}", icon="⏭️")
    for warning in scheduler.conflict_warnings():
        st.warning(warning.removeprefix("Warning: "), icon="⚠️")
