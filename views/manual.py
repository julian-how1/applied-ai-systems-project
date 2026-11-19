"""Build-it-myself page: the original PawPal+ owner / pets / tasks / schedule workflow."""

from datetime import time

import streamlit as st

import ui_shared as ui
from pawpal_ai import CATEGORIES
from pawpal_system import Pet, Scheduler, Task

PRIORITIES = ["high", "medium", "low"]
FREQUENCIES = ["daily", "weekly"]

owner = ui.get_owner()

ui.hero(
    "Build it myself",
    '📝 Your <span class="pp-accent">care plan</span>',
    "Set up your day, your pets, and their tasks, then let the scheduler fit it all together.",
)

tab_you, tab_pets, tab_tasks, tab_today = st.tabs(["👤 You", "🐾 Pets", "📋 Tasks", "📅 Today's schedule"])

# --- You ----------------------------------------------------------------------------------

with tab_you:
    with st.container(border=True):
        st.markdown("#### How much time do you have today?")
        col1, col2, col3 = st.columns(3)
        with col1:
            owner.name = st.text_input("Your name", key="owner_name")
        with col2:
            owner.set_availability(int(st.number_input(
                "Minutes available", min_value=0, max_value=1440, step=5, key="owner_minutes")))
        with col3:
            owner.day_start = st.time_input("Day starts at", key="owner_start")
        st.caption("High-priority care is scheduled first, then everything else that fits in your time.")

# --- Pets ---------------------------------------------------------------------------------

with tab_pets:
    with st.container(border=True):
        st.markdown("#### Who are we caring for?")
        with st.form("add_pet", clear_on_submit=True, border=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                pet_name = st.text_input("Pet name", value="Mochi")
            with col2:
                species = st.selectbox("Species", ["dog", "cat", "other"],
                                       format_func=lambda s: f"{ui.SPECIES_ICONS.get(s, '🐾')} {s}")
            with col3:
                age = st.number_input("Age", min_value=0, max_value=40, value=3)
            if st.form_submit_button("Add pet", icon="➕", type="primary"):
                if not pet_name.strip():
                    st.error("Pet name can't be empty.")
                elif any(p.name == pet_name.strip() for p in owner.get_pets()):
                    st.error(f"{owner.name} already has a pet named {pet_name.strip()}.")
                else:
                    owner.add_pet(Pet(name=pet_name.strip(), species=species, age=int(age)))

    if owner.get_pets():
        cols = st.columns(min(len(owner.get_pets()), 3))
        for i, pet in enumerate(owner.get_pets()):
            with cols[i % len(cols)]:
                with st.container(border=True):
                    st.markdown(f"### {ui.pet_label(pet)}")
                    open_tasks = sum(1 for t in pet.tasks if t.last_completed is None)
                    st.caption(f"{pet.species.title()} · {pet.age} yr{'s' if pet.age != 1 else ''} · "
                               f"{open_tasks} open task{'s' if open_tasks != 1 else ''}")
    else:
        st.info("No pets yet. Add one above.", icon="🐾")

# --- Tasks --------------------------------------------------------------------------------

with tab_tasks:
    if not owner.get_pets():
        st.info("Add a pet first, then you can give it tasks.", icon="🐾")
    else:
        with st.container(border=True):
            st.markdown("#### Add a care task")
            with st.form("add_task", clear_on_submit=True, border=False):
                pet_choice = st.selectbox("For pet", [p.name for p in owner.get_pets()])
                col1, col2 = st.columns(2)
                with col1:
                    task_title = st.text_input("Task title", value="Morning walk")
                    category = st.selectbox("Category", CATEGORIES,
                                            format_func=lambda c: f"{ui.CATEGORY_ICONS[c]} {c}")
                    duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
                with col2:
                    priority = st.selectbox("Priority", PRIORITIES, format_func=ui.PRIORITY_LABELS.get)
                    frequency = st.selectbox("Frequency", FREQUENCIES)
                    pin_time = st.checkbox("Set a preferred time")
                    preferred = st.time_input("Preferred time", value=time(8, 0))
                if st.form_submit_button("Add task", icon="➕", type="primary"):
                    if not task_title.strip():
                        st.error("Task title can't be empty.")
                    else:
                        pet = next(p for p in owner.get_pets() if p.name == pet_choice)
                        pet.add_task(
                            Task(
                                title=task_title.strip(),
                                category=category,
                                duration_minutes=int(duration),
                                priority=priority,
                                frequency=frequency,
                                preferred_time=preferred if pin_time else None,
                            )
                        )
                        st.rerun()  # refresh the task list below

        all_tasks = [(pet, task) for pet in owner.get_pets() for task in pet.get_tasks()]
        if all_tasks:
            st.dataframe(
                [
                    {
                        "Pet": ui.pet_label(pet),
                        "Task": f"{ui.CATEGORY_ICONS.get(task.category, '🐾')} {task.title}",
                        "Minutes": task.duration_minutes,
                        "Priority": ui.PRIORITY_LABELS[task.priority],
                        "Repeats": task.frequency,
                        "Preferred time": ui.fmt_clock(task.preferred_time) if task.preferred_time else "flexible",
                        "Status": ui.task_status(task),
                    }
                    for pet, task in all_tasks
                ],
                hide_index=True,
            )
        else:
            st.info("No tasks yet. Add one above.", icon="📋")

        if all_tasks:
            with st.container(border=True):
                st.markdown("#### Manage a task")
                idx = st.selectbox(
                    "Task",
                    range(len(all_tasks)),
                    format_func=lambda i: f"{all_tasks[i][0].name}: {all_tasks[i][1].title} "
                                          f"({ui.task_status(all_tasks[i][1])})",
                )
                sel_pet, sel_task = all_tasks[idx]

                col1, col2 = st.columns(2)
                with col1:
                    if sel_task.last_completed is not None:
                        if st.button("Mark not done", icon="↩️", width="stretch"):
                            sel_pet.undo_complete(sel_task)
                            st.rerun()
                    elif st.button("Mark done today", icon="✅", width="stretch"):
                        sel_pet.complete_task(sel_task)  # also schedules the next daily/weekly occurrence
                        st.rerun()
                with col2:
                    if st.button("Remove task", icon="🗑️", width="stretch"):
                        sel_pet.remove_task(sel_task)
                        st.rerun()

                # Keys use the task object's id so the form refills with the right values when the selection changes.
                tid = id(sel_task)
                with st.expander("✏️ Edit this task"):
                    with st.form(f"edit_task_{tid}", border=False):
                        col1, col2 = st.columns(2)
                        with col1:
                            new_title = st.text_input("Title", value=sel_task.title, key=f"edit_title_{tid}")
                            new_category = st.selectbox(
                                "Category", CATEGORIES, index=CATEGORIES.index(sel_task.category),
                                format_func=lambda c: f"{ui.CATEGORY_ICONS[c]} {c}", key=f"edit_cat_{tid}",
                            )
                            new_duration = st.number_input(
                                "Duration (minutes)", min_value=1, max_value=240,
                                value=sel_task.duration_minutes, key=f"edit_dur_{tid}",
                            )
                        with col2:
                            new_priority = st.selectbox(
                                "Priority", PRIORITIES, index=PRIORITIES.index(sel_task.priority),
                                format_func=ui.PRIORITY_LABELS.get, key=f"edit_pri_{tid}",
                            )
                            new_frequency = st.selectbox(
                                "Frequency", FREQUENCIES, index=FREQUENCIES.index(sel_task.frequency),
                                key=f"edit_freq_{tid}",
                            )
                            new_pin = st.checkbox(
                                "Set a preferred time", value=sel_task.preferred_time is not None,
                                key=f"edit_pin_{tid}",
                            )
                            new_time = st.time_input(
                                "Preferred time", value=sel_task.preferred_time or time(8, 0),
                                key=f"edit_time_{tid}",
                            )
                        if st.form_submit_button("Save changes", icon="💾", type="primary"):
                            if not new_title.strip():
                                st.error("Task title can't be empty.")
                            else:
                                sel_task.edit(
                                    title=new_title.strip(),
                                    category=new_category,
                                    duration_minutes=int(new_duration),
                                    priority=new_priority,
                                    frequency=new_frequency,
                                    preferred_time=new_time if new_pin else None,
                                )
                                st.rerun()

# --- Today's schedule -----------------------------------------------------------------------

with tab_today:
    with st.container(border=True):
        st.markdown("#### 📅 Today's schedule")
        st.caption("Pinned tasks stay at their preferred time; everything else fills the earliest open gaps, "
                   "most important first.")
        if st.button("Generate schedule", type="primary", icon="🪄", width="stretch"):
            scheduler = Scheduler(owner)
            scheduler.generate_plan()
            ui.render_plan(scheduler)
            with st.expander("📖 Plan explanation"):
                st.text(scheduler.explain_plan())
    st.page_link(ui.AI_PAGE, label="Want help? Ask PawPal AI to adjust this schedule", icon="✨")
