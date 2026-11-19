"""Tests for the PawPal+ AI agent. A scripted fake model stands in for Gemini, so these run offline."""

import json
from datetime import date, time

import pytest

from pawpal_ai import (
    ActionError,
    PawPalAgent,
    ProposalFormatError,
    apply_action,
    apply_proposal,
    check_proposal,
    parse_proposal,
    ungrounded_times,
)
from pawpal_system import Owner, Pet, Scheduler, Task

DAY = date(2026, 10, 7)


class ScriptedLLM:
    """Fake LLMClient: returns queued replies in order and records every prompt it was sent."""

    def __init__(self, *replies, explanation="Your plan is ready."):
        self.replies = list(replies)
        self.explanation = explanation
        self.prompts = []

    def complete(self, system, prompt, json_mode=False):
        self.prompts.append(prompt)
        if not json_mode:  # the explain step
            if isinstance(self.explanation, Exception):
                raise self.explanation
            return self.explanation
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def proposal(*actions, understanding="ok"):
    return {"understanding": understanding, "actions": list(actions), "notes": ""}


def owner_with_buddy(**task_kwargs):
    owner = Owner(name="Jordan", available_minutes=120, day_start=time(8, 0))
    buddy = Pet(name="Buddy", species="dog", age=4)
    owner.add_pet(buddy)
    if task_kwargs:
        buddy.add_task(Task(**task_kwargs))
    return owner, buddy


# --- Action validation ------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, message",
    [
        ({"op": "fly_to_moon"}, "unknown op"),
        ({"op": "add_task", "pet": "Max", "title": "Walk", "category": "exercise", "duration_minutes": 30}, "no pet named"),
        ({"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise", "duration_minutes": 30,
          "priority": "urgent"}, "priority"),
        ({"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "jogging", "duration_minutes": 30}, "category"),
        ({"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise", "duration_minutes": 0}, "between"),
        ({"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise", "duration_minutes": 30,
          "preferred_time": "7pm"}, "HH:MM"),
        ({"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise"}, "missing"),
        ({"op": "add_pet", "name": "buddy", "species": "dog", "age": 2}, "already exists"),
        ({"op": "set_availability", "minutes": -5}, "between"),
        ({"op": "remove_task", "pet": "Buddy", "title": "Nonexistent"}, "no open task"),
        ({"op": "edit_task", "pet": "Buddy", "title": "Breakfast", "changes": {}}, "non-empty"),
    ],
)
def test_invalid_actions_are_rejected_without_changing_owner(action, message):
    owner, buddy = owner_with_buddy(title="Breakfast", category="feeding", duration_minutes=10)
    before = (owner.available_minutes, len(owner.pets), [(t.title, t.priority) for t in buddy.tasks])

    with pytest.raises(ActionError, match=message):
        apply_action(owner, action)

    assert (owner.available_minutes, len(owner.pets), [(t.title, t.priority) for t in buddy.tasks]) == before


def test_valid_actions_update_owner_and_names_match_case_insensitively():
    owner, buddy = owner_with_buddy(title="Evening walk", category="exercise", duration_minutes=30)

    apply_action(owner, {"op": "edit_task", "pet": "buddy", "title": "evening WALK",
                         "changes": {"preferred_time": "19:00", "priority": "High"}})
    apply_action(owner, {"op": "set_day_start", "time": "06:30"})

    assert buddy.tasks[0].preferred_time == time(19, 0)
    assert buddy.tasks[0].priority == "high"
    assert owner.day_start == time(6, 30)


def test_completed_tasks_cannot_be_edited_by_the_ai():
    owner, buddy = owner_with_buddy(title="Walk", category="exercise", duration_minutes=30)
    buddy.complete_task(buddy.tasks[0], on=DAY)  # spawns tomorrow's open "Walk"

    apply_action(owner, {"op": "edit_task", "pet": "Buddy", "title": "Walk", "changes": {"duration_minutes": 45}})

    done, upcoming = buddy.tasks
    assert done.duration_minutes == 30 and upcoming.duration_minutes == 45


# --- Constraint checking --------------------------------------------------------------------


def test_check_proposal_never_mutates_the_real_owner():
    owner, buddy = owner_with_buddy()
    actions = [
        {"op": "set_availability", "minutes": 60},
        {"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise", "duration_minutes": 30},
    ]

    sandbox, scheduler, report = check_proposal(owner, actions, DAY)

    assert report.ok
    assert owner.available_minutes == 120 and buddy.tasks == []
    assert sandbox.available_minutes == 60 and [e.task.title for e in scheduler.plan] == ["Walk"]


def test_new_overlap_is_an_error_but_preexisting_overlap_is_only_a_warning():
    owner, buddy = owner_with_buddy(title="Dinner", category="feeding", duration_minutes=10,
                                    preferred_time=time(17, 30))
    buddy.add_task(Task("Ear drops", "medication", 5, priority="high", preferred_time=time(17, 30)))

    _, _, unchanged = check_proposal(owner, [], DAY)
    _, _, clashing = check_proposal(owner, [{"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise",
                                             "duration_minutes": 30, "preferred_time": "17:20"}], DAY)

    assert unchanged.ok and any("Existing conflict" in w for w in unchanged.warnings)
    assert not clashing.ok and any("Creates a time conflict" in e and "Walk" in e for e in clashing.errors)


def test_cutting_time_so_high_priority_care_is_dropped_is_an_error():
    owner, _ = owner_with_buddy(title="Meds", category="medication", duration_minutes=20, priority="high")

    _, _, report = check_proposal(owner, [{"op": "set_availability", "minutes": 10}], DAY)

    assert not report.ok
    assert "high-priority" in report.errors[0]


def test_apply_proposal_rechecks_and_applies_only_when_valid():
    owner, buddy = owner_with_buddy()
    good = [{"op": "add_task", "pet": "Buddy", "title": "Walk", "category": "exercise", "duration_minutes": 30}]
    bad = [{"op": "add_task", "pet": "Ghost", "title": "Walk", "category": "exercise", "duration_minutes": 30}]

    assert not apply_proposal(owner, bad, DAY).ok
    assert buddy.tasks == []
    assert apply_proposal(owner, good, DAY).ok
    assert [t.title for t in buddy.tasks] == ["Walk"]


# --- Parsing the model's reply -----------------------------------------------------------------


def test_parse_proposal_accepts_code_fenced_json():
    raw = '```json\n{"understanding": "x", "actions": [{"op": "set_availability", "minutes": 30}]}\n```'
    assert parse_proposal(raw).actions == [{"op": "set_availability", "minutes": 30}]


@pytest.mark.parametrize("raw", ["Sure! Here's your plan.", "[]", '{"actions": "none"}', '{"actions": [{"pet": "x"}]}'])
def test_parse_proposal_rejects_malformed_replies(raw):
    with pytest.raises(ProposalFormatError):
        parse_proposal(raw)


# --- The agent loop -----------------------------------------------------------------------------


def test_agent_creates_schedule_from_scratch_using_the_real_scheduler():
    owner = Owner(name="Jordan", available_minutes=120)
    llm = ScriptedLLM(proposal(
        {"op": "set_availability", "minutes": 90},
        {"op": "set_day_start", "time": "07:00"},
        {"op": "add_pet", "name": "Rex", "species": "dog", "age": 3},
        {"op": "add_task", "pet": "Rex", "title": "Breakfast", "category": "feeding", "duration_minutes": 10,
         "priority": "high", "preferred_time": "07:30"},
        {"op": "add_task", "pet": "Rex", "title": "Walk", "category": "exercise", "duration_minutes": 30},
    ))

    result = PawPalAgent(llm, day=DAY).run(owner, "New dog Rex, 90 min from 7am, breakfast 7:30 and a walk")

    assert result.mode == "create"
    assert result.accepted and result.rounds == 1 and result.first_try_valid
    plan = [(e.task.title, e.start.time()) for e in result.scheduler.plan]
    assert plan == [("Walk", time(7, 0)), ("Breakfast", time(7, 30))]
    assert owner.pets == []  # nothing applied until the user approves
    assert [s.stage for s in result.steps] == ["interpret", "retrieve", "propose", "check", "explain"]
    assert "Care guidelines retrieved" in llm.prompts[0]  # RAG context reached the model
    assert any(g.id == "G01" for g in result.guidelines)  # adult dog walk guideline


def test_agent_can_run_without_guidelines_for_comparison():
    owner = Owner(name="Jordan", available_minutes=120)
    llm = ScriptedLLM(proposal())

    result = PawPalAgent(llm, day=DAY, use_guidelines=False).run(owner, "My dog needs a walk")

    assert result.guidelines == []
    assert "Care guidelines" not in llm.prompts[0]
    assert "retrieve" not in [s.stage for s in result.steps]


def test_agent_revises_after_validator_feedback():
    owner, _ = owner_with_buddy(title="Evening walk", category="exercise", duration_minutes=30,
                                preferred_time=time(18, 0))
    bad = proposal({"op": "edit_task", "pet": "Max", "title": "Evening walk", "changes": {"preferred_time": "19:00"}})
    good = proposal({"op": "edit_task", "pet": "Buddy", "title": "Evening walk", "changes": {"preferred_time": "19:00"}})
    llm = ScriptedLLM(bad, good)

    result = PawPalAgent(llm, day=DAY).run(owner, "Move the evening walk to 7pm")

    assert result.mode == "adjust"
    assert result.accepted and result.rounds == 2 and not result.first_try_valid
    assert "no pet named 'Max'" in llm.prompts[1]  # the error was fed back to the model
    assert "Current schedule produced by the scheduler" in llm.prompts[0]


def test_agent_recovers_from_unparseable_reply():
    owner, _ = owner_with_buddy()
    llm = ScriptedLLM("I think you should walk Buddy!", proposal())

    result = PawPalAgent(llm, day=DAY).run(owner, "anything?")

    assert result.accepted and result.rounds == 2
    assert "not valid JSON" in llm.prompts[1]


def test_agent_gives_up_after_max_rounds_and_reports_failure():
    owner, _ = owner_with_buddy()
    bad = proposal({"op": "remove_task", "pet": "Buddy", "title": "Ghost task"})
    llm = ScriptedLLM(bad, bad, bad)

    result = PawPalAgent(llm, max_rounds=3, day=DAY).run(owner, "remove the ghost task")

    assert not result.accepted and result.rounds == 3
    assert result.report.errors


def test_agent_handles_ai_service_errors_without_crashing():
    owner, _ = owner_with_buddy()
    result = PawPalAgent(ScriptedLLM(RuntimeError("429 RESOURCE_EXHAUSTED")), day=DAY).run(owner, "plan my day")

    assert not result.accepted
    assert "AI service failed" in result.error


def test_explanation_falls_back_to_scheduler_when_it_invents_times():
    owner, _ = owner_with_buddy(title="Walk", category="exercise", duration_minutes=30, preferred_time=time(9, 0))
    llm = ScriptedLLM(proposal(), explanation="Walk Buddy at 9:00, then brush him at 11:45.")

    result = PawPalAgent(llm, day=DAY).run(owner, "explain my day")

    assert result.explanation_issues == ["11:45"]
    assert result.explanation.startswith("Daily plan for Jordan")


def test_explanation_falls_back_when_explain_call_fails():
    owner, _ = owner_with_buddy(title="Walk", category="exercise", duration_minutes=30)
    llm = ScriptedLLM(proposal(), explanation=RuntimeError("timeout"))

    result = PawPalAgent(llm, day=DAY).run(owner, "explain")

    assert result.accepted and result.explanation.startswith("Daily plan for Jordan")


def test_gemini_client_falls_back_and_skips_busy_model_on_next_call():
    pytest.importorskip("google.genai")
    from types import SimpleNamespace

    from pawpal_ai import GeminiClient

    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        if model == "busy-model":
            raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
        return SimpleNamespace(text=f"answer from {model}")

    client = object.__new__(GeminiClient)  # skip __init__ so no API key or network is needed
    client._client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    client.model, client.models = "busy-model", ["busy-model", "backup-model"]
    client.retries_per_model, client.last_model, client._busy_until = 0, None, {}

    assert client.complete("sys", "first") == "answer from backup-model"
    assert client.complete("sys", "second") == "answer from backup-model"
    assert calls == ["busy-model", "backup-model", "backup-model"]  # busy model skipped the 2nd time
    assert client.last_model == "backup-model"


def test_ungrounded_times_understands_12_and_24_hour_formats():
    owner, _ = owner_with_buddy(title="Dinner", category="feeding", duration_minutes=10, preferred_time=time(17, 30))
    scheduler = Scheduler(owner, day=DAY)
    scheduler.generate_plan()

    assert ungrounded_times("Dinner at 17:30, or 5:30 PM, ends 5:40", scheduler) == []
    assert ungrounded_times("Dinner at 5:30 AM", scheduler) == ["5:30 AM"]
