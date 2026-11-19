"""Tests for guideline retrieval (the RAG step). These use the real CSV knowledge base."""

from pawpal_ai import CATEGORIES
from pawpal_rag import detect_pets, format_for_prompt, load_guidelines, retrieve
from pawpal_system import Owner, Pet, Task


def ids(guidelines):
    return [g.id for g in guidelines]


def test_knowledge_base_rows_are_valid_for_the_app():
    rows = load_guidelines()
    assert len(rows) >= 30
    assert len({g.id for g in rows}) == len(rows)  # unique ids
    for g in rows:
        assert g.category in CATEGORIES
        assert g.priority in ("high", "medium", "low")
        assert g.frequency in ("daily", "weekly")
        assert 1 <= g.duration_min <= 240
        assert g.life_stage in ("young", "adult", "senior", "any")
        assert ":" not in g.timing  # no clock times, so explanations can't copy invented times


def test_detects_species_and_age_from_request_text():
    assert detect_pets("I have a 3 year old dog named Rex") == {"dog": {"adult"}}
    assert detect_pets("I just got a 4 month old kitten") == {"cat": {"young"}}
    assert detect_pets("My senior dog Max is slowing down") == {"dog": {"senior"}}
    assert detect_pets("What's the capital of France?") == {}


def test_existing_pets_count_even_if_request_does_not_mention_them():
    owner = Owner(name="Jordan", available_minutes=60)
    owner.add_pet(Pet(name="Whiskers", species="cat", age=12))

    assert detect_pets("fix my schedule", owner) == {"cat": {"senior"}}


def test_rex_request_retrieves_walk_breakfast_and_medication_with_food():
    found = ids(retrieve("I have a 3 year old dog named Rex. He needs a 30 minute walk, "
                         "breakfast at 7:30, and his allergy pill with breakfast"))

    assert {"G01", "G04", "G32"} <= set(found)
    assert found[0] == "G32"  # most keyword hits: pill + with breakfast
    assert not any(g.startswith(("G14", "G23")) for g in found)  # no cat or rabbit rows


def test_kitten_gets_kitten_guidelines_not_adult_or_dog_ones():
    found = retrieve("I just got a 4 month old kitten named Mochi. Set up her feeding and litter.")
    found_ids = ids(found)

    assert "G16" in found_ids and "G18" in found_ids  # kitten meals, litter box
    assert "G14" not in found_ids  # adult cat breakfast
    assert all(g.species in ("cat", "any") for g in found)


def test_general_request_still_gets_each_pets_core_daily_care():
    # Found by the evaluation: "set up her daily care" used to miss the litter box guideline.
    found_ids = ids(retrieve("I just got a 4 month old kitten named Mochi. Please set up her daily care."))

    assert {"G16", "G18", "G20"} <= set(found_ids)  # kitten meals, litter box, play
    assert "G22" not in found_ids  # optional low-priority brushing isn't pushed on the AI


def test_unrelated_request_retrieves_nothing():
    assert retrieve("What's the capital of France?") == []


def test_existing_task_titles_help_retrieval_in_adjust_mode():
    owner = Owner(name="Jordan", available_minutes=60)
    buddy = Pet(name="Buddy", species="dog", age=4)
    owner.add_pet(buddy)
    buddy.add_task(Task("Ear drops", "medication", 5, priority="high"))

    assert "G34" in ids(retrieve("can you fix any clashes?", owner))


def test_retrieval_respects_k_and_prompt_block_lists_ids():
    found = retrieve("my dog and cat need feeding, walks, play, brushing, and their meds", k=5)

    assert len(found) == 5
    block = format_for_prompt(found)
    assert block.startswith("Care guidelines retrieved")
    assert all(f"[{g.id}]" in block for g in found)
    assert format_for_prompt([]) == ""
