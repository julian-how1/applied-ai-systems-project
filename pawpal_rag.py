"""Retrieval for PawPal AI: look up relevant pet-care guidelines before the model answers (RAG).

The knowledge base is a plain CSV (data/pet_care_guidelines.csv) that anyone can edit. For each
request, `retrieve` works out which species and life stages are involved (from the owner's existing
pets and from the request text) and which care topics are mentioned (walk, pill, litter...), then
scores every guideline row and returns the most relevant ones. The agent pastes them into the prompt
so the model's durations, priorities, and timing come from the knowledge base instead of guesses.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pawpal_system import Owner

GUIDELINES_PATH = Path(__file__).with_name("data") / "pet_care_guidelines.csv"
DEFAULT_TOP_K = 8

# Words in a request that tell us which species it's about.
SPECIES_WORDS = {
    "dog": ("dog", "dogs", "puppy", "puppies", "pup", "pups", "doggo"),
    "cat": ("cat", "cats", "kitten", "kittens", "kitty"),
    "rabbit": ("rabbit", "rabbits", "bunny", "bunnies"),
    "bird": ("bird", "birds", "parrot", "parakeet", "budgie", "cockatiel"),
    "fish": ("fish", "goldfish", "betta", "aquarium"),
}
YOUNG_WORDS = ("puppy", "puppies", "pup", "pups", "kitten", "kittens", "baby")
SENIOR_WORDS = ("senior", "elderly", "aging", "older")
# Age (in years) at which each species counts as senior.
SENIOR_AGE = {"dog": 8, "cat": 11, "rabbit": 6, "bird": 15, "fish": 99}


@dataclass(frozen=True)
class Guideline:
    id: str
    species: str  # dog / cat / ... or "any"
    life_stage: str  # young / adult / senior / any
    task: str
    category: str
    duration_min: int
    frequency: str
    priority: str
    timing: str
    keywords: tuple[str, ...]
    notes: str

    def as_prompt_line(self) -> str:
        who = "any pet" if self.species == "any" else self.species
        stage = "" if self.life_stage == "any" else f" ({self.life_stage})"
        return (f"[{self.id}] {who}{stage} - {self.task}: category {self.category}, {self.duration_min} min, "
                f"{self.frequency}, {self.priority} priority. Timing: {self.timing}. Note: {self.notes}")


@lru_cache(maxsize=4)
def load_guidelines(path: Path = GUIDELINES_PATH) -> tuple[Guideline, ...]:
    """Read the CSV knowledge base (cached)."""
    with open(path, newline="", encoding="utf-8") as f:
        return tuple(
            Guideline(
                id=row["id"].strip(),
                species=row["species"].strip().lower(),
                life_stage=row["life_stage"].strip().lower(),
                task=row["task"].strip(),
                category=row["category"].strip().lower(),
                duration_min=int(row["duration_min"]),
                frequency=row["frequency"].strip().lower(),
                priority=row["priority"].strip().lower(),
                timing=row["timing"].strip(),
                keywords=tuple(k.strip().lower() for k in row["keywords"].split(",") if k.strip()),
                notes=row["notes"].strip(),
            )
            for row in csv.DictReader(f)
        )


def _has_word(text: str, phrase: str) -> bool:
    """True if `phrase` appears in `text` as whole words (plural 's'/'es' allowed)."""
    return re.search(rf"\b{re.escape(phrase)}(?:s|es)?\b", text) is not None


def _stage_for_age(species: str, years: float) -> str:
    if years < 1:
        return "young"
    return "senior" if years >= SENIOR_AGE.get(species, 99) else "adult"


def detect_pets(request: str, owner: Optional[Owner] = None) -> dict[str, set[str]]:
    """Return {species: {life stages}} for the owner's existing pets plus any mentioned in the request."""
    found: dict[str, set[str]] = {}
    if owner is not None:
        for pet in owner.pets:
            species = pet.species.lower()
            if species in SPECIES_WORDS:
                found.setdefault(species, set()).add(_stage_for_age(species, pet.age))

    text = request.lower()
    mentioned = [s for s, words in SPECIES_WORDS.items() if any(_has_word(text, w) for w in words)]
    for species in mentioned:
        stages = found.setdefault(species, set())
        words = SPECIES_WORDS[species]
        if any(_has_word(text, w) for w in YOUNG_WORDS if w in words or w == "baby"):
            stages.add("young")
        if any(_has_word(text, w) for w in SENIOR_WORDS):
            stages.add("senior")

    # Ages in the request ("3 year old", "4 month old", "(2)") apply to the species mentioned with them.
    if len(mentioned) == 1:
        species = mentioned[0]
        for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*(years?|yrs?|months?|mos?|weeks?|wks?)\b", text):
            years = float(number) if unit.startswith("y") else float(number) / (12 if unit.startswith("mo") else 52)
            found[species].add(_stage_for_age(species, years))

    for species, stages in found.items():
        if not stages:
            stages.add("adult")
    return found


def retrieve(
    request: str,
    owner: Optional[Owner] = None,
    k: int = DEFAULT_TOP_K,
    guidelines: Optional[tuple[Guideline, ...]] = None,
) -> list[Guideline]:
    """Return up to `k` guideline rows most relevant to this request and household, best first."""
    guidelines = guidelines if guidelines is not None else load_guidelines()
    pets = detect_pets(request, owner)
    text = request.lower()
    task_text = " ".join(t.title.lower() for p in (owner.pets if owner else []) for t in p.tasks)

    scored = []
    for g in guidelines:
        if g.species == "any":
            stages = set().union(*pets.values()) if pets else {"adult"}
        elif g.species in pets:
            stages = pets[g.species]
        else:
            continue  # a species this household doesn't have
        if g.life_stage != "any" and g.life_stage not in stages:
            continue

        hits = sum(1 for kw in g.keywords if _has_word(text, kw))
        existing = sum(1 for kw in g.keywords if _has_word(task_text, kw))
        score = 3 * hits + existing
        if g.species != "any" and g.life_stage != "any":
            score += 1  # prefer rows written for this exact pet and age
        if score == 0 and not (g.species != "any" and g.priority in ("high", "medium")):
            continue  # drop unrelated rows, but keep each pet's core daily care (meals, walks, litter...)
        if score == 0:
            score = 0.5
        scored.append((score, g))

    scored.sort(key=lambda pair: (-pair[0], ("high", "medium", "low").index(pair[1].priority), pair[1].id))
    return [g for _, g in scored[:k]]


def format_for_prompt(guidelines: list[Guideline]) -> str:
    """Render retrieved guidelines as the block the agent adds to its prompt."""
    if not guidelines:
        return ""
    lines = "\n".join(g.as_prompt_line() for g in guidelines)
    return ("Care guidelines retrieved from PawPal's knowledge base. Use them for duration, priority, "
            "frequency, and timing unless the owner gave different details, and list the ids you relied "
            f"on in \"notes\" (for example \"Used G01, G32\"):\n{lines}")
