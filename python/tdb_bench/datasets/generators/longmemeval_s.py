"""LongMemEval-S synthetic dataset generator.

Generates session chains in LongMemEval format:
- single-session-user     (user facts across turns)
- single-session-assistant (assistant facts across turns)
- multi-session           (same user across N sessions)
- temporal-reasoning     (time-ordered events)
- knowledge-update       (old fact → new fact)
- abstention              (unanswerable questions)

Each item: {id, context, question, ground_truth, task_type, metadata}
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from tdb_bench.models import BenchmarkItem


@dataclass
class UserProfile:
    name: str
    age: int
    occupation: str
    city: str
    preferences: list[str]
    facts: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionTurn:
    speaker: str  # "user" or "assistant"
    text: str
    timestamp: datetime
    extracted_fact: str | None = None


def _make_user_profile(seed: int) -> UserProfile:
    """Create a synthetic user profile."""
    random.seed(seed)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    cities = ["San Francisco", "New York", "Austin", "Seattle", "Boston", "Chicago"]
    occupations = ["engineer", "designer", "teacher", "doctor", "lawyer", "artist"]
    preferences = [
        "coffee", "tea", "reading", "hiking", "cooking", "gaming",
        "music", "movies", "travel", "sports", "art", "photography",
    ]
    profile = UserProfile(
        name=random.choice(names),
        age=random.randint(25, 55),
        occupation=random.choice(occupations),
        city=random.choice(cities),
        preferences=random.sample(preferences, k=random.randint(2, 4)),
    )
    profile.facts = {
        "name": profile.name,
        "age": str(profile.age),
        "occupation": profile.occupation,
        "city": profile.city,
        "preferences": ", ".join(profile.preferences),
    }
    return profile


def _build_single_session_user(profile: UserProfile, num_turns: int = 8) -> list[SessionTurn]:
    """Single-session user facts: user reveals personal info across conversation."""
    turns = []
    base_time = datetime(2024, 6, 1, 10, 0, 0)

    fact_order = list(profile.facts.keys())
    random.shuffle(fact_order)

    templates = {
        "name": "Hi, I'm {name}.",
        "age": "I'm {age} years old.",
        "occupation": "I work as a {occupation}.",
        "city": "I live in {city}.",
        "preferences": "I really like {preferences}.",
    }

    # Build conversation interleaving user + assistant turns
    user_idx = 0
    for i in range(num_turns):
        if user_idx < len(fact_order):
            key = fact_order[user_idx]
            turn = SessionTurn(
                speaker="user",
                text=templates[key].format(**profile.facts),
                timestamp=base_time + timedelta(minutes=5 * i),
                extracted_fact=f"{key}: {profile.facts[key]}",
            )
            turns.append(turn)
            user_idx += 1

        # Add assistant response
        responses = [
            "That's interesting!",
            "Got it.",
            "Thanks for sharing.",
            "Nice to know.",
            "I see.",
        ]
        turns.append(SessionTurn(
            speaker="assistant",
            text=random.choice(responses),
            timestamp=base_time + timedelta(minutes=5 * i + 2),
        ))

    return turns


def _build_multi_session(profile: UserProfile, num_sessions: int = 4) -> list[list[SessionTurn]]:
    """Multi-session: same user across N separate sessions (days apart)."""
    sessions = []
    base_time = datetime(2024, 6, 1, 10, 0, 0)

    for s in range(num_sessions):
        session_turns = []
        session_date = base_time + timedelta(days=s * 3)

        # Each session: user introduces/updates something
        session_intro = SessionTurn(
            speaker="user",
            text=f"Hey, it's {profile.name} again. I've been thinking about my {profile.facts['occupation']} job.",
            timestamp=session_date,
            extracted_fact=f"session_{s}: revisiting {profile.facts['occupation']}",
        )
        session_turns.append(session_intro)

        for t in range(3):
            session_turns.append(SessionTurn(
                speaker="assistant",
                text="How's that going?",
                timestamp=session_date + timedelta(minutes=2 + t * 5),
            ))
            session_turns.append(SessionTurn(
                speaker="user",
                text="It's been challenging but rewarding.",
                timestamp=session_date + timedelta(minutes=4 + t * 5),
            ))

        sessions.append(session_turns)

    return sessions


def _build_temporal_reasoning(profile: UserProfile) -> list[SessionTurn]:
    """Temporal reasoning: events with dates that require ordering/date math."""
    events = [
        ("hired", datetime(2022, 3, 15), "started new job"),
        ("moved", datetime(2023, 7, 1), "relocated to new city"),
        ("promoted", datetime(2024, 1, 10), "got promoted"),
        ("vacation", datetime(2024, 8, 20), "went on vacation"),
    ]
    turns = []
    base_time = datetime(2024, 5, 1, 9, 0, 0)

    for i, (event_type, event_date, description) in enumerate(events):
        turns.append(SessionTurn(
            speaker="user",
            text=f"On {event_date.strftime('%B %d, %Y')}, I {description}.",
            timestamp=base_time + timedelta(days=i * 10),
            extracted_fact=f"{event_type}: {event_date.date()} | {description}",
        ))
        turns.append(SessionTurn(
            speaker="assistant",
            text="Thanks for sharing.",
            timestamp=base_time + timedelta(days=i * 10, minutes=2),
        ))

    return turns


def _build_knowledge_update(profile: UserProfile) -> list[SessionTurn]:
    """Knowledge update: old fact → new fact (system should prefer updated)."""
    old_city = "Portland"
    new_city = profile.facts["city"]
    turns = []
    base_time = datetime(2024, 6, 1, 10, 0, 0)

    turns.append(SessionTurn(
        speaker="user",
        text=f"I used to live in {old_city}.",
        timestamp=base_time,
        extracted_fact=f"old_city: {old_city}",
    ))
    turns.append(SessionTurn(
        speaker="assistant",
        text="Oh, how was that?",
        timestamp=base_time + timedelta(minutes=2),
    ))
    turns.append(SessionTurn(
        speaker="user",
        text=f"It was nice but I moved to {new_city} last year.",
        timestamp=base_time + timedelta(minutes=5),
        extracted_fact=f"current_city: {new_city}",
    ))
    turns.append(SessionTurn(
        speaker="assistant",
        text="Great, welcome to your new city!",
        timestamp=base_time + timedelta(minutes=7),
    ))

    return turns


def _build_abstention(profile: UserProfile) -> list[SessionTurn]:
    """Abstention: question is unanswerable from provided context."""
    base_time = datetime(2024, 6, 1, 10, 0, 0)
    turns = [
        SessionTurn(
            speaker="user",
            text=f"I'm {profile.facts['age']} years old and work as a {profile.facts['occupation']}.",
            timestamp=base_time,
            extracted_fact=f"age: {profile.facts['age']}, job: {profile.facts['occupation']}",
        ),
        SessionTurn(
            speaker="assistant",
            text="Thanks for the update.",
            timestamp=base_time + timedelta(minutes=2),
        ),
    ]
    return turns


def _session_to_context(turns: list[SessionTurn]) -> str:
    """Render a list of turns into a conversation context string."""
    lines = []
    for turn in turns:
        speaker = "User" if turn.speaker == "user" else "Assistant"
        lines.append(f"[{turn.timestamp.isoformat()}] {speaker}: {turn.text}")
    return "\n".join(lines)


class LongMemEvalSDatasetGenerator:
    """Generates synthetic LongMemEval-S items.

    Synthesizes 5 task types:
    - single-session-user
    - single-session-assistant
    - multi-session
    - temporal-reasoning
    - knowledge-update

    Plus abstention items (unanswerable).
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate(self, num_items: int | None = None, tasks: list[str] | None = None) -> list[BenchmarkItem]:
        """Generate benchmark items.

        Args:
            num_items: Total number of items to generate (None = all tasks).
            tasks: List of task types to generate (None = all 5 types).

        Returns:
            List of BenchmarkItem in LongMemEval-S format.
        """
        all_tasks = [
            "single-session-user",
            "single-session-assistant",
            "multi-session",
            "temporal-reasoning",
            "knowledge-update",
            "abstention",
        ]
        target_tasks = tasks or all_tasks

        items: list[BenchmarkItem] = []
        for idx in range(100):  # Generate up to 100 profiles × tasks
            profile = _make_user_profile(self.seed + idx)

            for task_type in target_tasks:
                item = self._generate_item(profile, task_type, idx, task_type)
                if item:
                    items.append(item)

            if num_items and len(items) >= num_items:
                break

        if num_items:
            items = items[:num_items]

        return items

    def _generate_item(self, profile: UserProfile, task: str, idx: int, task_key: str) -> BenchmarkItem | None:
        """Generate a single benchmark item for a given task type."""
        item_id = f"longmemeval_s-{task}-{idx:04d}"
        metadata = {"task_type": task, "profile_seed": self.seed + idx}

        if task == "single-session-user":
            turns = _build_single_session_user(profile, num_turns=8)
            context = _session_to_context(turns)
            # Question asks for a fact that was shared in conversation
            facts = [t.extracted_fact for t in turns if t.extracted_fact]
            if not facts:
                return None
            fact_key = facts[0].split(":")[0].strip()
            templates = [
                f"What is the user's {fact_key}?",
                f"Tell me about the user's {fact_key}.",
                f"What do you know about the user's {fact_key}?",
            ]
            question = self.rng.choice(templates)
            ground_truth = profile.facts.get(fact_key, "unknown")

        elif task == "single-session-assistant":
            # Assistant-generated facts (the assistant remembers what user said)
            turns = _build_single_session_user(profile, num_turns=6)
            context = _session_to_context(turns)
            # Fact is extracted from user's turns but the question is about what was shared
            facts = [t.extracted_fact for t in turns if t.extracted_fact]
            if not facts:
                return None
            question = f"Based on the conversation, what did the user share about their {facts[0].split(':')[0].strip()}?"
            ground_truth = facts[0].split(":", 1)[1].strip()

        elif task == "multi-session":
            sessions = _build_multi_session(profile, num_sessions=4)
            # Flatten sessions into context with session markers
            session_contexts = []
            for si, session in enumerate(sessions):
                session_contexts.append(f"--- Session {si + 1} ---")
                session_contexts.append(_session_to_context(session))
            context = "\n".join(session_contexts)
            question = f"What was the user's occupation mentioned in the first session?"
            ground_truth = profile.facts["occupation"]

        elif task == "temporal-reasoning":
            turns = _build_temporal_reasoning(profile)
            context = _session_to_context(turns)
            events = [t.extracted_fact for t in turns if t.extracted_fact]
            # Question: how many days between first and third event?
            dates = []
            for e in events:
                # Parse: "hired: 2022-03-15 | started new job"
                parts = e.split("|")[0].strip()
                date_str = parts.split(":")[1].strip()
                dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
            days_between = abs((dates[2] - dates[0]).days)
            question = "How many days passed between the first and third event?"
            ground_truth = str(days_between)
            metadata["reasoning"] = "date_math"

        elif task == "knowledge-update":
            turns = _build_knowledge_update(profile)
            context = _session_to_context(turns)
            question = "Where does the user currently live?"
            ground_truth = profile.facts["city"]  # Should be the NEW city, not old one

        elif task == "abstention":
            turns = _build_abstention(profile)
            context = _session_to_context(turns)
            # Question about something NOT in the context
            question = "What is the user's favorite food?"
            ground_truth = "UNANSWERABLE"
            metadata["abstention"] = True

        else:
            return None

        return BenchmarkItem(
            item_id=item_id,
            dataset="longmemeval_s",
            context=context,
            question=question,
            ground_truth=ground_truth,
        )

    def to_jsonl(self, items: list[BenchmarkItem], path: str) -> None:
        """Write items to a JSONL file (LongMemEval-S dataset format)."""
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps({
                    "id": item.item_id,
                    "dataset": item.dataset,
                    "context": item.context,
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                }) + "\n")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate LongMemEval-S synthetic dataset")
    parser.add_argument("--output", type=str, default="longmemeval_s.jsonl", help="Output JSONL path")
    parser.add_argument("--num-items", type=int, default=None, help="Total items (default: all)")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated tasks (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    gen = LongMemEvalSDatasetGenerator(seed=args.seed)
    items = gen.generate(num_items=args.num_items, tasks=tasks)
    gen.to_jsonl(items, args.output)

    print(f"Generated {len(items)} items → {args.output}")
    task_counts: dict[str, int] = {}
    for item in items:
        task_counts[item.item_id.split("-")[1]] = task_counts.get(item.item_id.split("-")[1], 0) + 1
    for task, count in task_counts.items():
        print(f"  {task}: {count}")

    sys.exit(0)
