"""Project configuration constants.

Centralizes required prompt template paths and related config used by the
driver and pipelines. Keeping these in one place avoids duplication and makes
it easier to evolve defaults.
"""
from __future__ import annotations

REQUIRED_PROMPTS = [
    # Core polishing and summaries
    "prompts/polish_prose_prompt.md",
    "prompts/story_so_far_prompt.md",
    "prompts/story_relative_to_prompt.md",
    # Deterministic pipeline steps
    "prompts/narration_brain_storm_prompt.md",
    "prompts/dialog_brain_storm_prompt.md",
    "prompts/ordering_prompt.md",
    "prompts/implicit_brain_storm_prompt.md",
    "prompts/generate_narration_prompt.md",
    "prompts/actor_assignment_prompt.md",
    "prompts/body_language_prompt.md",
    "prompts/agenda_prompt.md",
    "prompts/reaction_prompt.md",
    "prompts/subtle_edit_prompt.md",
]
"""Configuration constants and paths for GhostWriter."""

REQUIRED_PROMPTS = [
    # Core polishing and summaries
    "prompts/polish_prose_prompt.md",
    "prompts/story_so_far_prompt.md",
    "prompts/story_relative_to_prompt.md",
    # Deterministic pipeline steps
    "prompts/narration_brain_storm_prompt.md",
    "prompts/dialog_brain_storm_prompt.md",
    "prompts/ordering_prompt.md",
    "prompts/implicit_brain_storm_prompt.md",
    "prompts/generate_narration_prompt.md",
    "prompts/actor_assignment_prompt.md",
    "prompts/body_language_prompt.md",
    "prompts/agenda_prompt.md",
    "prompts/reaction_prompt.md",
    "prompts/subtle_edit_prompt.md",
]
