"""Music pipeline utilities for GhostWriter.

This package contains helper modules that support symbolic music workflows,
including import normalization, sanitization, and future score generation
stages referenced in MUSIC_REQUIREMENTS.md.
"""

from .importer import process_import_directory, generate_import_musiccsv
from .sanitizer import sanitize_import, ensure_sanitized
from .context import (
    VoiceSpec,
    ScoreSummary,
    VoiceContext,
    parse_voice_token,
    build_voice_context,
    build_music_prompt_context,
    write_voice_token,
)
from .pipeline import (
    run_metadata_tracks_step,
    run_melody_edges_step,
    ensure_first_score_gate,
    run_subtle_score_pass,
    run_melody_construction_step,
    run_multi_voice_first_pass,
    assemble_first_pass_variant,
)
from .exporter import finalize_music_exports

__all__ = [
    "process_import_directory",
    "generate_import_musiccsv",
    "sanitize_import",
    "ensure_sanitized",
    "VoiceSpec",
    "ScoreSummary",
    "VoiceContext",
    "parse_voice_token",
    "build_voice_context",
    "build_music_prompt_context",
    "write_voice_token",
    "run_metadata_tracks_step",
    "run_melody_edges_step",
    "run_melody_construction_step",
    "ensure_first_score_gate",
    "run_subtle_score_pass",
    "run_multi_voice_first_pass",
    "assemble_first_pass_variant",
    "finalize_music_exports",
]
