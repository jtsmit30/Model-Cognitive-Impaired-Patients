#!/usr/bin/env python3
"""
=============================================================================
TESTING ONLY — High Thinking Mode (Single Conversation)
=============================================================================
This test version uses thinking_level=HIGH and increased token budgets to
evaluate whether deeper reasoning produces more nuanced, adaptive
conversations — particularly whether the conversation partner (clinician,
family member, etc.) responds to the patient's cues in a clinically
realistic, adaptive way rather than following a rigid script.

Compare the output of this test against generate_conversations_testing_only.py
(which uses thinking_level=LOW) to determine if the quality improvement
justifies the ~2-3x cost increase.

Changes from the LOW-thinking test:
  - thinking_level: low → HIGH (all 3 API call types)
  - max_output_tokens: 16384 → 32768 (transcript generation)
  - max_output_tokens: 12288 → 16384 (annotation chunks)
  - max_output_tokens: 4096 → 8192  (session summary)
  - temperature: kept at 1.0 (Google's recommendation for Gemini 3)

Usage:
  1. Set your API key:  export GEMINI_API_KEY="your-key-here"
  2. Run:               python generate_conversations_testing_high_thinking.py
  3. Output will be in: ./test_conversations_high_thinking/

Cost estimate: ~$0.85-1.20 per conversation (vs ~$0.45 with LOW thinking)

Requirements:
  pip install google-genai tqdm
=============================================================================
"""

import os
import re
import json
import time
import random
import argparse
import logging
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "gemini-3.1-pro-preview" # this is the most recent version of Gemini and released after Febuary 19th 2026

# All 38 markers from the taxonomy
MARKERS = {
    # Affective
    "AFF-01": "Persistent Negative Sentiment",
    "AFF-02": "Sentiment Trajectory Decline",
    "AFF-03": "Anhedonia Language",
    "AFF-04": "Hopelessness Expressions",
    "AFF-05": "Irritability / Agitation Language",
    "AFF-06": "Emotional Flatness / Absence",
    "AFF-07": "Anxiety / Worry Language",
    "AFF-08": "Guilt / Self-Blame Language",
    # Cognitive
    "COG-01": "Reduced Lexical Diversity",
    "COG-02": "Word-Finding Difficulty",
    "COG-03": "Syntactic Simplification",
    "COG-04": "Content Repetition",
    "COG-05": "Tangential Speech",
    "COG-06": "Temporal Confusion",
    "COG-07": "Pronoun Ambiguity Increase",
    "COG-08": "Reduced Information Density",
    # Behavioral
    "BEH-01": "Increased Response Latency",
    "BEH-02": "Response Brevity / Truncation",
    "BEH-03": "Absent Topic Initiation",
    "BEH-04": "Topic Avoidance / Deflection",
    "BEH-05": "Conversation Termination Attempts",
    "BEH-06": "Denial / Minimization Pattern",
    "BEH-07": "Help-Rejecting Behavior",
    # Self-Reference
    "SLF-01": "Elevated First-Person Singular Pronoun Ratio",
    "SLF-02": "Reduced First-Person Plural Pronouns",
    "SLF-03": "Negative Self-Evaluation",
    "SLF-04": "Perceived Burden Language",
    "SLF-05": "Identity Confusion / Loss",
    # Somatic
    "SOM-01": "Sleep Disruption Reports",
    "SOM-02": "Appetite / Weight Change Reports",
    "SOM-03": "Fatigue / Energy Reports",
    "SOM-04": "Pain / Discomfort Reports",
    "SOM-05": "Concentration / Focus Complaints",
    "SOM-06": "Activity / Social Withdrawal Reports",
    # Paralinguistic-Text Concordance
    "PLC-01": "Positive Content / Negative Tone",
    "PLC-02": "Defensive Escalation",
    "PLC-03": "Verbal-Prosodic Dissociation",
    "PLC-04": "Effortful Speech Pattern",
}

# Marker groups for balanced generation
MARKER_GROUPS = {
    "affective": [k for k in MARKERS if k.startswith("AFF")],
    "cognitive": [k for k in MARKERS if k.startswith("COG")],
    "behavioral": [k for k in MARKERS if k.startswith("BEH")],
    "self_reference": [k for k in MARKERS if k.startswith("SLF")],
    "somatic": [k for k in MARKERS if k.startswith("SOM")],
    "concordance": [k for k in MARKERS if k.startswith("PLC")],
}

# Conversation scenario variations
SETTINGS = ["home", "clinic", "phone call", "day program", "care facility"]
PARTNERS = [
    ("Spouse", "FAMILY_MEMBER"),
    ("Adult child", "FAMILY_MEMBER"),
    ("Friend", "FRIEND"),
    ("Home care nurse", "CLINICIAN"),
    ("Social worker", "CLINICIAN"),
    ("Occupational therapist", "CLINICIAN"),
    ("Neighbor", "ACQUAINTANCE"),
]
PATIENT_AGES = list(range(55, 90))
PATIENT_GENDERS = ["male", "female"]
SEVERITIES = ["mild", "moderate", "severe"]

# Clinical profiles for realistic marker combinations
CLINICAL_PROFILES = [
    {
        "name": "Major Depressive Episode",
        "description": "Patient presenting with depressive symptoms",
        "core_markers": ["AFF-01", "AFF-03", "AFF-06", "BEH-01", "BEH-02",
                         "BEH-03", "SOM-01", "SOM-03", "SLF-01"],
        "optional_markers": ["AFF-04", "AFF-08", "SLF-03", "SLF-04",
                             "SOM-02", "SOM-06", "PLC-01", "BEH-06"],
        "weight": 0.20,
    },
    {
        "name": "Anxiety with Depression",
        "description": "Mixed anxiety and depressive presentation",
        "core_markers": ["AFF-01", "AFF-07", "BEH-01", "SOM-01", "SOM-05"],
        "optional_markers": ["AFF-02", "AFF-05", "BEH-04", "SLF-01",
                             "SOM-03", "SOM-04", "PLC-03"],
        "weight": 0.15,
    },
    {
        "name": "Early Cognitive Decline",
        "description": "Patient showing early signs of cognitive impairment",
        "core_markers": ["COG-01", "COG-02", "COG-03", "COG-04", "COG-08",
                         "BEH-01"],
        "optional_markers": ["COG-05", "COG-06", "COG-07", "AFF-06",
                             "SOM-05", "BEH-04", "SLF-05"],
        "weight": 0.15,
    },
    {
        "name": "Social Withdrawal / Isolation",
        "description": "Patient withdrawing from social engagement",
        "core_markers": ["BEH-02", "BEH-03", "BEH-05", "SOM-06", "SLF-02",
                         "AFF-06"],
        "optional_markers": ["AFF-03", "BEH-07", "SLF-01", "PLC-01",
                             "BEH-06"],
        "weight": 0.10,
    },
    {
        "name": "Caregiver Conflict / Irritability",
        "description": "Patient showing irritability and resistance to care",
        "core_markers": ["AFF-05", "BEH-04", "BEH-05", "BEH-06", "BEH-07",
                         "PLC-02"],
        "optional_markers": ["AFF-01", "SLF-03", "SOM-03", "SOM-04"],
        "weight": 0.10,
    },
    {
        "name": "Masked Depression",
        "description": "Patient hiding distress behind positive facade",
        "core_markers": ["PLC-01", "PLC-03", "BEH-06", "AFF-06"],
        "optional_markers": ["AFF-01", "BEH-04", "SOM-01", "SOM-03",
                             "SLF-01", "BEH-02"],
        "weight": 0.10,
    },
    {
        "name": "Healthy Baseline",
        "description": "Patient with no significant markers (control data)",
        "core_markers": [],
        "optional_markers": [],
        "weight": 0.20,
    },
]

# Normalize weights
_total_weight = sum(p["weight"] for p in CLINICAL_PROFILES)
for p in CLINICAL_PROFILES:
    p["weight"] /= _total_weight


# Rare marker boost profile
RARE_MARKER_BOOST_PROFILE = {
    "name": "Rare Marker Boost",
    "description": "Supplemental profile targeting rare but clinically critical markers",
    "core_markers": ["SLF-04", "BEH-05", "COG-06", "SLF-05"],
    "optional_markers": ["AFF-04", "AFF-08", "BEH-04", "BEH-07",
                         "SLF-03", "COG-05", "AFF-06", "SOM-06"],
    "weight": 0.0,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConversationConfig:
    """Configuration for a single conversation generation."""
    conversation_id: str
    profile_name: str
    profile_description: str
    active_markers: list
    severity: str
    patient_age: int
    patient_gender: str
    setting: str
    partner_role: str
    partner_label: str
    num_exchanges: int


@dataclass
class GeneratedConversation:
    """Result of a generation run."""
    config: dict
    transcript: Optional[str] = None
    annotation: Optional[str] = None
    generation_time: float = 0.0
    annotation_time: float = 0.0
    error: Optional[str] = None
    truncation_warning: Optional[str] = None
    annotation_warnings: list = field(default_factory=list)
    quality_score: float = 0.0  # 0.0-1.0 overall quality rating


# ---------------------------------------------------------------------------
# JSON Repair Utilities
# ---------------------------------------------------------------------------

from JsonUtils import repair_json


# ---------------------------------------------------------------------------
# Transcript Validation
# ---------------------------------------------------------------------------

def parse_transcript(transcript: str) -> dict:
    """Parse a generated transcript and extract structured information.

    Handles the template format from the audio pipeline specification:
      [HH:MM:SS.mmm] SPEAKER (Role): "Utterance text in quotes"
      [tone:flat energy:low pace:slow pause_before:4.2 jitter:1.8 pitch_mean:142 pitch_range:18]

    Also handles the older format as fallback:
      [HH:MM:SS] SPEAKER: [tone: X | energy: X | ...] text without quotes

    Returns a dict with:
      - patient_turns: list of dicts with text, paralinguistic info, utterance_id
      - partner_turns: list of partner utterance texts
      - exchange_count: number of complete exchanges
      - has_metadata: whether SESSION_METADATA block was found
      - issues: list of any problems found
    """
    result = {
        "patient_turns": [],
        "partner_turns": [],
        "exchange_count": 0,
        "has_metadata": False,
        "issues": [],
    }

    if not transcript or not transcript.strip():
        result["issues"].append("CRITICAL: Transcript is empty")
        return result

    if "SESSION_METADATA:" in transcript:
        result["has_metadata"] = True

    lines = transcript.split('\n')

    # --- Detect partner labels used in this transcript ---
    partner_labels = set()
    for line in lines:
        # Match: [timestamp] LABEL (Role): or [timestamp] LABEL:
        label_match = re.match(
            r'(?:\[[\d:.]+\]\s*)?([A-Z_]+(?:\s+[A-Z_]+)*)'
            r'(?:\s*\([^)]*\))?\s*:\s',
            line.strip()
        )
        if label_match:
            label = label_match.group(1).strip()
            if label != "PATIENT" and label not in (
                "SESSION_METADATA", "TRANSCRIPT", "tone", "energy", "pace"
            ):
                partner_labels.add(label)

    # --- Regex for the new template paralinguistic format ---
    # Matches: [tone:flat energy:low pace:slow pause_before:4.2 jitter:1.8 pitch_mean:142 pitch_range:18]
    # All fields are optional except tone; space-separated, no pipes
    PARA_NEW_RE = re.compile(
        r'\[\s*'
        r'(?:tone:(\S+))?\s*'
        r'(?:energy:(\S+))?\s*'
        r'(?:pace:(\S+))?\s*'
        r'(?:pause_before:([\d.]+))?\s*'
        r'(?:jitter:([\d.]+))?\s*'
        r'(?:pitch_mean:([\d.]+))?\s*'
        r'(?:pitch_range:([\d.]+))?\s*'
        r'\]'
    )

    # --- Regex for the old pipe-separated format ---
    # Matches: [tone: flat | energy: low | pace: slow | pause_before: 4.2s]
    PARA_OLD_RE = re.compile(
        r'\[\s*'
        r'(?:tone:\s*(\w+))?\s*\|?\s*'
        r'(?:energy:\s*(\w+))?\s*\|?\s*'
        r'(?:pace:\s*(\w+))?\s*\|?\s*'
        r'(?:pause_before:\s*([\d.]+)s?)?\s*'
        r'\]'
    )

    def _parse_paralinguistic_line(line_text: str) -> Optional[dict]:
        """Try to parse a paralinguistic annotation from a line.
        Returns dict with all available fields, or None if no match."""
        line_text = line_text.strip()

        # Try new format first (space-separated, 7 fields)
        m = PARA_NEW_RE.search(line_text)
        if m and m.group(1):  # At least tone must be present
            para = {
                "tone": m.group(1) or "neutral",
                "energy": m.group(2) or "moderate",
                "pace": m.group(3) or "normal",
                "pause_before": float(m.group(4)) if m.group(4) else 1.0,
            }
            # Add extended fields if present
            if m.group(5):
                para["jitter"] = float(m.group(5))
            if m.group(6):
                para["pitch_mean"] = float(m.group(6))
            if m.group(7):
                para["pitch_range"] = float(m.group(7))
            return para

        # Try old format (pipe-separated, 4 fields)
        m = PARA_OLD_RE.search(line_text)
        if m and m.group(1):
            return {
                "tone": m.group(1) or "neutral",
                "energy": m.group(2) or "moderate",
                "pace": m.group(3) or "normal",
                "pause_before": float(m.group(4)) if m.group(4) else 1.0,
            }

        return None

    # --- Build the speaker turn pattern ---
    # Matches lines like:
    #   [14:02:15.300] PATIENT (Patient): "Some text here"
    #   [09:00:15] CLINICIAN: Some text here
    all_labels = {"PATIENT"} | partner_labels
    label_alt = '|'.join(re.escape(l) for l in sorted(all_labels, key=len, reverse=True))
    SPEAKER_RE = re.compile(
        r'(?:\[([\d:.]+)\]\s*)?'           # Optional timestamp
        r'(' + label_alt + r')'            # Speaker label
        r'(?:\s*\([^)]*\))?'              # Optional (Role) in parens
        r'\s*:\s*'                         # Colon separator
        r'(.*)'                            # Rest of line (text or annotation)
    )

    # --- Main parsing loop ---
    patient_count = 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        speaker_match = SPEAKER_RE.match(line)
        if not speaker_match:
            i += 1
            continue

        timestamp = speaker_match.group(1)
        speaker = speaker_match.group(2)
        first_content = speaker_match.group(3).strip()

        if speaker == "PATIENT":
            patient_count += 1
            utterance_id = f"U-{patient_count:03d}"

            # --- Extract text and paralinguistic ---
            # Strategy: collect text and look for paralinguistic on same line,
            # next line, or preceding the text (old format).
            text_parts = []
            paralinguistic = None

            # Check if first_content starts with a paralinguistic annotation
            # (old format: annotation before text)
            para_on_first = _parse_paralinguistic_line(first_content)
            if para_on_first and first_content.startswith('['):
                paralinguistic = para_on_first
                # Text might follow after the closing bracket on the same line
                bracket_end = first_content.find(']')
                if bracket_end >= 0:
                    after_bracket = first_content[bracket_end + 1:].strip()
                    if after_bracket:
                        text_parts.append(after_bracket)
            else:
                # New format: text is first (possibly quoted)
                # Strip surrounding quotes if present
                text_content = first_content.strip('"').strip()
                if text_content:
                    text_parts.append(text_content)

            # Scan subsequent lines for continuation text and paralinguistic
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue

                # Is this a new speaker turn?
                if SPEAKER_RE.match(next_line):
                    break

                # Is this a paralinguistic annotation line?
                para_check = _parse_paralinguistic_line(next_line)
                if para_check and next_line.startswith('['):
                    if paralinguistic is None:
                        paralinguistic = para_check
                    i += 1
                    continue

                # It's continuation text
                # Strip quotes if present
                text_content = next_line.strip('"').strip()
                if text_content:
                    text_parts.append(text_content)
                i += 1

            utterance_text = ' '.join(text_parts).strip()

            # Validate paralinguistic completeness
            if paralinguistic is None:
                paralinguistic = {
                    "tone": "neutral", "energy": "moderate",
                    "pace": "normal", "pause_before": 1.0,
                }
                result["issues"].append(
                    f"{utterance_id}: Missing paralinguistic annotation"
                )
            else:
                # Check for missing extended fields
                missing_fields = []
                for field in ("jitter", "pitch_mean", "pitch_range"):
                    if field not in paralinguistic:
                        missing_fields.append(field)
                if missing_fields:
                    result["issues"].append(
                        f"{utterance_id}: Missing extended paralinguistic "
                        f"field(s): {', '.join(missing_fields)}"
                    )

            result["patient_turns"].append({
                "utterance_id": utterance_id,
                "text": utterance_text,
                "paralinguistic": paralinguistic,
            })
            continue

        elif speaker in partner_labels:
            # Partner turn — extract text (strip quotes)
            text_content = first_content.strip('"').strip()
            text_parts = [text_content] if text_content else []

            # Check for continuation lines
            i += 1
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    continue
                if SPEAKER_RE.match(next_line):
                    break
                # Skip paralinguistic annotation lines for partners
                if next_line.startswith('[') and _parse_paralinguistic_line(next_line):
                    i += 1
                    continue
                text_parts.append(next_line.strip('"').strip())
                i += 1

            result["partner_turns"].append(' '.join(text_parts).strip())
            continue

        i += 1

    # Compute exchange count
    result["exchange_count"] = min(
        len(result["patient_turns"]),
        len(result["partner_turns"])
    )

    # Quality checks
    if len(result["patient_turns"]) == 0:
        result["issues"].append(
            "CRITICAL: No PATIENT turns found. The model may have used "
            "a different speaker label format."
        )
    elif len(result["patient_turns"]) < 3:
        result["issues"].append(
            f"WARNING: Only {len(result['patient_turns'])} PATIENT turns found "
            f"(expected 10+)"
        )

    if not result["has_metadata"]:
        result["issues"].append(
            "MINOR: SESSION_METADATA block not found in transcript"
        )

    return result


# ---------------------------------------------------------------------------
# Annotation Validation
# ---------------------------------------------------------------------------

def validate_annotation(annotation_json: dict, config: ConversationConfig,
                        expected_utterance_count: int = 0) -> list:
    """Validate annotation structure and marker completeness.

    Returns a list of warning strings. Empty list means the annotation is clean.
    """
    warnings = []
    utterance_annotations = annotation_json.get("utterance_annotations", [])

    if not utterance_annotations:
        warnings.append("CRITICAL: utterance_annotations list is empty")
        return warnings

    # Check utterance count alignment
    actual_count = len(utterance_annotations)
    if expected_utterance_count > 0:
        if actual_count < expected_utterance_count * 0.7:
            warnings.append(
                f"SIGNIFICANT: Only {actual_count}/{expected_utterance_count} "
                f"utterances annotated (>30% missing)"
            )
        elif actual_count < expected_utterance_count:
            warnings.append(
                f"MINOR: {actual_count}/{expected_utterance_count} "
                f"utterances annotated"
            )

    for utt in utterance_annotations:
        utt_id = utt.get("utterance_id", "UNKNOWN")
        markers_data = utt.get("markers", {})

        if not isinstance(markers_data, dict):
            warnings.append(f"{utt_id}: 'markers' field is not a dict")
            continue

        present_markers = set(markers_data.keys())
        missing = set(MARKERS.keys()) - present_markers

        if missing:
            warnings.append(
                f"{utt_id}: missing {len(missing)} marker(s): "
                f"{', '.join(sorted(missing))}"
            )

        # Check that present markers have the required fields
        for m_id, m_data in markers_data.items():
            if not isinstance(m_data, dict):
                warnings.append(f"{utt_id}/{m_id}: marker value is not a dict")
                continue
            for required_field in ("present", "severity", "confidence"):
                if required_field not in m_data:
                    warnings.append(
                        f"{utt_id}/{m_id}: missing field '{required_field}'"
                    )

    # Check session_summary
    if "session_summary" not in annotation_json:
        warnings.append("MINOR: session_summary block missing")

    return warnings


# ---------------------------------------------------------------------------
# Atomic File I/O
# ---------------------------------------------------------------------------

def atomic_write(filepath: Path, content: str, encoding: str = "utf-8"):
    """Write file atomically: write to temp file, then rename.

    Prevents partial files if the process crashes mid-write.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.name}.",
        suffix=".tmp"
    )
    try:
        with os.fdopen(fd, 'w', encoding=encoding) as f:
            f.write(content)
        shutil.move(tmp_path, filepath)
    except Exception:
        # Clean up temp file on error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_json(filepath: Path, data: dict):
    """Write JSON file atomically with pretty formatting."""
    content = json.dumps(data, indent=2, ensure_ascii=False)
    atomic_write(filepath, content)


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

def save_conversation(result: GeneratedConversation, output_dir: Path,
                      model_id: str):
    """Save a single conversation and its annotation to disk.

    Uses atomic writes to prevent partial files on crash.
    Only saves transcript if it contains actual content.
    """
    conv_id = result.config["conversation_id"]

    # Save transcript (only if non-empty)
    if result.transcript and result.transcript.strip():
        transcript_dir = output_dir / "transcripts"
        transcript_path = transcript_dir / f"{conv_id}_transcript.txt"
        atomic_write(transcript_path, result.transcript)
    else:
        logging.warning(f"{conv_id}: Skipping empty transcript save")

    # Save annotation
    if result.annotation and result.annotation.strip():
        annotation_dir = output_dir / "annotations"
        annotation_path = annotation_dir / f"{conv_id}_annotation.json"

        try:
            parsed = json.loads(result.annotation)
            atomic_write_json(annotation_path, parsed)
        except json.JSONDecodeError:
            # Save raw text as .txt if JSON is invalid
            raw_path = annotation_dir / f"{conv_id}_annotation_raw.txt"
            atomic_write(raw_path, result.annotation)
            logging.warning(
                f"{conv_id}: Saved raw annotation text "
                f"(JSON invalid) to {raw_path.name}"
            )

    # Compute marker_utterance_counts from the annotation
    marker_utterance_counts = {m: 0 for m in MARKERS}
    utterance_count = 0
    if result.annotation:
        try:
            parsed = json.loads(result.annotation)
            utterance_annotations = parsed.get("utterance_annotations", [])
            utterance_count = len(utterance_annotations)
            for utt in utterance_annotations:
                for m_id, m_data in utt.get("markers", {}).items():
                    if isinstance(m_data, dict) and m_data.get("present", False):
                        if m_id in marker_utterance_counts:
                            marker_utterance_counts[m_id] += 1
        except (json.JSONDecodeError, AttributeError):
            pass

    # Save metadata
    meta_dir = output_dir / "metadata"
    meta_path = meta_dir / f"{conv_id}_meta.json"
    meta = {
        "schema_version": "2.0",
        "generator_model": model_id,
        "annotation_model": model_id,
        "config": result.config,
        "generation_time": result.generation_time,
        "annotation_time": result.annotation_time,
        "utterance_count": utterance_count,
        "marker_utterance_counts": marker_utterance_counts,
        "quality_score": result.quality_score,
        "truncation_warning": result.truncation_warning,
        "annotation_warnings": result.annotation_warnings,
        "error": result.error,
        "has_valid_transcript": bool(
            result.transcript and result.transcript.strip()
        ),
        "has_valid_annotation": bool(
            result.annotation and _is_valid_annotation(result.annotation)
        ),
    }
    atomic_write_json(meta_path, meta)


def _is_valid_annotation(annotation_text: str) -> bool:
    """Check if annotation text is valid JSON with utterance_annotations."""
    try:
        parsed = json.loads(annotation_text)
        return bool(parsed.get("utterance_annotations"))
    except (json.JSONDecodeError, AttributeError):
        return False


def save_generation_report(
    results: list,
    output_dir: Path,
    total_time: float
):
    """Save a summary report of the generation run."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    # Count marker coverage
    marker_counts = {m_id: 0 for m_id in MARKERS}
    profile_counts = {}
    severity_counts = {"mild": 0, "moderate": 0, "severe": 0, "none": 0}

    for r in successful:
        profile = r.config["profile_name"]
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        severity_counts[r.config["severity"]] += 1
        for m in r.config["active_markers"]:
            marker_counts[m] += 1

    # Quality distribution
    quality_scores = [r.quality_score for r in successful]
    avg_quality = (
        sum(quality_scores) / len(quality_scores) if quality_scores else 0
    )
    high_quality = sum(1 for q in quality_scores if q >= 0.7)
    low_quality = sum(1 for q in quality_scores if q < 0.4)

    # Find under-represented markers
    min_target = 30
    underrepresented = {
        m_id: count for m_id, count in marker_counts.items()
        if count < min_target and m_id in [
            m for group in MARKER_GROUPS.values() for m in group
        ]
    }

    report = {
        "summary": {
            "total_requested": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_time_seconds": round(total_time, 1),
            "avg_generation_time": round(
                sum(r.generation_time for r in successful)
                / max(len(successful), 1), 2
            ),
            "avg_annotation_time": round(
                sum(r.annotation_time for r in successful)
                / max(len(successful), 1), 2
            ),
            "avg_quality_score": round(avg_quality, 3),
            "high_quality_count": high_quality,
            "low_quality_count": low_quality,
        },
        "profile_distribution": profile_counts,
        "severity_distribution": severity_counts,
        "marker_coverage": marker_counts,
        "underrepresented_markers": underrepresented,
        "failed_conversations": [
            {"id": r.config["conversation_id"], "error": r.error}
            for r in failed
        ],
    }

    report_path = output_dir / "generation_report.json"
    atomic_write_json(report_path, report)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Successful:      {len(successful)} / {len(results)}")
    print(f"  Failed:          {len(failed)}")
    print(f"  Total time:      {total_time / 60:.1f} minutes")
    print(f"  Avg quality:     {avg_quality:.3f}")
    print(f"  High quality:    {high_quality} (score >= 0.7)")
    print(f"  Low quality:     {low_quality} (score < 0.4)")
    print(f"\n  Profile distribution:")
    for profile, count in sorted(profile_counts.items(), key=lambda x: -x[1]):
        print(f"    {profile}: {count}")
    print(f"\n  Severity distribution:")
    for sev, count in severity_counts.items():
        print(f"    {sev}: {count}")
    if underrepresented:
        print(f"\n  WARNING: {len(underrepresented)} markers below target "
              f"({min_target} conversations):")
        for m_id, count in sorted(underrepresented.items(), key=lambda x: x[1]):
            print(f"    {m_id} ({MARKERS[m_id]}): {count}")
        print("  Consider running with --boost-rare to top up.")
    print(f"\n  Output saved to: {output_dir}")
    print("=" * 60)
