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

DEFAULT_MODEL_ID = "gemini-3.1-pro-preview"

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

def repair_json(raw_text: str) -> Optional[dict]:
    """Attempt to parse and repair malformed JSON from LLM output.

    Handles common Gemini failure modes:
      1. Markdown code fences wrapping the JSON
      2. Trailing commas before closing braces/brackets
      3. Unterminated strings (truncated output)
      4. Single quotes instead of double quotes
      5. NaN/Infinity values (invalid in JSON)
      6. Control characters inside strings
      7. Partial/truncated JSON (attempts to close open structures)

    Returns parsed dict on success, None on failure.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Step 1: Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    text = text.strip()

    # Step 2: Try parsing as-is first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 3: Apply progressive repairs
    repaired = text

    # 3a: Remove control characters inside strings (except \n, \r, \t)
    repaired = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', repaired)

    # 3b: Fix trailing commas: ,} or ,]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # 3c: Replace NaN/Infinity with null
    repaired = re.sub(r'\bNaN\b', 'null', repaired)
    repaired = re.sub(r'\bInfinity\b', 'null', repaired)
    repaired = re.sub(r'-Infinity\b', 'null', repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 4: Handle truncated JSON by closing open structures
    # Count unmatched braces and brackets
    repaired = _close_truncated_json(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 5: Try to fix unterminated strings
    # Find the position of the error and try to close the string
    repaired = _fix_unterminated_strings(repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Step 6: Last resort — extract the largest valid JSON object
    result = _extract_partial_json(text)
    if result is not None:
        return result

    return None


def _close_truncated_json(text: str) -> str:
    """Close unclosed braces/brackets in truncated JSON."""
    # Track nesting, accounting for strings
    in_string = False
    escape_next = False
    open_stack = []

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_stack.append('}')
        elif ch == '[':
            open_stack.append(']')
        elif ch in ('}', ']') and open_stack and open_stack[-1] == ch:
            open_stack.pop()

    # If we're inside an unterminated string, close it
    if in_string:
        text += '"'

    # Remove any trailing comma before closing
    text = re.sub(r',\s*$', '', text)

    # Close remaining open structures
    while open_stack:
        text += open_stack.pop()

    return text


def _fix_unterminated_strings(text: str) -> str:
    """Try to fix unterminated string literals by finding the break point."""
    # Find the last properly terminated key-value pair
    # and truncate everything after it, then close structures
    lines = text.split('\n')
    for i in range(len(lines) - 1, -1, -1):
        candidate = '\n'.join(lines[:i + 1])
        # Remove trailing comma
        candidate = re.sub(r',\s*$', '', candidate.rstrip())
        candidate = _close_truncated_json(candidate)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    return text


def _extract_partial_json(text: str) -> Optional[dict]:
    """Extract the largest parseable JSON object from text.

    Useful when the LLM wraps JSON in explanation text or the JSON
    is truncated but the first N objects are valid.
    """
    # Find all potential JSON start positions
    starts = [i for i, ch in enumerate(text) if ch == '{']

    best = None
    best_len = 0

    for start in starts[:5]:  # Only try first 5 candidates
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if len(candidate) > best_len:
                            best = parsed
                            best_len = len(candidate)
                    except json.JSONDecodeError:
                        pass
                    break

    return best


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
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are an expert clinical conversation simulator specializing in 
generating realistic patient dialogues for mental health and cognitive decline research. 
Your conversations must be naturalistic, clinically plausible, and contain specific 
behavioral markers as instructed. You never break character or add meta-commentary."""


def build_generation_prompt(config: ConversationConfig) -> str:
    """Build the prompt for generating a single conversation."""

    if not config.active_markers:
        marker_section = """The patient should appear HEALTHY with no significant clinical markers.
They should be engaged, responsive, show normal conversational dynamics, varied vocabulary,
appropriate emotional range, and natural topic initiation. This is control data — the
conversation should feel warm, normal, and unremarkable."""
    else:
        marker_lines = []
        for m_id in config.active_markers:
            marker_lines.append(f"  - {m_id}: {MARKERS[m_id]}")
        marker_list = "\n".join(marker_lines)
        marker_section = f"""The patient should exhibit the following markers at {config.severity.upper()} severity:

{marker_list}

Important guidelines for marker expression:
- Markers should emerge naturally through dialogue, not be stated explicitly
- Some markers should be subtle and require careful analysis to detect
- Include 2-3 moments where markers are clearly present and 2-3 where the
  patient appears more normal (realistic variation within a single conversation)
- If PLC markers are active, ensure the paralinguistic annotations CONTRADICT
  the text content at specific points"""

    prompt = f"""Generate a realistic {config.num_exchanges * 2}-turn conversation between a patient
and their conversation partner.

SCENARIO:
- Patient: Age {config.patient_age}, {config.patient_gender}
- Conversation partner: {config.partner_role} (labeled as {config.partner_label})
- Setting: {config.setting}

CLINICAL PROFILE: {config.profile_name}
{config.profile_description}

MARKERS TO EXPRESS:
{marker_section}

FORMAT REQUIREMENTS:
Output the conversation in EXACTLY this format. Do not add any other text before or after.
This format simulates output from an audio transcription pipeline (Whisper + pyannote +
openSMILE + emotion2vec). Follow it precisely.

SESSION_METADATA:
conversation_id: {config.conversation_id}
patient_age: {config.patient_age}
patient_gender: {config.patient_gender}
setting: {config.setting}
partner_role: {config.partner_role}
partner_label: {config.partner_label}
profile: {config.profile_name}
severity: {config.severity}

TRANSCRIPT:

[HH:MM:SS.mmm] {config.partner_label} ({config.partner_role}): "Their opening line"
[tone:warm energy:medium pace:normal]

[HH:MM:SS.mmm] PATIENT (Patient): "Patient's response"
[tone:flat energy:low pace:slow pause_before:4.2 jitter:1.8 pitch_mean:142 pitch_range:18]

Continue alternating speakers. Rules for the transcript:

UTTERANCE FORMAT (follow exactly):
1. Line 1: [HH:MM:SS.mmm] SPEAKER_LABEL (Role): "Utterance text in double quotes"
2. Line 2: [paralinguistic annotations in square brackets]
3. Then a blank line before the next utterance
4. Both speakers get paralinguistic annotations
5. The {config.partner_label} annotations include: tone, energy, pace (3 fields)
6. PATIENT annotations include ALL 7 fields: tone, energy, pace, pause_before, jitter, pitch_mean, pitch_range

PARALINGUISTIC FIELD REFERENCE (space-separated, no pipes, no colons with spaces):
- tone: flat, sad, anxious, irritable, warm, neutral, cheerful, hesitant, defensive
- energy: very_low, low, medium, high
- pace: very_slow, slow, normal, fast
- pause_before: float seconds (0.5-12.0; higher for BEH-01 markers)
- jitter: float percent (typical 0.5-3.0; higher values indicate vocal instability)
- pitch_mean: float Hz (typical male 85-180, female 165-255)
- pitch_range: float Hz (typical 20-80; very low <15 suggests flat affect AFF-06)

TIMESTAMP AND PACING RULES:
1. Use millisecond-precision timestamps: [HH:MM:SS.mmm] (e.g., [14:02:15.300])
2. Start the conversation at a realistic time between 08:00 and 20:00
3. CRITICAL — timestamps must reflect realistic speech durations:
   - A short utterance (2-5 words) takes 1-2 seconds to speak
   - A medium utterance (10-20 words) takes 3-6 seconds to speak
   - A long utterance (30+ words) takes 8-15 seconds to speak
   - After a speaker finishes, the next speaker begins within 0.5-3 seconds
     (unless the patient has elevated pause_before for BEH-01)
   - The pause_before value in the PATIENT annotation IS the gap between the
     partner finishing and the patient starting — the timestamp should reflect this
   - Do NOT add unexplained 30-60 second gaps between turns
4. A {config.num_exchanges}-exchange conversation at realistic pacing should span
   approximately 4-10 minutes total, NOT {config.num_exchanges} minutes

CONTENT RULES:
1. Generate exactly {config.num_exchanges} exchanges (one exchange = partner speaks + patient responds)
2. Include natural conversational elements: interruptions, overlaps noted as [overlap],
   trailing off noted as "...", false starts, filler words (um, uh) where appropriate
3. Wrap all spoken text in double quotes
4. If PLC markers are active, the paralinguistic values must CONTRADICT the text content"""

    return prompt


def build_chunk_annotation_prompt(transcript: str, config: ConversationConfig,
                                  patient_turns_chunk: list,
                                  chunk_index: int, total_chunks: int) -> str:
    """Build an annotation prompt for a CHUNK of patient utterances.

    Instead of asking the model to annotate all utterances in one call (which
    produces ~1,580 tokens per utterance × 10-20 utterances = 15,800-31,600
    tokens, far exceeding the 8,192 max_output_tokens limit), we split the
    utterances into chunks of ANNOTATION_CHUNK_SIZE and make one API call per
    chunk. Each chunk produces a partial annotation JSON that is merged after
    all chunks complete.
    """

    # Build the full marker reference
    marker_ref_lines = []
    for m_id, m_name in MARKERS.items():
        marker_ref_lines.append(f'    "{m_id}": "{m_name}"')
    marker_ref = ",\n".join(marker_ref_lines)

    # List the specific utterances this chunk must annotate
    utt_lines = []
    for turn in patient_turns_chunk:
        text_preview = turn.get("text", "")
        if len(text_preview) > 100:
            text_preview = text_preview[:100] + "..."
        utt_lines.append(f'  - {turn["utterance_id"]}: "{text_preview}"')
    utterance_listing = chr(10).join(utt_lines)

    prompt = f"""You are an expert clinical conversation analyst. Analyze the following
conversation transcript and produce utterance-level marker annotations.

This is chunk {chunk_index + 1} of {total_chunks}. You must annotate ONLY these
{len(patient_turns_chunk)} patient utterances:
{utterance_listing}

For each utterance above, identify which of the 38 markers are present, their severity,
your confidence, and the specific evidence. Use the FULL conversation transcript below
for context, but only annotate the utterances listed above.

MARKER REFERENCE:
{{
{marker_ref}
}}

SEVERITY LEVELS: absent, mild, moderate, severe
CONFIDENCE: float from 0.0 to 1.0

CRITICAL OUTPUT RULES:
1. Respond with ONLY a valid JSON object — no markdown, no backticks, no explanation text.
2. Do NOT include any text before or after the JSON.
3. Make sure all strings are properly terminated with closing quotes.
4. Make sure all brackets and braces are properly closed.
5. Use double quotes for all keys and string values (not single quotes).
6. For EACH utterance, include ALL 38 markers. Set "present": false for markers not detected.

The JSON must have this exact structure:

{{
  "utterance_annotations": [
    {{
      "utterance_id": "U-001",
      "speaker": "PATIENT",
      "text": "[exact text of the utterance]",
      "paralinguistic": {{
        "tone": "flat",
        "energy": "low",
        "pace": "slow",
        "pause_before": 4.2,
        "jitter": 1.8,
        "pitch_mean": 142,
        "pitch_range": 18
      }},
      "markers": {{
        "AFF-01": {{"present": true, "severity": "moderate", "confidence": 0.82, "evidence": "brief explanation"}},
        "AFF-02": {{"present": false, "severity": "absent", "confidence": 0.90, "evidence": ""}},
        ... (include ALL 38 markers for each utterance)
      }}
    }}
  ]
}}

TRANSCRIPT (for context — annotate ONLY the {len(patient_turns_chunk)} utterances listed above):
{transcript}"""

    return prompt


def build_session_summary_prompt(transcript: str, config: ConversationConfig,
                                 all_annotations: list) -> str:
    """Build a prompt to generate the session_summary from completed annotations.

    Called once after all chunks are annotated to produce the session-level
    summary (active markers, proxy scores, risk flags).
    """

    # Summarize which markers were detected across all utterances
    detected_markers = {}
    for utt in all_annotations:
        for m_id, m_data in utt.get("markers", {}).items():
            if isinstance(m_data, dict) and m_data.get("present", False):
                severity = m_data.get("severity", "mild")
                if m_id not in detected_markers:
                    detected_markers[m_id] = {"count": 0, "max_severity": severity}
                detected_markers[m_id]["count"] += 1
                # Track max severity
                sev_order = {"mild": 1, "moderate": 2, "severe": 3}
                if sev_order.get(severity, 0) > sev_order.get(
                    detected_markers[m_id]["max_severity"], 0
                ):
                    detected_markers[m_id]["max_severity"] = severity

    marker_summary_lines = []
    for m_id, info in sorted(detected_markers.items()):
        marker_summary_lines.append(
            f"  {m_id} ({MARKERS.get(m_id, '?')}): "
            f"detected in {info['count']} utterances, "
            f"max severity={info['max_severity']}"
        )
    marker_summary = "\n".join(marker_summary_lines) if marker_summary_lines else "  No markers detected"

    prompt = f"""You are an expert clinical conversation analyst. Based on the marker
detections below, produce a session-level summary for conversation {config.conversation_id}.

DETECTED MARKERS ACROSS ALL UTTERANCES:
{marker_summary}

Total patient utterances analyzed: {len(all_annotations)}
Clinical profile: {config.profile_name} ({config.severity})

Respond with ONLY a valid JSON object containing the session summary:

{{
  "session_summary": {{
    "active_markers": ["list of marker IDs detected at least once"],
    "dominant_severity": "the most common severity level",
    "proxy_scores": {{
      "phq9_estimate": 0,
      "gad7_estimate": 0,
      "who5_estimate": 0,
      "moca_proxy": 0
    }},
    "overall_wellbeing": "healthy / mild_concern / concerning / critical",
    "risk_flags": ["list any urgent clinical items"],
    "confidence": 0.0
  }}
}}

Use the full transcript for additional context:
{transcript[:3000]}"""

    return prompt


# Maximum utterances per annotation API call. At ~1,580 tokens per utterance
# (38 markers × ~40 tokens + overhead), 4 utterances ≈ 6,320 tokens — safely
# under the 8,192 output token limit with room for JSON structure overhead.
ANNOTATION_CHUNK_SIZE = 4


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------

def select_profile() -> dict:
    """Weighted random selection of a clinical profile."""
    r = random.random()
    cumulative = 0.0
    for profile in CLINICAL_PROFILES:
        cumulative += profile["weight"]
        if r <= cumulative:
            return profile
    return CLINICAL_PROFILES[-1]


def select_markers(profile: dict, severity: str) -> list:
    """Select active markers for a conversation based on clinical profile."""
    if not profile["core_markers"]:
        if random.random() < 0.3:
            all_markers = list(MARKERS.keys())
            return random.sample(all_markers, k=random.randint(1, 2))
        return []

    # Core marker dropout for mild severity
    if severity == "mild" and len(profile["core_markers"]) > 3:
        drop_count = random.randint(1, 2)
        active = random.sample(
            profile["core_markers"],
            k=len(profile["core_markers"]) - drop_count
        )
    else:
        active = list(profile["core_markers"])

    # Severity affects how many optional markers appear
    if severity == "mild":
        num_optional = random.randint(0, min(2, len(profile["optional_markers"])))
    elif severity == "moderate":
        num_optional = random.randint(1, min(4, len(profile["optional_markers"])))
    else:  # severe
        num_optional = random.randint(3, min(6, len(profile["optional_markers"])))

    if profile["optional_markers"]:
        optional = random.sample(
            profile["optional_markers"],
            k=min(num_optional, len(profile["optional_markers"]))
        )
        active.extend(optional)

    return list(set(active))  # deduplicate


def create_conversation_config(conv_id: int) -> ConversationConfig:
    """Create a randomized conversation configuration."""
    profile = select_profile()
    severity = random.choice(SEVERITIES)
    partner = random.choice(PARTNERS)

    return ConversationConfig(
        conversation_id=f"CONV-{conv_id:04d}",
        profile_name=profile["name"],
        profile_description=profile["description"],
        active_markers=select_markers(profile, severity),
        severity=severity if profile["core_markers"] else "none",
        patient_age=random.choice(PATIENT_AGES),
        patient_gender=random.choice(PATIENT_GENDERS),
        setting=random.choice(SETTINGS),
        partner_role=partner[0],
        partner_label=partner[1],
        num_exchanges=random.randint(10, 20),
    )


def generate_conversation(
    client: genai.Client,
    config: ConversationConfig,
    model_id: str,
    max_retries: int = 3,
    skip_annotation: bool = False,
) -> GeneratedConversation:
    """Generate a single conversation and its annotation.

    Key improvements:
      - Validates transcript structure before attempting annotation
      - Uses JSON repair on annotation output
      - Passes parsed patient turns to annotation prompt for alignment
      - Computes quality score based on completeness
    """

    result = GeneratedConversation(config=asdict(config))

    # ----- Step 1: Generate the conversation transcript -----
    generation_prompt = build_generation_prompt(config)

    transcript_ok = False
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.models.generate_content(
                model=model_id,
                contents=generation_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=1.0,
                    max_output_tokens=32768,
                    top_p=0.95,
                    top_k=40,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high",
                    ),
                ),
            )
            result.transcript = response.text
            result.generation_time = time.time() - t0

            # Validate transcript structure
            parsed_tx = parse_transcript(result.transcript)
            patient_turn_count = len(parsed_tx["patient_turns"])

            if patient_turn_count == 0:
                logging.warning(
                    f"{config.conversation_id} attempt {attempt + 1}: "
                    f"No PATIENT turns found in transcript. "
                    f"Transcript preview: {result.transcript[:200]!r}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                else:
                    result.error = (
                        "Transcript contained no parseable PATIENT turns "
                        f"after {max_retries} attempts"
                    )
                    result.truncation_warning = "0 PATIENT turns found"
                    return result

            # Check for truncation
            if patient_turn_count < config.num_exchanges * 0.8:
                result.truncation_warning = (
                    f"Only {patient_turn_count}/{config.num_exchanges} "
                    f"exchanges found"
                )
                logging.warning(
                    f"{config.conversation_id}: Possible truncation — "
                    f"{result.truncation_warning}"
                )

            # Log any issues found
            for issue in parsed_tx["issues"]:
                logging.info(f"  {config.conversation_id}: {issue}")

            transcript_ok = True
            break

        except Exception as e:
            logging.warning(
                f"Generation attempt {attempt + 1} failed for "
                f"{config.conversation_id}: {e}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                result.error = (
                    f"Generation failed after {max_retries} attempts: {e}"
                )
                return result

    if not transcript_ok:
        return result

    # ----- Step 2: Skip annotation if requested -----
    if skip_annotation:
        result.quality_score = _compute_quality_score(
            result, parsed_tx, has_annotation=False
        )
        return result

    # ----- Step 3: Annotate the conversation in chunks -----
    # 38 markers × ~40 tokens each = ~1,580 tokens per utterance.
    # A 15-exchange conversation would need ~23,700 output tokens in one call,
    # far exceeding the 8,192 limit. We split into chunks of ANNOTATION_CHUNK_SIZE
    # utterances and merge the results.

    patient_turns = parsed_tx["patient_turns"]
    chunks = [
        patient_turns[i:i + ANNOTATION_CHUNK_SIZE]
        for i in range(0, len(patient_turns), ANNOTATION_CHUNK_SIZE)
    ]
    total_chunks = len(chunks)
    logging.info(
        f"{config.conversation_id}: Annotating {len(patient_turns)} utterances "
        f"in {total_chunks} chunk(s) of up to {ANNOTATION_CHUNK_SIZE}"
    )

    all_utterance_annotations = []
    annotation_ok = False
    t0_annotation = time.time()

    for chunk_idx, chunk in enumerate(chunks):
        chunk_prompt = build_chunk_annotation_prompt(
            result.transcript, config,
            patient_turns_chunk=chunk,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
        )

        chunk_parsed = None
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                response = client.models.generate_content(
                    model=model_id,
                    contents=chunk_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a clinical conversation annotation system. "
                            "Respond ONLY with valid JSON. No markdown formatting, "
                            "no backticks, no explanation text. Ensure all strings "
                            "are properly closed and all brackets are matched."
                        ),
                        temperature=0.5,
                        max_output_tokens=16384,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high",
                        ),
                    ),
                )
                raw_chunk = response.text

                # Try standard JSON parse first
                try:
                    chunk_parsed = json.loads(raw_chunk)
                except json.JSONDecodeError as parse_err:
                    logging.info(
                        f"{config.conversation_id} chunk {chunk_idx + 1} "
                        f"attempt {attempt + 1}: JSON parse failed "
                        f"({parse_err}), attempting repair..."
                    )
                    chunk_parsed = repair_json(raw_chunk)
                    if chunk_parsed:
                        logging.info(
                            f"{config.conversation_id} chunk {chunk_idx + 1}: "
                            f"JSON repair successful"
                        )

                if chunk_parsed is not None:
                    chunk_annotations = chunk_parsed.get(
                        "utterance_annotations", []
                    )
                    if chunk_annotations:
                        all_utterance_annotations.extend(chunk_annotations)
                        logging.info(
                            f"  Chunk {chunk_idx + 1}/{total_chunks}: "
                            f"{len(chunk_annotations)} utterances annotated"
                        )
                        break
                    else:
                        logging.warning(
                            f"{config.conversation_id} chunk {chunk_idx + 1}: "
                            f"Parsed JSON but utterance_annotations is empty"
                        )

                if chunk_parsed is None:
                    logging.warning(
                        f"{config.conversation_id} chunk {chunk_idx + 1} "
                        f"attempt {attempt + 1}: JSON repair also failed"
                    )

                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

            except Exception as e:
                logging.warning(
                    f"Annotation chunk {chunk_idx + 1} attempt {attempt + 1} "
                    f"failed for {config.conversation_id}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

        if chunk_parsed is None:
            result.annotation_warnings.append(
                f"Chunk {chunk_idx + 1}/{total_chunks} failed after "
                f"{max_retries} attempts"
            )

    # ----- Step 4: Generate session summary -----
    session_summary = {}
    if all_utterance_annotations:
        summary_prompt = build_session_summary_prompt(
            result.transcript, config, all_utterance_annotations
        )
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=summary_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=(
                            "You are a clinical conversation annotation system. "
                            "Respond ONLY with valid JSON."
                        ),
                        temperature=0.5,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high",
                        ),
                    ),
                )
                summary_parsed = None
                try:
                    summary_parsed = json.loads(response.text)
                except json.JSONDecodeError:
                    summary_parsed = repair_json(response.text)

                if summary_parsed:
                    session_summary = summary_parsed.get(
                        "session_summary",
                        summary_parsed  # In case model returns flat structure
                    )
                    break
            except Exception as e:
                logging.warning(
                    f"Session summary attempt {attempt + 1} failed for "
                    f"{config.conversation_id}: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

    # ----- Step 5: Assemble final annotation -----
    result.annotation_time = time.time() - t0_annotation

    if all_utterance_annotations:
        final_annotation = {
            "conversation_id": config.conversation_id,
            "utterance_annotations": all_utterance_annotations,
            "session_summary": session_summary,
        }
        result.annotation = json.dumps(final_annotation, ensure_ascii=False)

        # Validate
        annotation_warnings = validate_annotation(
            final_annotation, config,
            expected_utterance_count=len(patient_turns)
        )
        if annotation_warnings:
            logging.info(
                f"{config.conversation_id}: "
                f"{len(annotation_warnings)} annotation warning(s)"
            )
            for w in annotation_warnings[:3]:
                logging.info(f"  {w}")
            if len(annotation_warnings) > 3:
                logging.info(
                    f"  ... and {len(annotation_warnings) - 3} more"
                )
        result.annotation_warnings.extend(annotation_warnings)
        annotation_ok = True
    else:
        result.annotation_warnings.append(
            "CRITICAL: No utterance annotations produced from any chunk"
        )

    # Compute quality score
    result.quality_score = _compute_quality_score(
        result, parsed_tx, has_annotation=annotation_ok
    )

    return result


def _compute_quality_score(
    result: GeneratedConversation,
    parsed_transcript: dict,
    has_annotation: bool
) -> float:
    """Compute a 0.0-1.0 quality score for a generated conversation.

    Components:
      - Exchange completeness (0.3 weight)
      - Paralinguistic coverage (0.2 weight)
      - Annotation validity (0.3 weight)
      - No critical issues (0.2 weight)
    """
    score = 0.0
    config = result.config

    # Exchange completeness
    expected = config.get("num_exchanges", 10)
    actual = len(parsed_transcript["patient_turns"])
    exchange_ratio = min(actual / max(expected, 1), 1.0)
    score += 0.3 * exchange_ratio

    # Paralinguistic coverage — check both basic annotation presence
    # and whether the extended fields (jitter, pitch_mean, pitch_range)
    # were generated, since the concordance classifier needs them.
    turns_with_para = 0
    turns_with_extended = 0
    for t in parsed_transcript["patient_turns"]:
        para = t.get("paralinguistic", {})
        if para.get("tone") != "neutral" or para.get("energy") != "moderate":
            turns_with_para += 1
        if all(f in para for f in ("jitter", "pitch_mean", "pitch_range")):
            turns_with_extended += 1
    basic_ratio = turns_with_para / max(actual, 1)
    extended_ratio = turns_with_extended / max(actual, 1)
    # Weight: 60% basic coverage + 40% extended field coverage
    score += 0.2 * (0.6 * basic_ratio + 0.4 * extended_ratio)

    # Annotation validity
    if has_annotation and result.annotation:
        try:
            parsed = json.loads(result.annotation)
            utts = parsed.get("utterance_annotations", [])
            if utts:
                # Check marker completeness across utterances
                total_markers = 0
                for utt in utts:
                    total_markers += len(utt.get("markers", {}))
                expected_markers = len(utts) * len(MARKERS)
                marker_coverage = total_markers / max(expected_markers, 1)
                score += 0.3 * min(marker_coverage, 1.0)
        except (json.JSONDecodeError, AttributeError):
            pass  # No annotation score
    elif not has_annotation:
        # If annotation was skipped, don't penalize
        score += 0.15  # Half credit

    # No critical issues
    critical_count = sum(
        1 for issue in parsed_transcript["issues"]
        if "CRITICAL" in issue
    )
    if critical_count == 0:
        score += 0.2
    elif critical_count == 1:
        score += 0.1

    return round(score, 3)


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


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient conversations for marker training"
    )
    parser.add_argument(
        "--num", type=int, default=1,
        help="Number of conversations to generate (default: 1 for testing)"
    )
    parser.add_argument(
        "--output", type=str, default="./test_conversations_high_thinking",
        help="Output directory (default: ./test_conversations_high_thinking)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Pause interval between batches for rate limiting (default: 1)"
    )
    parser.add_argument(
        "--start-id", type=int, default=0,
        help="Starting conversation ID for resuming (default: 0)"
    )
    parser.add_argument(
        "--skip-annotation", action="store_true",
        help="Generate transcripts only, skip annotation (faster, cheaper)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--boost-rare", action="store_true",
        help="After main generation, auto-generate top-up conversations for "
             "underrepresented markers (those below 30-conversation threshold)"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_ID,
        help=f"Gemini model ID to use (default: {DEFAULT_MODEL_ID})"
    )
    args = parser.parse_args()

    # Setup
    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "generation.log"),
            logging.StreamHandler(),
        ]
    )

    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable.")
        print("  Get a key at: https://aistudio.google.com/apikey")
        return

    client = genai.Client(api_key=api_key)
    model_id = args.model
    logging.info(f"Initialized Gemini client with model: {model_id}")
    logging.info(f"Generating {args.num} conversations -> {output_dir}")

    # Check for existing conversations (resume support)
    existing = set()
    meta_dir = output_dir / "metadata"
    if meta_dir.exists():
        for f in meta_dir.glob("CONV-*_meta.json"):
            try:
                with open(f) as fh:
                    meta = json.load(fh)
                    # Only skip if it has a valid transcript (not just metadata)
                    if (meta.get("error") is None
                            and meta.get("has_valid_transcript", False)):
                        existing.add(meta["config"]["conversation_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    if existing:
        logging.info(
            f"Found {len(existing)} existing valid conversations, "
            f"skipping those."
        )

    # Generate conversation configs
    configs = []
    conv_id = args.start_id
    while len(configs) < args.num:
        config = create_conversation_config(conv_id)
        if config.conversation_id not in existing:
            configs.append(config)
        conv_id += 1

    # Run generation
    results = []
    t_start = time.time()

    # Time-based batch pacing
    BATCH_MIN_SECONDS = 12.0

    for i, config in enumerate(tqdm(configs, desc="Generating conversations")):
        if i % args.batch_size == 0:
            batch_start_time = time.time()

        logging.info(
            f"[{i + 1}/{len(configs)}] {config.conversation_id}: "
            f"{config.profile_name} ({config.severity}) - "
            f"{len(config.active_markers)} markers"
        )

        result = generate_conversation(
            client, config, model_id,
            skip_annotation=args.skip_annotation
        )

        save_conversation(result, output_dir, model_id)
        results.append(result)

        # Rate limiting
        is_last_item = (i == len(configs) - 1)
        is_end_of_batch = ((i + 1) % args.batch_size == 0)
        if is_end_of_batch and not is_last_item:
            elapsed = time.time() - batch_start_time
            sleep_needed = BATCH_MIN_SECONDS - elapsed
            if sleep_needed > 0:
                logging.debug(
                    f"Rate limit pause: {sleep_needed:.1f}s "
                    f"(batch took {elapsed:.1f}s)"
                )
                time.sleep(sleep_needed)

    total_time = time.time() - t_start
    save_generation_report(results, output_dir, total_time)

    # Optional rare marker top-up pass
    if args.boost_rare:
        report_path = output_dir / "generation_report.json"
        with open(report_path) as f:
            report = json.load(f)

        underrepresented = report.get("underrepresented_markers", {})
        if not underrepresented:
            logging.info(
                "No underrepresented markers found — skipping boost pass."
            )
        else:
            logging.info(
                f"Starting rare marker boost pass for "
                f"{len(underrepresented)} marker(s): "
                f"{list(underrepresented.keys())}"
            )
            boost_results = []
            boost_id = conv_id

            for rare_marker_id in underrepresented:
                current_count = underrepresented[rare_marker_id]
                top_up_needed = max(0, 30 - current_count)
                logging.info(
                    f"  {rare_marker_id}: {current_count} conversations, "
                    f"generating {top_up_needed} more"
                )

                for _ in range(top_up_needed):
                    boost_profile = dict(RARE_MARKER_BOOST_PROFILE)
                    core = list(RARE_MARKER_BOOST_PROFILE["core_markers"])
                    if rare_marker_id not in core:
                        core.append(rare_marker_id)
                    boost_profile["core_markers"] = core

                    severity = random.choice(["mild", "moderate", "severe"])
                    partner = random.choice(PARTNERS)

                    boost_config = ConversationConfig(
                        conversation_id=f"CONV-{boost_id:04d}",
                        profile_name=boost_profile["name"],
                        profile_description=boost_profile["description"],
                        active_markers=select_markers(boost_profile, severity),
                        severity=severity,
                        patient_age=random.choice(PATIENT_AGES),
                        patient_gender=random.choice(PATIENT_GENDERS),
                        setting=random.choice(SETTINGS),
                        partner_role=partner[0],
                        partner_label=partner[1],
                        num_exchanges=random.randint(10, 20),
                    )
                    boost_id += 1

                    boost_result = generate_conversation(
                        client, boost_config, model_id,
                        skip_annotation=args.skip_annotation
                    )
                    save_conversation(boost_result, output_dir, model_id)
                    boost_results.append(boost_result)

            # Refresh the generation report
            all_results = results + boost_results
            total_time_with_boost = time.time() - t_start
            save_generation_report(
                all_results, output_dir, total_time_with_boost
            )
            logging.info(
                f"Boost pass complete: {len(boost_results)} additional "
                f"conversations generated."
            )

    client.close()

    # =====================================================================
    # TEST VALIDATION SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("TEST VALIDATION SUMMARY")
    print("=" * 70)

    for r in results:
        conv_id = r.config["conversation_id"]
        print(f"\n--- {conv_id} ---")

        checks = []

        # 1. Transcript exists
        has_transcript = bool(r.transcript and r.transcript.strip())
        checks.append(("Transcript generated", has_transcript))

        if has_transcript:
            parsed = parse_transcript(r.transcript)

            # 2. Patient turns found
            pt_count = len(parsed["patient_turns"])
            expected = r.config["num_exchanges"]
            checks.append((
                f"Patient turns: {pt_count}/{expected}",
                pt_count >= expected * 0.8
            ))

            # 3. Millisecond timestamps
            import re as _re
            has_ms = bool(_re.search(r'\[\d+:\d+:\d+\.\d+\]', r.transcript))
            checks.append(("Millisecond timestamps [HH:MM:SS.mmm]", has_ms))

            # 4. Quoted text
            has_quotes = bool(_re.search(
                r'PATIENT.*?:\s*"', r.transcript
            ))
            checks.append(("Text in double quotes", has_quotes))

            # 5. Extended paralinguistic fields
            has_jitter = "jitter:" in r.transcript
            has_pitch_mean = "pitch_mean:" in r.transcript
            has_pitch_range = "pitch_range:" in r.transcript
            checks.append(("jitter field present", has_jitter))
            checks.append(("pitch_mean field present", has_pitch_mean))
            checks.append(("pitch_range field present", has_pitch_range))

            # 6. Space-separated format (no pipes)
            has_pipes = "| energy:" in r.transcript or "| pace:" in r.transcript
            checks.append(("Space-separated para (no pipes)", not has_pipes))

            # 7. Para after text (not before)
            # Check if any PATIENT line has [tone: right after the colon
            para_before = bool(_re.search(
                r'PATIENT[^:]*:\s*\[tone:', r.transcript
            ))
            checks.append((
                "Paralinguistic AFTER text (not before)",
                not para_before
            ))

            # 8. Realistic pacing
            timestamps = _re.findall(
                r'\[(\d+:\d+:\d+(?:\.\d+)?)\]', r.transcript
            )
            if len(timestamps) > 2:
                ts_secs = []
                for ts in timestamps:
                    parts = ts.split(':')
                    h, m = int(parts[0]), int(parts[1])
                    s = float(parts[2])
                    ts_secs.append(h * 3600 + m * 60 + s)
                gaps = [
                    ts_secs[i+1] - ts_secs[i]
                    for i in range(len(ts_secs) - 1)
                ]
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(gaps)
                checks.append((
                    f"Avg turn gap: {avg_gap:.1f}s (should be <20s)",
                    avg_gap < 20
                ))
                checks.append((
                    f"Max turn gap: {max_gap:.1f}s (should be <45s)",
                    max_gap < 45
                ))

            # 9. Extended fields in parsed turns
            turns_with_all_7 = sum(
                1 for t in parsed["patient_turns"]
                if all(f in t.get("paralinguistic", {})
                       for f in ("jitter", "pitch_mean", "pitch_range"))
            )
            checks.append((
                f"Parsed turns with 7/7 para fields: "
                f"{turns_with_all_7}/{pt_count}",
                turns_with_all_7 == pt_count
            ))

        # 10. Annotation valid
        has_annotation = bool(
            r.annotation and r.annotation.strip()
        )
        checks.append(("Annotation generated", has_annotation))

        if has_annotation:
            try:
                ann = json.loads(r.annotation)
                utt_anns = ann.get("utterance_annotations", [])
                checks.append((
                    f"Annotated utterances: {len(utt_anns)}",
                    len(utt_anns) > 0
                ))
                # Check all 38 markers
                if utt_anns:
                    complete = sum(
                        1 for u in utt_anns
                        if len(u.get("markers", {})) == 38
                    )
                    checks.append((
                        f"Utterances with all 38 markers: "
                        f"{complete}/{len(utt_anns)}",
                        complete == len(utt_anns)
                    ))
            except json.JSONDecodeError:
                checks.append(("Annotation JSON valid", False))

        # 11. No errors
        checks.append(("No errors", r.error is None))

        # 12. Quality score
        checks.append((
            f"Quality score: {r.quality_score:.3f}",
            r.quality_score >= 0.7
        ))

        # Print results
        passed = 0
        failed = 0
        for label, ok in checks:
            status = "PASS" if ok else "FAIL"
            marker = "  [+]" if ok else "  [!]"
            print(f"  {marker} {status}: {label}")
            if ok:
                passed += 1
            else:
                failed += 1

        print(f"\n  Result: {passed} passed, {failed} failed")

    print("\n" + "=" * 70)
    total_pass = all(r.error is None for r in results)
    if total_pass and failed == 0:
        print("ALL CHECKS PASSED — safe to run full generation.")
    else:
        print("SOME CHECKS FAILED — review output before full generation.")
    print(f"Test output saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
