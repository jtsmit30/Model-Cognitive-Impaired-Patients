import random
import json
import time
import logging
from dataclasses import asdict

from google import genai
from google.genai import types

from generate_conversations_testing import (
    ConversationConfig,
    GeneratedConversation,
    MARKERS,
    CLINICAL_PROFILES,
    SEVERITIES,
    PARTNERS,
    PATIENT_AGES,
    PATIENT_GENDERS,
    SETTINGS,
    parse_transcript,
    validate_annotation,
)
from JsonUtils import repair_json


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


