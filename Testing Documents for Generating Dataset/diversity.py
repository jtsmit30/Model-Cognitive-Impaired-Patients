"""
diversity.py — Composable Trait Architecture for Synthetic Conversation Generation

Provides all characterization, structural variation, and batch-level tracking
needed to produce a diverse training corpus. No dependencies beyond the Python
standard library and the project's own trait data.

Components:
  - Patient biography composition (5 trait pools)
  - Age-calibration layer (graduated thresholds 55–89)
  - Partner system (role guidelines + dynamic traits + turn distribution)
  - Conversation arcs (10 structural patterns)
  - Conversation seeds (15 situational openings)
  - Severity anchors (per-profile behavioral definitions)
  - DiversityTracker (least-used-first selection with batch-aware context)

See: Diversity_System_Documentation.docx for design rationale.
"""

import random
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from new_traits import (
    TRAIT_COMMUNICATION_STYLE_NEW,
    TRAIT_VULNERABILITY_STYLE_NEW,
    TRAIT_SOCIAL_ORIENTATION_NEW,
    TRAIT_LIFE_ANCHORS_NEW,
)

from StaticSeedData import (
    PARTNER_GUIDELINES,
    TRAIT_PARTNER_DYNAMIC,
    CONVERSATION_ARCS,
    CONVERSATION_SEEDS,
    SEVERITY_ANCHORS,
    TRAIT_OCCUPATION_NEW,
)
# ============================================================================
# AGE-CALIBRATION LAYER
# ============================================================================

def build_age_calibration(age: int) -> str:
    """Generate an age-appropriate constraint block for the patient.

    Addresses five dimensions with graduated thresholds at 65, 75, and 85.
    The block is injected into the generation prompt after the biography.
    """

    if age < 55 or age > 89:
        logging.warning(
            f"Age {age} is outside the calibrated range 55-89. "
            f"Clamping to nearest boundary."
        )
        age = max(55, min(89, age))

    lines = [f"AGE-APPROPRIATE CONSTRAINTS (age {age}):"]

    # --- Cohort anchoring ---
    if age >= 85:
        lines.append(
            "- Formative years were during or just after WWII. Cultural "
            "references come from the 1940s-50s. Pre-digital norms are "
            "deeply ingrained — the patient grew up without television in "
            "many cases and certainly without computers."
        )
    elif age >= 75:
        lines.append(
            "- Formative years were in the 1940s-50s. Cultural references "
            "come from the post-war period through the early 1960s. "
            "Technology adoption is selective — comfortable with telephone "
            "and television but may not use digital devices fluently."
        )
    elif age >= 65:
        lines.append(
            "- Formative years were in the 1950s-60s. Cultural references "
            "span the 1960s-70s. May use some technology but it is not "
            "native. Professional career was largely pre-digital."
        )
    else:
        lines.append(
            "- Formative years were in the 1960s-70s. Cultural references "
            "span the 1970s-80s. May be comfortable with basic technology. "
            "Professional career straddled the analogue-digital transition."
        )

    # --- Speech timing ---
    if age >= 85:
        lines.append(
            "- Natural speech pauses of 2-4 seconds between hearing a "
            "question and beginning to respond. Occasional sentence "
            "restarts where the patient begins, pauses, and reformulates. "
            "This is NORMAL aging, not necessarily a cognitive marker — "
            "the pause_before values should reflect this baseline."
        )
    elif age >= 75:
        lines.append(
            "- Slightly slower speech pace than a younger person. "
            "Occasional retrieval pauses of 1-2 seconds that resolve "
            "naturally. Fatigue accumulates over the conversation — the "
            "patient may speak more slowly or briefly in the final third."
        )
    elif age >= 65:
        lines.append(
            "- Occasional word-retrieval pauses that resolve quickly. "
            "Speech pace is essentially normal but may slow when the "
            "patient is tired or discussing complex topics."
        )
    else:
        lines.append(
            "- Speech timing is essentially normal for an adult. No "
            "age-related calibration needed for pace or retrieval."
        )

    # --- Vocabulary ---
    if age >= 85:
        lines.append(
            "- Uses habitual phrases and generational idioms that a "
            "younger person would not. Vocabulary register tends toward "
            "the formal — 'supper' not 'dinner,' 'picture show' not "
            "'movie,' 'icebox' not 'fridge.' These are lifelong habits, "
            "not affectations. The patient does NOT use contemporary "
            "slang or internet-era expressions."
        )
    elif age >= 75:
        lines.append(
            "- May use dated expressions and a slightly more formal "
            "register than contemporary casual speech. Vocabulary is "
            "rich from a lifetime of reading and conversation but "
            "grounded in the patient's era."
        )
    # No calibration needed for 55-74

    # --- Physical context ---
    if age >= 85:
        lines.append(
            "- Physical limitations are a CONSTANT background presence — "
            "arthritis, reduced mobility, fatigue, sensory decline. "
            "Energy is LIMITED. The patient cannot sustain high-energy "
            "interaction (rapid speech, raised voice, extended argument) "
            "for more than 1-2 exchanges before needing to pause or "
            "withdraw. Physical comfort affects everything."
        )
    elif age >= 75:
        lines.append(
            "- Physical limitations are present and manageable but shape "
            "daily life — joint stiffness, reduced stamina, perhaps "
            "hearing or vision changes. The patient is aware of their "
            "body as a constraint in a way they were not at 60."
        )
    elif age >= 65:
        lines.append(
            "- Emerging awareness of physical change — recovery takes "
            "longer, energy is not infinite, some activities require more "
            "effort than they used to. The body is a background presence "
            "but not yet a primary preoccupation."
        )
    else:
        lines.append(
            "- Physical context is generally unremarkable. The patient "
            "may have specific health concerns but age-related physical "
            "decline is not yet a dominant theme."
        )

    # --- Negative constraints ---
    if age >= 85:
        lines.append(
            "- CRITICAL CONSTRAINTS: The patient does NOT sustain sarcasm "
            "or ironic performance across multiple turns — the executive "
            "function load is too high. No rapid-fire retorts in quick "
            "succession. No extended high-energy verbal conflict lasting "
            "more than 1-2 exchanges. Each response should feel "
            "considered rather than reactive. The patient CAN be sharp, "
            "funny, and articulate — but in brief moments, not sustained "
            "performances."
        )
    elif age >= 75:
        lines.append(
            "- The patient should not sound like a 40-year-old with grey "
            "hair. Sustained high-energy verbal sparring is unrealistic. "
            "Wit may be present but is dry and economical rather than "
            "rapid and performative."
        )
    elif age >= 65:
        lines.append(
            "- The patient should not sound like a 40-year-old. Speech "
            "patterns, cultural references, and energy levels should be "
            "age-appropriate. Avoid generic 'old person' stereotypes — "
            "be specific to this character."
        )
    # No negative constraints for 55-64

    return "\n".join(lines)


# ============================================================================
# TURN DISTRIBUTION COMPUTATION
# ============================================================================

# Profile-severity effect on partner share (higher = patient talks less).
# Values derived from documentation Section 5.2.
_PROFILE_SEVERITY_OFFSETS = {
    "Major Depressive Episode": {
        "mild": 0.10, "moderate": 0.25, "severe": 0.45,
    },
    "Anxiety with Depression": {
        "mild": 0.05, "moderate": 0.15, "severe": 0.30,
    },
    "Early Cognitive Decline": {
        "mild": 0.05, "moderate": 0.20, "severe": 0.35,
    },
    "Social Withdrawal / Isolation": {
        "mild": 0.15, "moderate": 0.30, "severe": 0.45,
    },
    "Caregiver Conflict / Irritability": {
        "mild": -0.05, "moderate": 0.05, "severe": 0.15,
    },
    "Masked Depression": {
        "mild": 0.00, "moderate": 0.10, "severe": 0.25,
    },
    "Healthy Baseline": {
        "mild": 0.00, "moderate": 0.00, "severe": 0.00,
        "none": 0.00,
    },
}

# Behavioral descriptions for each turn distribution category.
_TURN_DISTRIBUTION_DESCRIPTIONS = {
    "partner_dominant": (
        "The PARTNER dominates the conversation. Patient responses are "
        "minimal — mostly 1-3 word answers, nods, or silence. The partner "
        "carries both sides of the conversation, sometimes producing "
        "multiple turns in a row with only minimal patient acknowledgment "
        "between them. The partner fills the patient's silence rather than "
        "waiting."
    ),
    "partner_leads": (
        "The PARTNER's turns are longer and more frequent than the "
        "patient's. The patient responds but with shorter, less elaborated "
        "turns. The partner does more conversational work — initiating "
        "topics, asking follow-up questions, bridging silences. The "
        "patient is present but not driving."
    ),
    "roughly_even": (
        "Both speakers contribute approximately equally. There is a mix "
        "of shorter and longer turns from both. Topic initiation is "
        "shared. Neither speaker consistently dominates or withdraws. "
        "This feels like a balanced conversation between two engaged "
        "participants."
    ),
    "patient_leads": (
        "The PATIENT's turns are longer and more elaborated than the "
        "partner's. The partner asks brief questions, gives short "
        "acknowledgments, and steers gently. The patient provides more "
        "of the conversational content and may initiate topics."
    ),
    "patient_dominant": (
        "The PATIENT does most of the talking — venting, storytelling, "
        "explaining, or filling conversational space. The partner "
        "primarily listens, offering brief responses and encouragements. "
        "The partner's turns are short and facilitative rather than "
        "substantive."
    ),
}


def build_turn_distribution(
    profile_name: str,
    severity: str,
    partner_dynamic_id: str,
    comm_style_offset: float,
) -> tuple:
    """Compute turn distribution from profile, partner dynamic, and comm style.

    Args:
        profile_name: Clinical profile name.
        severity: Severity level string (e.g. "mild", "moderate", "severe").
        partner_dynamic_id: ID of the selected partner dynamic trait.
        comm_style_offset: partner_share_offset from the selected comm style trait.

    Returns:
        (partner_share, category_name, behavioral_description)

    partner_share is a float between 0.30 and 0.80.
    """

    # Base share: 0.50 (even)
    share = 0.50

    # 1. Profile + severity offset
    profile_offsets = _PROFILE_SEVERITY_OFFSETS.get(
        profile_name,
        _PROFILE_SEVERITY_OFFSETS.get("Healthy Baseline", {}),
    )
    share += profile_offsets.get(severity, 0.0)

    # 2. Partner dynamic offset
    partner_offset = 0.0
    for dynamic in TRAIT_PARTNER_DYNAMIC:
        if dynamic["id"] == partner_dynamic_id:
            partner_offset = dynamic.get("partner_share_offset", 0.0)
            break
    share += partner_offset

    # 3. Communication style offset (read directly from the trait dict)
    share += comm_style_offset

    # Clamp to valid range
    share = max(0.30, min(0.80, share))

    # Map to category
    if share >= 0.70:
        category = "partner_dominant"
    elif share >= 0.60:
        category = "partner_leads"
    elif share >= 0.45:
        category = "roughly_even"
    elif share >= 0.35:
        category = "patient_leads"
    else:
        category = "patient_dominant"

    description = _TURN_DISTRIBUTION_DESCRIPTIONS[category]

    return share, category, description


# ============================================================================
# BIOGRAPHY COMPOSITION
# ============================================================================

def compose_biography(
    comm_style: dict,
    vulnerability: dict,
    social_orientation: dict,
    life_anchor: dict,
    occupation: dict,
) -> str:
    """Compose a patient biography from independently selected traits.

    Each argument is a trait dict with 'id' and 'text' keys.
    Returns a formatted text block for prompt injection.
    """

    return (
        "PATIENT CHARACTER:\n"
        "\n"
        "Communication style:\n"
        f"  {comm_style['text']}\n"
        "\n"
        "How this person handles distress:\n"
        f"  {vulnerability['text']}\n"
        "\n"
        "How this person relates to the conversation partner:\n"
        f"  {social_orientation['text']}\n"
        "\n"
        "Conversational anchors (topics they gravitate toward):\n"
        f"  {life_anchor['text']}\n"
        "\n"
        "Occupational background:\n"
        f"  {occupation['text']}"
    )


def compose_partner(
    role: str,
    dynamic: dict,
) -> str:
    """Compose a partner description from role guidelines and dynamic trait.

    Args:
        role: Partner role string (e.g. "Occupational therapist")
        dynamic: Partner dynamic trait dict with 'id' and 'text' keys

    Returns a formatted text block for prompt injection.
    """

    guidelines = PARTNER_GUIDELINES.get(role, "")
    if not guidelines:
        logging.warning(
            f"No partner guidelines found for role '{role}'. "
            f"Using generic guidelines."
        )
        guidelines = (
            "Engages with the patient respectfully. Follows the "
            "conversational lead appropriate to their role."
        )

    return (
        "CONVERSATION PARTNER CHARACTER:\n"
        "\n"
        f"Role expectations ({role}):\n"
        f"  {guidelines}\n"
        "\n"
        "This partner's conversational style:\n"
        f"  {dynamic['text']}"
    )


# ============================================================================
# DIVERSITY TRACKER
# ============================================================================

class DiversityTracker:
    """Tracks all trait selections and steers toward least-used options.

    Maintains a Counter per tracked dimension. Uses least-used-first
    selection with random tie-breaking to ensure even distribution.

    Dimensions tracked:
      - communication_style
      - vulnerability_style
      - social_orientation
      - life_anchors
      - occupation
      - partner_dynamic
      - conversation_arc
      - conversation_seed

    Also provides batch-aware diversity context for prompt injection.
    """

    # The pools this tracker draws from.
    _POOL_REGISTRY = {
        "communication_style": TRAIT_COMMUNICATION_STYLE_NEW,
        "vulnerability_style": TRAIT_VULNERABILITY_STYLE_NEW,
        "social_orientation": TRAIT_SOCIAL_ORIENTATION_NEW,
        "life_anchors": TRAIT_LIFE_ANCHORS_NEW,
        "occupation": TRAIT_OCCUPATION_NEW,
        "partner_dynamic": TRAIT_PARTNER_DYNAMIC,
        "conversation_arc": CONVERSATION_ARCS,
        "conversation_seed": CONVERSATION_SEEDS,
    }

    def __init__(self):
        """Initialise empty counters for every tracked dimension."""
        self._counters: dict[str, Counter] = {}
        for dimension, pool in self._POOL_REGISTRY.items():
            self._counters[dimension] = Counter(
                {trait["id"]: 0 for trait in pool}
            )
        self._total_conversations = 0

    # ------------------------------------------------------------------
    # Core selection algorithm
    # ------------------------------------------------------------------

    def select_trait(self, dimension: str) -> dict:
        """Select the least-used trait from a dimension's pool.

        Algorithm (from documentation Section 3.1):
          1. Find the minimum count across all options in the pool.
          2. Collect all traits that share that minimum count.
          3. Randomly select one from the candidates.
          4. Increment that trait's count immediately.

        Returns the full trait dict (with 'id', 'text', and any extras).
        """
        if dimension not in self._counters:
            raise ValueError(
                f"Unknown dimension '{dimension}'. "
                f"Valid: {list(self._counters.keys())}"
            )

        counter = self._counters[dimension]
        pool = self._POOL_REGISTRY[dimension]

        # Build a lookup from id -> trait dict
        pool_lookup = {trait["id"]: trait for trait in pool}

        # Find minimum count
        min_count = min(counter.values())

        # Collect candidates at that count
        candidates = [
            trait_id for trait_id, count in counter.items()
            if count == min_count
        ]

        # Random selection among tied candidates
        selected_id = random.choice(candidates)

        # Increment immediately
        counter[selected_id] += 1

        return pool_lookup[selected_id]

    # ------------------------------------------------------------------
    # Convenience: select a full patient character
    # ------------------------------------------------------------------

    def select_patient_traits(self) -> dict:
        """Select one trait from each patient pool.

        Returns a dict keyed by dimension name, values are trait dicts.
        """
        return {
            "communication_style": self.select_trait("communication_style"),
            "vulnerability_style": self.select_trait("vulnerability_style"),
            "social_orientation": self.select_trait("social_orientation"),
            "life_anchors": self.select_trait("life_anchors"),
            "occupation": self.select_trait("occupation"),
        }

    # ------------------------------------------------------------------
    # Full scene composition
    # ------------------------------------------------------------------

    def compose_scene(
        self,
        profile_name: str,
        severity: str,
        partner_role: str,
        patient_age: int,
    ) -> dict:
        """Compose a complete diversity scene for one conversation.

        Selects all traits, composes biography and partner, builds age
        calibration, computes turn distribution, selects arc and seed,
        and generates the batch-aware diversity context.

        Returns a dict with all diversity fields ready for injection
        into ConversationConfig and the generation prompt.
        """

        # 1. Patient traits
        patient_traits = self.select_patient_traits()

        # 2. Partner dynamic
        partner_dynamic = self.select_trait("partner_dynamic")

        # 3. Conversation arc and seed
        arc = self.select_trait("conversation_arc")
        seed = self.select_trait("conversation_seed")

        # 4. Compose biography
        biography = compose_biography(
            comm_style=patient_traits["communication_style"],
            vulnerability=patient_traits["vulnerability_style"],
            social_orientation=patient_traits["social_orientation"],
            life_anchor=patient_traits["life_anchors"],
            occupation=patient_traits["occupation"],
        )

        # 5. Age calibration
        age_calibration = build_age_calibration(patient_age)

        # 6. Partner description
        partner_description = compose_partner(partner_role, partner_dynamic)

        # 7. Severity anchors
        severity_anchor = get_severity_anchor(profile_name, severity)

        # 8. Turn distribution
        share, turn_category, turn_description = build_turn_distribution(
            profile_name=profile_name,
            severity=severity,
            partner_dynamic_id=partner_dynamic["id"],
            comm_style_offset=patient_traits["communication_style"].get(
                "partner_share_offset", 0.0
            ),
        )

        # 9. Batch-aware diversity context
        diversity_context = self._build_diversity_context()

        return {
            # Trait IDs (for metadata / tracking)
            "communication_style_id": patient_traits["communication_style"]["id"],
            "vulnerability_style_id": patient_traits["vulnerability_style"]["id"],
            "social_orientation_id": patient_traits["social_orientation"]["id"],
            "life_anchors_id": patient_traits["life_anchors"]["id"],
            "occupation_id": patient_traits["occupation"]["id"],
            "partner_dynamic_id": partner_dynamic["id"],
            "arc_id": arc["id"],
            "seed_id": seed["id"],
            # Composed text blocks (for prompt injection)
            "biography": biography,
            "age_calibration": age_calibration,
            "partner_description": partner_description,
            "severity_anchor": severity_anchor,
            "arc_description": arc["text"],
            "seed_description": seed["text"],
            "turn_distribution_share": share,
            "turn_distribution_category": turn_category,
            "turn_distribution_description": turn_description,
            "diversity_context": diversity_context,
        }

    # ------------------------------------------------------------------
    # Recording and persistence
    # ------------------------------------------------------------------

    def record_conversation(self, scene: dict) -> None:
        """Record a completed conversation's trait selections.

        Called after a conversation is successfully generated and saved.
        The select_trait method already incremented counts at selection
        time, so this method only updates the total conversation count.
        """
        self._total_conversations += 1

    def load_existing(self, metadata_dir: Path) -> int:
        """Restore tracker state from existing metadata files.

        Reads all CONV-*_meta.json files in the directory and rebuilds
        the counters from their stored trait IDs. Returns the number of
        conversations loaded.

        This enables resume: if a generation run is interrupted, the
        tracker can pick up where it left off.
        """
        loaded = 0
        if not metadata_dir.exists():
            return loaded

        for meta_file in sorted(metadata_dir.glob("CONV-*_meta.json")):
            try:
                with open(meta_file) as f:
                    meta = json.load(f)

                # Skip conversations that failed
                if meta.get("error") is not None:
                    continue
                if not meta.get("has_valid_transcript", False):
                    continue

                config = meta.get("config", {})

                # Restore counts for each tracked dimension.
                # The metadata stores trait IDs under these keys.
                _id_fields = {
                    "communication_style": "communication_style_id",
                    "vulnerability_style": "vulnerability_style_id",
                    "social_orientation": "social_orientation_id",
                    "life_anchors": "life_anchors_id",
                    "occupation": "occupation_id",
                    "partner_dynamic": "partner_dynamic_id",
                    "conversation_arc": "arc_id",
                    "conversation_seed": "seed_id",
                }

                for dimension, config_key in _id_fields.items():
                    trait_id = config.get(config_key)
                    if trait_id and trait_id in self._counters[dimension]:
                        self._counters[dimension][trait_id] += 1

                loaded += 1

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logging.warning(
                    f"Could not load metadata from {meta_file.name}: {e}"
                )
                continue

        self._total_conversations = loaded
        logging.info(
            f"DiversityTracker restored state from {loaded} existing "
            f"conversations."
        )
        return loaded

    # ------------------------------------------------------------------
    # Batch-aware diversity context
    # ------------------------------------------------------------------

    def _build_diversity_context(self) -> str:
        """Generate a CORPUS DIVERSITY CONTEXT block for prompt injection.

        Tells the generation model what structural patterns already exist
        in the corpus so it can deliberately avoid overrepresented ones.
        Includes warnings when any single pattern exceeds 30% usage.
        """
        if self._total_conversations < 5:
            return ""  # Too few conversations for meaningful context

        lines = ["CORPUS DIVERSITY CONTEXT:"]
        lines.append(
            f"This is conversation #{self._total_conversations + 1} in "
            f"the corpus. The following patterns have been used so far:"
        )

        # Check each dimension for overrepresentation
        overrepresented = []
        threshold = max(1, self._total_conversations * 0.30)

        for dimension in ("conversation_arc", "conversation_seed",
                          "partner_dynamic"):
            counter = self._counters[dimension]
            pool_size = len(self._POOL_REGISTRY[dimension])

            for trait_id, count in counter.most_common(3):
                if count > threshold:
                    overrepresented.append((dimension, trait_id, count))

        if overrepresented:
            lines.append("")
            lines.append("OVERREPRESENTED PATTERNS (avoid these):")
            for dim, trait_id, count in overrepresented:
                pct = (count / self._total_conversations) * 100
                lines.append(
                    f"  - {dim}: '{trait_id}' has been used {count} times "
                    f"({pct:.0f}% of corpus). Avoid this pattern."
                )

        # Show arc distribution
        arc_counter = self._counters["conversation_arc"]
        lines.append("")
        lines.append("Conversation arc usage so far:")
        for arc_id, count in arc_counter.most_common():
            if count > 0:
                lines.append(f"  - {arc_id}: {count}")

        lines.append("")
        lines.append(
            "Aim for a conversation that feels structurally different "
            "from the patterns listed above."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_distribution_report(self) -> dict:
        """Return a summary of current trait distributions.

        Useful for monitoring balance during generation.
        """
        report = {"total_conversations": self._total_conversations}

        for dimension, counter in self._counters.items():
            counts = dict(counter)
            values = list(counts.values())
            report[dimension] = {
                "counts": counts,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "unique_used": sum(1 for v in values if v > 0),
                "total_available": len(values),
            }

        return report


# ============================================================================
# SEVERITY ANCHOR LOOKUP
# ============================================================================

def get_severity_anchor(profile_name: str, severity: str) -> str:
    """Look up the behavioral severity anchor for a profile.

    Falls back to the generic anchor if the profile has no specific one.
    Returns a formatted instruction string.
    """
    anchors = SEVERITY_ANCHORS.get(
        profile_name,
        SEVERITY_ANCHORS["_generic"],
    )
    anchor_text = anchors.get(severity, anchors.get("mild", ""))

    if not anchor_text:
        anchor_text = SEVERITY_ANCHORS["_generic"].get(severity, "")

    return (
        f"SEVERITY DEFINITION ({severity.upper()}):\n"
        f"  {anchor_text}"
    )
