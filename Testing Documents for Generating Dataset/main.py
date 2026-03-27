import os
import re
import json
import time
import random
import argparse
import logging
from pathlib import Path

from google import genai
from tqdm import tqdm

from generate_conversations_testing import (
    DEFAULT_MODEL_ID,
    RARE_MARKER_BOOST_PROFILE,
    PARTNERS,
    PATIENT_AGES,
    PATIENT_GENDERS,
    SETTINGS,
    ConversationConfig,
    save_conversation,
    save_generation_report,
    parse_transcript,
)
from PromptGenerationLogik import (
    create_conversation_config,
    generate_conversation,
    select_markers,
)


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
            has_ms = bool(re.search(r'\[\d+:\d+:\d+\.\d+\]', r.transcript))
            checks.append(("Millisecond timestamps [HH:MM:SS.mmm]", has_ms))

            # 4. Quoted text
            has_quotes = bool(re.search(
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
            para_before = bool(re.search(
                r'PATIENT[^:]*:\s*\[tone:', r.transcript
            ))
            checks.append((
                "Paralinguistic AFTER text (not before)",
                not para_before
            ))

            # 8. Realistic pacing
            timestamps = re.findall(
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
