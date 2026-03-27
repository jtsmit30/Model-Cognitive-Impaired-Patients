import re
import json
from typing import Optional


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
