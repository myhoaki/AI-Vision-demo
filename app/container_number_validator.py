"""ISO 6346 container number candidate normalization and check digit validation."""

from __future__ import annotations

import re

_CANDIDATE = re.compile(r"(?<![A-Z0-9])[A-Z]{4}[0-9]{7}(?![A-Z0-9])")
_LETTER_VALUES = {
    letter: value
    for letter, value in zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        (10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38),
    )
}


def normalize_ocr_text(text: str) -> str:
    """Uppercase OCR text and retain alphanumeric characters for inspection."""
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def check_digit(number: str) -> int:
    if not re.fullmatch(r"[A-Z]{4}[0-9]{6}", number):
        raise ValueError("Expected 4 letters followed by 6 serial digits")
    values = [_LETTER_VALUES[c] for c in number[:4]] + [int(c) for c in number[4:]]
    remainder = sum(value * (2**position) for position, value in enumerate(values)) % 11
    return remainder % 10


def candidates(text: str) -> list[dict]:
    """Return candidates without guessing, repairing, or inserting characters."""
    normalized = normalize_ocr_text(text)
    # Search the original normalized string with separators removed. Any candidate is
    # exactly what OCR emitted after normalization; the original text remains separate.
    found = []
    for match in _CANDIDATE.finditer(normalized):
        value = match.group(0)
        expected = check_digit(value[:10])
        found.append({
            "candidate": value,
            "validFormat": True,
            "checkDigit": int(value[-1]),
            "expectedCheckDigit": expected,
            "validCheckDigit": int(value[-1]) == expected,
        })
    return found
