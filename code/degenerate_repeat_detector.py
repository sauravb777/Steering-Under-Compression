"""
Degenerate Repeat Detector
==========================
Detects and truncates degenerate repetition loops in LLM output.

When small/medium language models hit low-confidence situations, they
often collapse into repeating the same phrase endlessly ("main entry
point, main entry point, main entry point..."). This module catches
those loops and returns the clean text before the repetition began.

Operates on decoded strings — no tokenizer dependency, works with
any LLM API that returns text.

Usage:
    from degenerate_repeat_detector import check_for_loops

    result = check_for_loops(response_text)
    if result.looped:
        clean = result.clean_text  # truncated before the loop
        print(f"Loop detected: '{result.pattern}' x{result.repeats}")

Thresholds (repeats required to trigger):
    n=1-2 words:  5+ consecutive repeats
    n=3-5 words:  3+ consecutive repeats
    n=6+ words:   2+ consecutive repeats

Performance: ~10-50 microseconds per call on typical LLM output.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LoopResult:
    """Result of a loop detection check."""
    looped: bool
    clean_text: str
    pattern: Optional[str] = None
    position: Optional[int] = None
    repeats: int = 0
    ngram_size: int = 0


def _get_threshold(n: int) -> int:
    """
    Minimum consecutive repeats to trigger detection at n-gram size n.

    Short phrases repeat legitimately in natural language ("of the",
    "it is"), so we require more evidence. Longer phrases essentially
    never repeat consecutively in well-formed text.
    """
    if n <= 2:
        return 5
    elif n <= 5:
        return 3
    else:
        return 2


def _check_char_loops(text, min_len=2, max_len=10, threshold=5):
    """
    Character-level substring repetition check.

    Catches sub-word loops like "nessnessnessness" where a short
    character sequence repeats many times inside a single token/word.
    Only triggers on high repeat counts to avoid false positives.

    Returns:
        Tuple of (char_position, pattern_len, repeats, pattern) or None.
    """
    for n in range(min_len, max_len + 1):
        i = 0
        while i <= len(text) - (n * threshold):
            chunk = text[i:i + n]

            # Skip pure whitespace or punctuation-only chunks
            if not any(c.isalpha() for c in chunk):
                i += 1
                continue

            count = 1
            j = i + n
            while j + n <= len(text) and text[j:j + n] == chunk:
                count += 1
                j += n

            if count >= threshold:
                # Clean text is everything before the loop
                clean = text[:i].rstrip(' ,;:')
                # Return in a format compatible with best_hit
                # Use char position as word position (close enough for truncation)
                return (i, n, count, chunk, True)

            i += 1

    return None


def check_for_loops(
    text: str,
    max_ngram: int = 20,
    min_ngram: int = 1,
) -> LoopResult:
    """
    Check a string for degenerate repetition loops.

    Scans for repeated word-level n-grams from size min_ngram to
    max_ngram. Returns the earliest/shortest loop found, with the
    text truncated to just before the repetition began.

    Args:
        text: The LLM output string to check.
        max_ngram: Largest n-gram size to check (default 20, ~80 chars).
        min_ngram: Smallest n-gram size to check (default 1).

    Returns:
        LoopResult with looped=False if clean, or looped=True with
        clean_text truncated before the loop started.
    """
    if not text or not text.strip():
        return LoopResult(looped=False, clean_text=text)

    words = text.split()
    if len(words) < 3:
        return LoopResult(looped=False, clean_text=text)

    # Track the earliest loop found
    # Format: (position, ngram_size, repeats, pattern, is_char_level)
    best_hit = None

    for n in range(min_ngram, min(max_ngram + 1, len(words) // 2 + 1)):
        threshold = _get_threshold(n)

        # Slide through the word list
        i = 0
        while i <= len(words) - (n * threshold):
            phrase = tuple(words[i:i + n])
            count = 1

            # Count consecutive repeats of this phrase
            j = i + n
            while j + n <= len(words) and tuple(words[j:j + n]) == phrase:
                count += 1
                j += n

            if count >= threshold:
                # Found a loop — track if it's the earliest one
                if best_hit is None or i < best_hit[0]:
                    best_hit = (i, n, count, ' '.join(phrase), False)
                break  # No need to keep scanning at this n-gram size

            i += 1

    # --- Sub-word / character-level check ---
    # Catches loops like "nessnessnessness" that appear as one giant
    # word to the whitespace splitter. Scans for any character substring
    # of length 2-10 repeated 5+ times consecutively.
    if best_hit is None:
        best_hit = _check_char_loops(text)

    if best_hit is None:
        return LoopResult(looped=False, clean_text=text)

    pos, ngram_size, repeats, pattern, is_char_level = best_hit

    if is_char_level:
        # Character-level: pos is a character index
        clean_text = text[:pos].rstrip(' ,;:')
    else:
        # Word-level: pos is a word index
        clean_words = words[:pos]
        clean_text = ' '.join(clean_words).rstrip(' ,;:')

    # If the loop was at the very start, return empty
    if not clean_text:
        clean_text = ''

    return LoopResult(
        looped=True,
        clean_text=clean_text,
        pattern=pattern,
        position=pos,
        repeats=repeats,
        ngram_size=ngram_size,
    )


class LoopDetector:
    """
    Stateful detector for streaming use.

    Feed chunks as they arrive; check .looping after each feed.

        detector = LoopDetector()
        for chunk in stream:
            detector.feed(chunk_text)
            if detector.looping:
                stream.close()
                break
    """

    def __init__(self, max_ngram: int = 20, min_ngram: int = 1):
        self.max_ngram = max_ngram
        self.min_ngram = min_ngram
        self._buffer = ''
        self._result = None

    def feed(self, text: str):
        """Append text and check for loops."""
        self._buffer += text
        self._result = check_for_loops(
            self._buffer,
            max_ngram=self.max_ngram,
            min_ngram=self.min_ngram,
        )

    @property
    def looping(self) -> bool:
        """True if a loop has been detected in the buffered text."""
        return self._result is not None and self._result.looped

    @property
    def result(self) -> Optional[LoopResult]:
        """The current detection result, or None if feed() not yet called."""
        return self._result

    @property
    def clean_text(self) -> str:
        """The clean text (pre-loop) if looping, else the full buffer."""
        if self._result and self._result.looped:
            return self._result.clean_text
        return self._buffer

    def reset(self):
        """Clear the buffer and result."""
        self._buffer = ''
        self._result = None
