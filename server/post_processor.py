import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptMessage:
    type: str
    text: str
    segment_id: str
    start: float
    end: float
    confidence: float
    words: List = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "text": self.text,
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "words": [
                w if isinstance(w, dict) else {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in self.words
            ],
        }


# ---------------------------------------------------------------------------
# Currency word → symbol/digit mappings
# ---------------------------------------------------------------------------
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ORDINAL_ONES = {
    "first": (1, "st"), "second": (2, "nd"), "third": (3, "rd"),
    "fourth": (4, "th"), "fifth": (5, "th"), "sixth": (6, "th"),
    "seventh": (7, "th"), "eighth": (8, "th"), "ninth": (9, "th"),
    "tenth": (10, "th"), "eleventh": (11, "th"), "twelfth": (12, "th"),
    "thirteenth": (13, "th"), "fourteenth": (14, "th"), "fifteenth": (15, "th"),
    "sixteenth": (16, "th"), "seventeenth": (17, "th"), "eighteenth": (18, "th"),
    "nineteenth": (19, "th"), "twentieth": (20, "th"),
    "thirtieth": (30, "th"), "fortieth": (40, "th"), "fiftieth": (50, "th"),
}
_MONTHS = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June", "july": "July",
    "august": "August", "september": "September", "october": "October",
    "november": "November", "december": "December",
}
_CURRENCY_WORDS = {
    "dollars": "$", "dollar": "$",
    "euros": "€", "euro": "€",
    "pounds": "£", "pound": "£",
}
_ALL_NUMBERS = {**_ONES, **_TENS}


def _words_to_int(words: list[str]) -> int | None:
    """Convert a list of number words to an integer. Returns None on failure."""
    total = 0
    current = 0
    for w in words:
        if w in _ONES:
            current += _ONES[w]
        elif w in _TENS:
            current += _TENS[w]
        else:
            return None
    total += current
    return total if (total > 0 or words == ["zero"]) else None


class PostProcessor:
    def __init__(self, itn_processor=None, punct_model=None, use_punct_model: bool = False):
        self.itn_processor = itn_processor
        self.punct_model = punct_model
        self.use_punct_model = use_punct_model

    def process(self, result, type: str = "final") -> TranscriptMessage:
        text = result.text

        try:
            text = self._capitalise(text)
        except Exception as e:
            logger.warning("_capitalise failed: %s", e)

        try:
            text = self._apply_itn(text)
        except Exception as e:
            logger.warning("_apply_itn failed: %s", e)

        try:
            text = self._apply_punctuation(text)
        except Exception as e:
            logger.warning("_apply_punctuation failed: %s", e)

        return self._build_message(text, result, type)

    _FILLERS = {"um", "uh", "er"}

    def _capitalise(self, text: str) -> str:
        if not text:
            return text
        first_word = text.split()[0].lower() if text.split() else ""
        if first_word in self._FILLERS:
            return text
        return text[0].upper() + text[1:]

    def _apply_itn(self, text: str) -> str:
        if self.itn_processor is not None:
            try:
                return self.itn_processor.normalize(text, verbose=False)
            except Exception as e:
                logger.warning("itn_processor.normalize failed, falling back to regex: %s", e)
        return self._apply_itn_regex(text)

    def _apply_punctuation(self, text: str) -> str:
        if self.use_punct_model and self.punct_model is not None:
            return self.punct_model(text)
        if text and text[-1] not in ".!?":
            text += "."
        return text

    def _apply_itn_regex(self, text: str) -> str:
        # Rule 1 — currency: "fifty dollars" → "$50"
        def replace_currency(m):
            num_words = m.group(1).strip().lower().split()
            currency_word = m.group(2).lower()
            symbol = _CURRENCY_WORDS.get(currency_word)
            val = _words_to_int(num_words)
            if val is not None and symbol:
                return f"{symbol}{val}"
            return m.group(0)

        currency_pattern = (
            r'\b(' + '|'.join(re.escape(k) for k in sorted(
                list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True
            )) + r'(?:\s+(?:' + '|'.join(re.escape(k) for k in sorted(
                list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True
            )) + r'))*)\s+(' + '|'.join(re.escape(k) for k in _CURRENCY_WORDS) + r')\b'
        )
        text = re.sub(currency_pattern, replace_currency, text, flags=re.IGNORECASE)

        # Rule 2 — ordinal dates: "fourteenth of march" → "14th of March"
        ordinal_pat = (
            r'\b(' + '|'.join(re.escape(k) for k in sorted(_ORDINAL_ONES, key=len, reverse=True)) +
            r')\s+of\s+(' + '|'.join(re.escape(k) for k in _MONTHS) + r')\b'
        )
        def replace_ordinal_date(m):
            ord_word = m.group(1).lower()
            month_word = m.group(2).lower()
            num, suffix = _ORDINAL_ONES[ord_word]
            return f"{num}{suffix} of {_MONTHS[month_word]}"
        text = re.sub(ordinal_pat, replace_ordinal_date, text, flags=re.IGNORECASE)

        # Rule 3 — cardinals: "three meetings" → "3 meetings"
        cardinal_pat = (
            r'\b((?:' + '|'.join(re.escape(k) for k in sorted(
                list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True
            )) + r')(?:\s+(?:' + '|'.join(re.escape(k) for k in sorted(
                list(_ONES.keys()) + list(_TENS.keys()), key=len, reverse=True
            )) + r'))*)\b'
        )
        def replace_cardinal(m):
            words = m.group(1).strip().lower().split()
            val = _words_to_int(words)
            if val is not None:
                return str(val)
            return m.group(0)
        text = re.sub(cardinal_pat, replace_cardinal, text, flags=re.IGNORECASE)

        # Rule 4 — percentages: "forty percent" → "40%" (already digit after rule 3)
        text = re.sub(r'\b(\d+)\s+percent\b', r'\1%', text, flags=re.IGNORECASE)

        # Rule 5 — times: "3 30 pm" or "3 30 am" → "3:30 PM"
        text = re.sub(
            r'\b(\d+)\s+(\d{2})\s+(am|pm)\b',
            lambda m: f"{m.group(1)}:{m.group(2)} {m.group(3).upper()}",
            text, flags=re.IGNORECASE
        )

        # Rule 6 — sentence I: standalone "i" → "I"
        text = re.sub(r'(?<![a-zA-Z])i(?![a-zA-Z])', 'I', text)

        return text

    def _build_message(self, text: str, result, type: str) -> TranscriptMessage:
        return TranscriptMessage(
            type=type,
            text=text,
            segment_id=result.segment_id,
            start=getattr(result, "start", 0.0),
            end=getattr(result, "end", result.duration),
            confidence=result.confidence,
            words=result.words,
        )
