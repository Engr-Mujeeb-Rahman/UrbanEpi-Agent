"""Rule-based filter applied to a question before it is sent to any LLM provider.

It blocks requests that ask for identifying details or for individual people, so such questions never leave the health
authority. It was written from the dev questions and a general list of identifier types, before it was run on the test set.
A question it passes still goes through the typed-plan validator and policy check after parsing.
"""
import re
import unicodedata

_IDENTIFIERS = (
    r"phone( number)?s?", r"telephone", r"mobile( number)?s?", r"cell ?phone", r"e-?mail", r"(home |street )?address(es)?",
    r"where (do|does|did) .{0,40}\blives?\b", r"names?\b", r"surnames?", r"first names?", r"date of birth", r"birth ?dates?", r"birthdays?",
    r"curp", r"rfc", r"nss", r"social security", r"passport", r"id numbers?", r"identity", r"identification", r"record (numbers?|ids?)",
    r"case (numbers?|ids?)", r"folio", r"contact (details|information|info)", r"personal (details|data|information)", r"photos?",
)
_INDIVIDUALS = (
    r"\bwho (is|are|was|were)\b", r"\bwhich (patients?|people|persons?|individuals?)\b", r"\b(list|show|give|tell)( me)? (all |the )?(individual )?(patients?|people|persons?|cases)\b.{0,20}\b(names?|details|who)\b",
    r"\bindividual (patients?|records?|cases?)\b", r"\b(each|every) (patient|person)'?s?\b", r"\bpatient[- ]level\b", r"\brow[- ]level\b", r"\braw (records?|data)\b",
    r"\bcall them\b", r"\bcontact them\b", r"\bvisit them\b", r"\bfirst (\w+ )?(cases?|patients?)\b",
)
_PATTERNS = [re.compile(p) for p in _IDENTIFIERS + _INDIVIDUALS]


def _normal(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return " ".join(folded.lower().split())


def blocked(question: str) -> str | None:
    """Return the matched pattern if the question must not be sent to the LLM, otherwise None."""
    text = _normal(question)
    for pattern in _PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None
