"""Check that every number in LLM text appears in the evidence it was given, and belongs to the place named."""
import re
import unicodedata

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_ALWAYS_ALLOWED = {95.0}  # The stated confidence level.


def _plain(text: str) -> str:
    """Lowercase and remove accents, so place names match however they're written."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()


def _numbers_in(value) -> list[float]:
    numbers: list[float] = []
    _flatten(value, numbers, set())
    return numbers


def misplaced_values(text: str, evidence: dict) -> list[str]:
    """Return numbers in a sentence naming one borough that aren't that borough's own or shared values.

    Sentences naming several boroughs, or none, can't be attributed, so only the plain number check covers them.
    """
    cells = evidence.get("boroughs", [])
    if len(cells) < 2:
        return []
    shared = _numbers_in({k: v for k, v in evidence.items() if k != "boroughs"})
    problems = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        plain = _plain(sentence)
        named = [c for c in cells if _plain(c["district"]) in plain]
        if len(named) != 1:
            continue
        allowed = _numbers_in(named[0]) + shared
        for token in _NUMBER.findall(_DATE.sub(" ", sentence)):
            value = float(token.replace(",", ""))
            decimals = len(token.split(".")[1]) if "." in token else 0
            if value not in _ALWAYS_ALLOWED and not any(round(abs(k), decimals) == abs(value) for k in allowed):
                problems.append(f"{token} doesn't belong to {named[0]['district']}")
    return problems


def _flatten(value, numbers: list[float], dates: set[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _flatten(item, numbers, dates)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, numbers, dates)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        numbers.append(float(value))
    elif isinstance(value, str):
        dates.update(_DATE.findall(value))


def ungrounded_values(text: str, evidence: dict) -> list[str]:
    """Return numbers and dates in the text that the evidence doesn't support, allowing whole-number rounding."""
    numbers: list[float] = []
    dates: set[str] = set()
    _flatten(evidence, numbers, dates)
    problems = [d for d in _DATE.findall(text) if d not in dates]
    remainder = _DATE.sub(" ", text)
    for token in _NUMBER.findall(remainder):
        value = float(token.replace(",", ""))
        if value in _ALWAYS_ALLOWED:
            continue
        decimals = len(token.split(".")[1]) if "." in token else 0
        if not any(round(known, decimals) == value or round(abs(known), decimals) == abs(value) for known in numbers):
            problems.append(token)
    return problems
