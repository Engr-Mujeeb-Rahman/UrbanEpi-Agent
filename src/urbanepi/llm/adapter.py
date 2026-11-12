"""The single path for LLM calls: prompts, response cache, call cap, and logging."""
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


class CallLimitReached(RuntimeError):
    """The configured cap on LLM calls was reached."""


def render_prompt(prompt_id: str, variables: dict) -> str:
    """Fill a versioned prompt file."""
    text = (PROMPT_DIR / f"{prompt_id}.txt").read_text(encoding="utf8")
    for name, value in variables.items():
        text = text.replace("{{" + name + "}}", str(value))
    return text


@dataclass
class CachedLLM:
    """Wraps a provider call with a disk cache, so reruns need no API calls."""

    provider_call: Callable[[str], str]
    model: str
    cache_dir: Path
    max_calls: int
    calls: int = 0
    cache_hits: int = 0
    log: list[dict] = field(default_factory=list)

    def complete(self, prompt_id: str, variables: dict) -> str:
        prompt = render_prompt(prompt_id, variables)
        key = hashlib.sha256(json.dumps([self.model, prompt_id, prompt]).encode("utf8")).hexdigest()
        path = Path(self.cache_dir) / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf8"))["response"]
        if self.calls >= self.max_calls:
            raise CallLimitReached(f"Reached {self.max_calls} calls")
        response = self.provider_call(prompt)
        self.calls += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model": self.model, "prompt_id": prompt_id, "prompt": prompt, "response": response},
                                   ensure_ascii=False, indent=1), encoding="utf8")
        self.log.append({"prompt_id": prompt_id, "chars_in": len(prompt), "chars_out": len(response)})
        return response


@dataclass
class ScriptedLLM:
    """Offline stand-in for tests and demos. Returns prepared plans for known questions. It isn't a real parser."""

    plans: dict[str, str]
    prompts_seen: list[str] = field(default_factory=list)

    def complete(self, prompt_id: str, variables: dict) -> str:
        self.prompts_seen.append(render_prompt(prompt_id, variables))
        return self.plans[variables["question"]]
