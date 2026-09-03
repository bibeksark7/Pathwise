"""Versioned prompts, loaded from files.

Prompts are the part of an LLM application most likely to change and least likely to
be reviewed. Keeping them as `.md` files under version control means a prompt change
is a diff someone can read, and recording `name + version + checksum` on every call
means any stored output can be traced back to the exact text that produced it.

Three decisions worth stating:

**The active version is pinned explicitly**, not inferred as "highest number".
Dropping a `v2.md` into the tree changes nothing until it is promoted in
``ACTIVE_VERSIONS`` — a one-line, reviewable diff that the evaluation suite gates.
Auto-activating the newest file would make an unreviewed prompt live the moment it
was written.

**Templates use ``$variable``, not ``{variable}``.** Prompts routinely contain literal
JSON braces in their examples, and `str.format` chokes on them — producing either a
crash or, worse, a silently mangled prompt.

**Substitution is strict.** A missing variable raises rather than rendering the
literal placeholder into the prompt, where the model would dutifully try to interpret
`$concept_name` as content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template
from typing import Any, Final

from pathwise.api.errors import NotFoundError, ValidationError

PROMPTS_DIR: Final = Path(__file__).resolve().parent

#: The version of each prompt that is actually used. Adding a file does not activate
#: it; promoting it here does. Keep this list alphabetical.
ACTIVE_VERSIONS: Final[dict[str, str]] = {
    "decision_explain": "v1",
    "diagnostic_generate": "v1",
    "goal_parse": "v1",
    "roadmap_annotate": "v1",
}

_VERSION_PATTERN: Final = re.compile(r"^v\d+$")
_PLACEHOLDER_PATTERN: Final = re.compile(r"\$\{?([a-z_][a-z0-9_]*)\}?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Prompt:
    """One versioned prompt template."""

    name: str
    version: str
    template: str

    @property
    def checksum(self) -> str:
        """A digest of the template text.

        Catches the case version numbers miss: a prompt edited in place without a
        version bump. Two calls recorded as `roadmap_generate v1` with different
        checksums means someone changed the text underneath a live version.
        """
        return hashlib.sha256(self.template.encode("utf-8")).hexdigest()[:16]

    @property
    def variables(self) -> frozenset[str]:
        """Every placeholder the template declares."""
        return frozenset(_PLACEHOLDER_PATTERN.findall(self.template))

    def render(self, **values: Any) -> str:
        """Fill in the template.

        Raises:
            ValidationError: if a declared variable was not supplied. Rendering a
                prompt with `$concept_name` still in it would send the placeholder to
                the model as if it were content.
        """
        missing = self.variables - set(values)
        if missing:
            raise ValidationError(
                f"Prompt '{self.name}' is missing required variables.",
                prompt=self.name,
                version=self.version,
                missing=sorted(missing),
            )
        try:
            return Template(self.template).substitute(**values)
        except (KeyError, ValueError) as exc:
            raise ValidationError(
                f"Prompt '{self.name}' failed to render: {exc}",
                prompt=self.name,
                version=self.version,
            ) from exc

    def __repr__(self) -> str:
        return f"<Prompt {self.name}@{self.version} {self.checksum}>"


class PromptRegistry:
    """Loads and caches prompt templates from disk."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or PROMPTS_DIR
        self._cache: dict[tuple[str, str], Prompt] = {}

    def get(self, name: str, version: str | None = None) -> Prompt:
        """Fetch a prompt, defaulting to its active version.

        Raises:
            NotFoundError: if the prompt or version does not exist.
        """
        resolved = version or ACTIVE_VERSIONS.get(name)
        if resolved is None:
            raise NotFoundError(
                f"No active version registered for prompt '{name}'. "
                "Add it to ACTIVE_VERSIONS in registry.py.",
                prompt=name,
                known=sorted(ACTIVE_VERSIONS),
            )

        key = (name, resolved)
        if key not in self._cache:
            self._cache[key] = self._load(name, resolved)
        return self._cache[key]

    def _load(self, name: str, version: str) -> Prompt:
        path = self._directory / name / f"{version}.md"
        if not path.is_file():
            raise NotFoundError(
                f"Prompt file not found: {name}/{version}.md",
                prompt=name,
                version=version,
                expected_path=str(path),
            )
        return Prompt(name=name, version=version, template=path.read_text(encoding="utf-8"))

    def versions(self, name: str) -> tuple[str, ...]:
        """Every version on disk for a prompt, oldest first."""
        directory = self._directory / name
        if not directory.is_dir():
            return ()
        return tuple(
            sorted(
                (p.stem for p in directory.glob("v*.md") if _VERSION_PATTERN.match(p.stem)),
                key=lambda v: int(v[1:]),
            )
        )

    def discovered_names(self) -> tuple[str, ...]:
        """Every prompt directory on disk, whether registered or not."""
        return tuple(
            sorted(
                p.name
                for p in self._directory.iterdir()
                if p.is_dir() and not p.name.startswith("_") and any(p.glob("v*.md"))
            )
        )

    def clear_cache(self) -> None:
        self._cache.clear()


@lru_cache(maxsize=1)
def get_registry() -> PromptRegistry:
    """The process-wide prompt registry."""
    return PromptRegistry()


def get_prompt(name: str, version: str | None = None) -> Prompt:
    """Convenience accessor for the process-wide registry."""
    return get_registry().get(name, version)
