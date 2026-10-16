"""
Provenance labels ("taint") attached to every value an agent handles.

A value remembers every source it was derived from. Sources are strings:

    "user"                  the user's own request (and plan literals derived from it)
    "contacts"              the user's address book
    "ext:email:<id>"        content of an email someone else sent
    "ext:web:<url>"         a fetched web page
    "ext:calendar:<id>"     an event (invites can come from anyone)

Anything starting with "ext:" is untrusted: a stranger could have written it.
Combining values unions their sources, so taint can only spread, never be
washed out by passing data through a function or an LLM.

This is integrity tracking ("could an attacker have chosen this value?"),
the property that matters for prompt injection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

TRUSTED_SOURCES = frozenset({"user", "contacts", "system"})


@dataclass(frozen=True)
class Labeled:
    value: Any
    sources: frozenset[str] = field(default_factory=frozenset)

    @property
    def trusted(self) -> bool:
        return all(s in TRUSTED_SOURCES for s in self.sources)

    @property
    def untrusted_sources(self) -> list[str]:
        return sorted(s for s in self.sources if s not in TRUSTED_SOURCES)

    def with_sources(self, more: Iterable[str]) -> "Labeled":
        return Labeled(self.value, self.sources | frozenset(more))


def trusted(value: Any, source: str = "user") -> Labeled:
    return Labeled(value, frozenset({source}))


def join(*vals: Labeled) -> frozenset[str]:
    """Sources of a value computed from all of `vals`."""
    out: frozenset[str] = frozenset()
    for v in vals:
        out |= v.sources
    return out


def describe(sources: frozenset[str]) -> str:
    return ", ".join(sorted(sources)) or "none"
