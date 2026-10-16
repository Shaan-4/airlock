"""
The firewall: decide whether a tool call may run, from the provenance of its
arguments alone.

The rule that matters: an argument that controls *where an effect lands*
(recipient, URL, file to share, emails to delete) must not have been chosen
by untrusted content. Data itself may still flow (an email body may contain a
summary of a web page) as long as the destination was chosen by the user.

"Trusted or contact" adds an endorsement step: an address that came out of an
email is still allowed if it exactly matches an entry in the user's address
book, because then the user, not the email, vouched for it. An attacker can
make the agent *mention* any address, but can't add one to the contacts.
"""
from __future__ import annotations

from dataclasses import dataclass

from .taint import Labeled, describe
from .workspace import ANY, TRUSTED, TRUSTED_OR_CONTACT, Tool, Workspace


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


class PolicyViolation(Exception):
    def __init__(self, tool: str, param: str, reason: str):
        super().__init__(f"blocked {tool}.{param}: {reason}")
        self.tool, self.param, self.reason = tool, param, reason


def _as_list(v):
    return v if isinstance(v, (list, tuple)) else [v]


def check(tool: Tool, args: dict[str, Labeled], ws: Workspace) -> Decision:
    for p in tool.params:
        if p.policy == ANY or p.name not in args:
            continue
        arg = args[p.name]
        if arg.trusted:
            continue
        if p.policy == TRUSTED_OR_CONTACT:
            known = set(ws.contacts.values())
            if all(isinstance(x, str) and x.lower() in {k.lower() for k in known}
                   for x in _as_list(arg.value)):
                continue    # endorsed by the address book
            return Decision(False, f"{p.name} came from untrusted content "
                                   f"({describe(frozenset(arg.untrusted_sources))}) "
                                   f"and is not a known contact")
        if p.policy == TRUSTED:
            return Decision(False, f"{p.name} came from untrusted content "
                                   f"({describe(frozenset(arg.untrusted_sources))})")
    return Decision(True)


def enforce(tool: Tool, args: dict[str, Labeled], ws: Workspace) -> None:
    d = check(tool, args, ws)
    if not d.allowed:
        param = d.reason.split(" ", 1)[0]
        raise PolicyViolation(tool.name, param, d.reason)
