"""The account-safety rule, enforced mechanically rather than by review.

social/live/ observes a browsing session the user drives. It must never navigate, scroll or
click page content: that is what keeps it indistinguishable from a person reading LinkedIn,
and it is the difference between a passive tool and one that risks the user's account.

One exception exists -- Browser.login() opens a login page at the user's explicit command --
and it is whitelisted by an explicit marker comment, so adding a second one is a deliberate
act someone has to write down, not an accident.
"""

import re
from pathlib import Path

LIVE = Path(__file__).resolve().parent.parent / "social" / "live"
ALLOW = "# noqa: laya-allow-navigation"

FORBIDDEN = {
    "goto": re.compile(r"\.goto\s*\("),
    "mouse.wheel": re.compile(r"\.mouse\s*\.\s*wheel\s*\("),
    "scroll_into_view": re.compile(r"\.scroll_into_view(_if_needed)?\s*\("),
    "page.click": re.compile(r"\bpage\s*\.\s*click\s*\("),
    "locator.click": re.compile(r"\.locator\([^)]*\)\s*\.\s*click\s*\("),
    "keyboard.press": re.compile(r"\.keyboard\s*\.\s*press\s*\("),
    "js scrollTo": re.compile(r"window\.scrollTo|scrollBy\s*\(|scrollIntoView\s*\("),
}


def test_live_package_never_drives_the_page():
    offences = []
    for path in sorted(LIVE.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if ALLOW in line:
                continue
            for name, pattern in FORBIDDEN.items():
                if pattern.search(line):
                    offences.append(f"{path.name}:{lineno}: {name} -> {line.strip()}")
    assert not offences, (
        "social/live/ must never drive the page. Found:\n  " + "\n  ".join(offences)
        + f"\nIf a call is genuinely required, append `{ALLOW}` and say why in the spec."
    )


def test_the_whitelist_is_not_load_bearing_everywhere():
    """Exactly one navigation escape hatch should exist. More means the rule is eroding."""
    allowed = [
        f"{p.name}:{n}"
        for p in sorted(LIVE.rglob("*.py"))
        for n, line in enumerate(p.read_text().splitlines(), 1)
        if ALLOW in line
    ]
    assert len(allowed) <= 1, f"more than one navigation exception: {allowed}"
