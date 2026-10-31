"""Confidence grade (A/B/C) from data coverage and match quality, with reasons."""
from __future__ import annotations

from . import store


def confidence_grade(features: dict, growth: dict, psh_source: str, level_source: str) -> dict:
    score, reasons = 0, []

    if features.get("source") == "satellite":
        score += 2
        reasons.append("Site features come from satellite and map data")
    elif features.get("source") == "satellite + user overrides":
        score += 1
        reasons.append("Some site features were changed by hand")
    else:
        reasons.append("No satellite features for this location: values are user-entered")

    n_good = growth.get("n_good", 0)
    if growth.get("source") == "assumed":
        reasons.append("No matched villages: growth uses an assumed curve")
    elif n_good >= 15:
        score += 2
        reasons.append(f"{n_good} closely matched villages")
    elif n_good >= 5:
        score += 1
        reasons.append(f"Only {n_good} closely matched villages")
    else:
        reasons.append("Few close matches: the range is wider")

    if any(m.get("source") == "meter" for m in growth.get("matches", [])):
        score += 1
        reasons.append("Matches include sites with real smart-meter data")

    if level_source.startswith("meter") or level_source.startswith("model"):
        score += 1
        reasons.append("Starting level calibrated on real meter data")
    else:
        reasons.append("Starting level from published benchmarks")

    if store.validation() is not None:
        score += 1
        reasons.append("Method validated on held-out metered sites")

    if not psh_source.startswith("NASA"):
        reasons.append("Solar resource is an assumed value")

    grade = "A" if score >= 6 else "B" if score >= 3 else "C"
    return {"grade": grade, "score": score, "reasons": reasons}
