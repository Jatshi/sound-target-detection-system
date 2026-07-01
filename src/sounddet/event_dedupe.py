from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from . import CLASS_NAMES

if TYPE_CHECKING:
    from .postprocess import EventRecord, WindowRecord


LABEL_CODES = {
    "Gunshot": "S",
    "Glass": "L",
    "Babycry": "C",
    "NonTarget": "N",
    "Background": "B",
}


def encode_window_sequence(windows: list["WindowRecord"]) -> str:
    return "".join(LABEL_CODES.get(CLASS_NAMES[w.pred], "?") for w in windows)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def dedupe_events_by_edit_distance(events: list["EventRecord"], gap_sec: float = 1.0, max_distance: int = 0) -> list["EventRecord"]:
    """Merge near-duplicate neighboring events using class sequence distance.

    The normal postprocessor already merges same-class spans. This second pass
    catches rapid oscillations such as gunshot-background-gunshot that usually
    represent one acoustic event.
    """

    if not events:
        return []
    ordered = sorted(events, key=lambda e: (e.stream_id, e.start))
    out: list["EventRecord"] = []
    for ev in ordered:
        if not out:
            out.append(ev)
            continue
        last = out[-1]
        same_stream = last.stream_id == ev.stream_id and ev.start - last.end <= gap_sec
        distance = levenshtein(str(last.label), str(ev.label))
        if same_stream and distance <= max_distance:
            label = last.label if last.confidence >= ev.confidence else ev.label
            out[-1] = replace(last, end=max(last.end, ev.end), label=label, confidence=max(last.confidence, ev.confidence))
        else:
            out.append(ev)
    return out
