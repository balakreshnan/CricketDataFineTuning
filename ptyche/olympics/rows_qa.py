"""One verifiable question per source row of athlete_events.csv (1:1 coverage of the source data).

Every row (an athlete's entry in one event at one Games) anchors exactly one question that names that athlete:
  event context   (default)  the event's full entry list at that Games (<= MAX_EVENT_ROWS entrants);
                  ev_same_noc  entrants from the anchor athlete's NOC (incl. them)
                  ev_older     entrants older than the athlete         (age known)
                  ev_taller    entrants taller than the athlete        (height known)
                  ev_noc_medal_count medal rows won by the athlete's NOC in this event
  athlete context (event too big, or name not unique in it): the athlete's career record (<= MAX_CAREER_ROWS);
                  at_events_at  events entered at that Games
                  at_medals_at  medals won at that Games
                  at_events_before events entered at Games in earlier years
                  at_other_events_at events at that Games other than this row's event (always unique per row)
The template is picked by a hash of the row, falling back to the next one whose prompt is still unused, so every
row gets a distinct prompt. Tables have a fixed per-table row order, so all questions about one event share the
same prompt prefix (Dynamo's KV-aware router reuses it). Split: rows of benchmark-test athletes go to test.
"""
import csv
import hashlib
import json
import random

from olympics_qa import SYSTEM_PROMPT, is_test

MAX_EVENT_ROWS = 150
MAX_CAREER_ROWS = 80


def h(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16)


def num(v):
    return None if v in ("NA", "") else float(v)


def load_rows(csv_path):
    """Unique rows of athlete_events.csv in file order (same order/index as pandas drop_duplicates)."""
    seen, rows = set(), []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = tuple(r.values())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"index": len(rows), "ID": int(r["ID"]), "Name": r["Name"], "Team": r["Team"], "NOC": r["NOC"],
                         "Games": r["Games"], "Year": int(r["Year"]), "City": r["City"], "Sport": r["Sport"],
                         "Event": r["Event"], "Age": num(r["Age"]), "Height": num(r["Height"]),
                         "Weight": num(r["Weight"]), "Medal": "" if r["Medal"] == "NA" else r["Medal"]})
    return rows


def fmt(v):
    if v is None:
        return "n/a"
    if v == "":
        return "-"
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


_RENDERED = {}


def render(rows, cols, key):
    """Markdown table, built once per table; fixed row order per table -> shared prompt prefix across questions."""
    if key not in _RENDERED:
        rs = list(rows)
        random.Random(h("order", key)).shuffle(rs)
        _RENDERED[key] = "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                                   ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rs])
    return _RENDERED[key]


def event_questions(r, ev, name):
    """(qtype, gold, question) candidates for row r inside its event table ev."""
    out = [("ev_same_noc", sum(e["NOC"] == r["NOC"] for e in ev),
            f"How many entrants in this table represented the same NOC as {name}, counting {name}?")]
    if r["Age"] is not None:
        out.append(("ev_older", sum(e["Age"] is not None and e["Age"] > r["Age"] for e in ev),
                    f"How many entrants were older than {name}? Ignore entrants whose age is n/a."))
    if r["Height"] is not None:
        out.append(("ev_taller", sum(e["Height"] is not None and e["Height"] > r["Height"] for e in ev),
                    f"How many entrants were taller than {name}? Ignore entrants whose height is n/a."))
    # v1 ("How many medal rows ... (count every row, ...)") was read as "rows of that NOC": 30% pass@1, retired
    out.append(("ev_noc_medal_count", sum(e["NOC"] == r["NOC"] and e["Medal"] != "" for e in ev),
                f"Count the rows in this table where the entrant is from the same NOC as {name} AND the Medal "
                f"column is Gold, Silver or Bronze (rows with '-' are not medals; every member of a medal-winning "
                f"team has their own row and counts separately). How many such rows are there?"))
    return out


def athlete_questions(r, car, name):
    at_g = [c for c in car if c["Games"] == r["Games"]]
    return [("at_events_at", len(at_g), f"How many events did {name} enter at the {r['Games']} Olympics?"),
            ("at_medals_at", sum(c["Medal"] != "" for c in at_g),
             f"How many medals did {name} win at the {r['Games']} Olympics?"),
            ("at_events_before", sum(c["Year"] < r["Year"] for c in car),
             f"How many events had {name} entered at Olympic Games held in years before {r['Year']}?"),
            ("at_other_events_at", len(at_g) - 1,
             f"Not counting {r['Event']}, how many other events did {name} enter at the {r['Games']} Olympics?")]


def candidates(r, events, careers):
    name = r["Name"]
    ev = events[(r["Event"], r["Games"])]
    cands = []
    if len(ev) <= MAX_EVENT_ROWS and sum(e["Name"] == name for e in ev) == 1:
        ctx = (f"All entrants in {r['Event']} at the {r['Games']} Olympics ({r['City']}). Medal '-' means no medal; "
               f"team events list every team member.\n\n"
               + render(ev, ["Name", "NOC", "Age", "Height", "Weight", "Medal"], (r["Event"], r["Games"])))
        cands += [(qt, g, ctx, q) for qt, g, q in event_questions(r, ev, name)]
    car = careers[r["ID"]]
    if len(car) <= MAX_CAREER_ROWS:
        ctx = (f"Olympic record of {name} ({r['Team']}). One row per event entered; Medal '-' means no medal.\n\n"
               + render(car, ["Games", "Year", "City", "Sport", "Event", "Age", "Medal"], ("ath", r["ID"])))
        cands += [(qt, g, ctx, q) for qt, g, q in athlete_questions(r, car, name)]
    return cands


def build(csv_path, out_train, out_test, test_frac=0.1, skip_first=0):
    """Stream one question per row to out_train / out_test (file objects). skip_first=1 writes, for every row, the
    candidate AFTER its usual one instead (fallback pass for rows whose first question had no correct trace)."""
    rows = load_rows(csv_path)
    events, careers = {}, {}
    for r in rows:
        events.setdefault((r["Event"], r["Games"]), []).append(r)
        careers.setdefault(r["ID"], []).append(r)
    used, n, uncovered = set(), {"train": 0, "test": 0}, []
    qtypes = {}
    for r in rows:
        cands = candidates(r, events, careers)
        if not cands:
            uncovered.append(r["index"])
            continue
        start = h(r["index"]) % len(cands)
        for qt, gold, ctx, q in (cands[start:] + cands[:start])[skip_first:] or cands:
            key = h(ctx, q)
            if key in used:
                continue
            used.add(key)
            split = "test" if is_test(r["ID"], test_frac) else "train"
            rec = {"id": f"row{r['index']}-{qt}", "qtype": qt, "answer_kind": "int", "gold": int(gold),
                   "source": {"row_index": r["index"], "athlete_id": r["ID"], "name": r["Name"], "games": r["Games"],
                              "event": r["Event"], "noc": r["NOC"]},
                   "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"{ctx}\n\n{q}"}]}
            (out_test if split == "test" else out_train).write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[split] += 1
            qtypes[qt] = qtypes.get(qt, 0) + 1
            break
        else:
            uncovered.append(r["index"])
    return {"source_rows": len(rows), "questions_train": n["train"], "questions_test": n["test"],
            "rows_covered": len(rows) - len(uncovered), "uncovered_rows": len(uncovered),
            "uncovered_examples": uncovered[:20], "qtypes": qtypes}
