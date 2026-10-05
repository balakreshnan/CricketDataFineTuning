"""Verifiable cricket reasoning questions built from Valarmathy/CricketData (T20I ball-by-ball).

Every sampled delivery (source row) yields 3 questions whose answers are computed exactly from the data:
  run_rate    current run rate after this ball; overs are given in cricket notation (14.3 = 14 overs 3 balls)
  chase_rate  innings 2: run rate needed to win; innings 1 (full 20 overs): rate scored over the rest of the innings
  milestone   balls the striker needs to reach their next multiple of 50 at their current strike rate
Splits are by match, so no test match contributes training rows. Standard library only.
"""
import csv
import json
import random
import re
from collections import Counter, defaultdict

QTYPES = ("run_rate", "chase_rate", "milestone")
SYSTEM_PROMPT = ("You are a cricket analyst. Work through the problem step by step, then give the final answer "
                 "on its own line as 'Final answer: <number>'.")
RATE_TOL = 0.011  # answers are rounded to 2 dp; allow one unit of rounding difference


def overs(balls):
    return f"{balls // 6}.{balls % 6}"


def load_matches(csv_path):
    matches = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            matches[r["Match ID"]].append(r)
    return matches


def clean_matches(matches):
    """Drop matches whose target is not first-innings total + 1 (rain-revised / DLS): answers would be ambiguous."""
    keep = {}
    for mid, rows in matches.items():
        i1 = [r for r in rows if r["Innings"] == "1"]
        if i1 and int(i1[-1]["Target Score"]) == int(i1[-1]["Innings Runs"]) + 1:
            keep[mid] = rows
    return keep


def _innings_info(rows):
    info = {}
    for inn in ("1", "2"):
        rs = [r for r in rows if r["Innings"] == inn]
        if rs:
            info[inn] = {"final_runs": int(rs[-1]["Innings Runs"]), "full_20": int(rs[-1]["Balls Remaining"]) == 0}
    return info


def eligible(r, info):
    if r["Valid Ball"] != "1":
        return False
    bb = 120 - int(r["Balls Remaining"])
    runs, br, bf = int(r["Innings Runs"]), int(r["Total Batter Runs"]), int(r["Batter Balls Faced"])
    if not (12 <= bb <= 108 and runs >= 1 and br >= 1 and bf >= 4):
        return False
    if r["Innings"] == "1":
        inn = info.get("1")
        return bool(inn and inn["full_20"] and inn["final_runs"] > runs)
    return float(r["Runs to Get"] or 0) > 0


def make_questions(r, info):
    inn = r["Innings"]
    bat1, bat2 = r["Bat First"], r["Bat Second"]
    batting, bowling = (bat1, bat2) if inn == "1" else (bat2, bat1)
    bb = 120 - int(r["Balls Remaining"])
    runs, wkts = int(r["Innings Runs"]), int(r["Innings Wickets"])
    batter, br, bf = r["Batter"], int(r["Total Batter Runs"]), int(r["Batter Balls Faced"])
    ns, nsr, nsf = r["Non Striker"], int(r["Total Non Striker Runs"]), int(r["Non Striker Balls Faced"])
    target = int(r["Target Score"])

    ctx = f"T20 International: {bat1} v {bat2} at {r['Venue']}, {r['Date']}.\n"
    if inn == "1":
        ctx += f"{batting} are batting first. "
    else:
        ctx += f"{batting} are chasing a target of {target} set by {bowling}. "
    ctx += (f"After {overs(bb)} overs (cricket notation: overs.balls) they are {runs}/{wkts}. "
            f"{batter} is on {br} off {bf} balls and {ns} is on {nsr} off {nsf}. {r['Bowler']} bowled the last delivery.")

    qs = {}
    qs["run_rate"] = (f"What is {batting}'s current run rate in runs per over? Round to 2 decimal places.",
                      round(runs * 6 / bb, 2), "rate")
    rem = 120 - bb
    if inn == "2":
        need = target - runs
        qs["chase_rate"] = (f"What run rate (runs per over) do {batting} need over the rest of their 20 overs to "
                            f"reach the target? Round to 2 decimal places.", round(need * 6 / rem, 2), "rate")
    else:
        final = info["1"]["final_runs"]
        qs["chase_rate"] = (f"{batting} finished their 20 overs on {final}. At what run rate (runs per over) did they "
                            f"score over the rest of their innings from this point? Round to 2 decimal places.",
                            round((final - runs) * 6 / rem, 2), "rate")
    milestone = (br // 50 + 1) * 50
    need_runs = milestone - br
    qs["milestone"] = (f"{batter} wants to reach {milestone} runs. If {batter} keeps scoring at their own current "
                       f"strike rate (their runs per ball faced so far), how many more balls must {batter} face to reach "
                       f"{milestone}? Round up to a whole number of balls.",
                       -(-need_runs * bf // br), "int")  # ceil(need_runs / (br / bf)) in exact integer arithmetic

    row_index = r.get("") or r.get("Unnamed: 0")  # the CSV's unnamed index column = row number in the source file
    source = {"row_index": int(row_index),
              "match_id": int(r["Match ID"]), "date": r["Date"], "venue": r["Venue"], "bat_first": bat1, "bat_second": bat2,
              "innings": int(inn), "over": int(r["Over"]), "ball": int(r["Ball"]), "batter": batter, "bowler": r["Bowler"],
              "innings_runs": runs, "innings_wickets": wkts, "balls_remaining": int(r["Balls Remaining"]),
              "target_score": target if inn == "2" else None, "batter_runs": br, "batter_balls_faced": bf}
    out = []
    for qt in QTYPES:
        q, gold, kind = qs[qt]
        out.append({
            "id": f"{r['Match ID']}-{inn}-{source['row_index']}-{qt}",
            "qtype": qt, "answer_kind": kind, "gold": gold,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{ctx}\n\n{q}"}],
            "source": source,
        })
    return out


def build_splits(csv_path, n_train, n_test, test_frac=0.1, seed=0):
    """Sample source rows (spread across matches/innings) and return (train_questions, test_questions, stats)."""
    rng = random.Random(seed)
    raw = load_matches(csv_path)
    matches = clean_matches(raw)
    mids = sorted(matches)
    rng.shuffle(mids)
    n_test_m = max(1, int(len(mids) * test_frac))
    split_m = {"test": mids[:n_test_m], "train": mids[n_test_m:]}
    out = {}
    for split, want in (("train", n_train), ("test", n_test)):
        pools = []
        for mid in split_m[split]:
            info = _innings_info(matches[mid])
            el = [r for r in matches[mid] if eligible(r, info)]
            rng.shuffle(el)
            if el:
                pools.append((info, el))
        rows, rnd = [], 0
        while len(rows) < want and any(len(el) > rnd for _, el in pools):  # one row per match per round
            for info, el in pools:
                if rnd < len(el) and len(rows) < want:
                    rows.append((el[rnd], info))
            rnd += 1
        out[split] = [q for r, info in rows for q in make_questions(r, info)]
    stats = {"matches_total": len(raw), "matches_dropped_revised_target": len(raw) - len(matches),
             "matches_train": len(split_m["train"]), "matches_test": len(split_m["test"]),
             "source_rows_train": len(out["train"]) // 3, "source_rows_test": len(out["test"]) // 3,
             "questions_train": len(out["train"]), "questions_test": len(out["test"])}
    return out["train"], out["test"], stats


# ------------------------------------------------------------------ answer checking

_FINAL = re.compile(r"final answer\s*[:：]\s*\**\s*([-+]?\d[\d,]*(?:\.\d+)?)", re.I)


def split_reasoning(text):
    """(reasoning, answer_text, closed_think). Thinking-mode prompts already open <think>, so the
    generated text is '<reasoning></think>\\n\\n<answer>'."""
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip(), True
    return text.replace("<think>", "").strip(), "", False


def parse_answer(text):
    _, answer, closed = split_reasoning(text)
    hits = _FINAL.findall(answer if closed else "")
    if not hits:
        return None
    try:
        return float(hits[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(pred, gold, kind):
    if pred is None:
        return False
    if kind == "int":
        return abs(pred - gold) < 1e-6
    return abs(pred - gold) <= RATE_TOL


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ================================================================== full one-to-one coverage
# Every source row (all 425,119, incl. extras, innings boundaries and rain-affected matches) gets >= 1 question.
# The original three types keep their exact wording; three more cover rows where those are undefined.
ALL_QTYPES = ("run_rate", "chase_rate", "milestone", "projection", "strike_rate", "legal_balls")
SOURCE_ROWS = 425_119  # Valarmathy/CricketData raw/ball_by_ball_it20.csv


def _ctx(r):
    """Same situation text as make_questions()."""
    inn, bat1, bat2 = r["Innings"], r["Bat First"], r["Bat Second"]
    batting, bowling = (bat1, bat2) if inn == "1" else (bat2, bat1)
    bb = 120 - int(r["Balls Remaining"])
    ctx = f"T20 International: {bat1} v {bat2} at {r['Venue']}, {r['Date']}.\n"
    if inn == "1":
        ctx += f"{batting} are batting first. "
    else:
        ctx += f"{batting} are chasing a target of {int(r['Target Score'])} set by {bowling}. "
    ctx += (f"After {overs(bb)} overs (cricket notation: overs.balls) they are {int(r['Innings Runs'])}/"
            f"{int(r['Innings Wickets'])}. {r['Batter']} is on {int(r['Total Batter Runs'])} off "
            f"{int(r['Batter Balls Faced'])} balls and {r['Non Striker']} is on {int(r['Total Non Striker Runs'])} off "
            f"{int(r['Non Striker Balls Faced'])}. {r['Bowler']} bowled the last delivery.")
    return ctx, batting


def row_questions(r, info, dls):
    """All question types that are well-defined for this row: {qtype: (question, gold, answer_kind)}."""
    inn, br_ = r["Innings"], int(r["Balls Remaining"])
    bb, runs = 120 - br_, int(r["Innings Runs"])
    batter, br, bf = r["Batter"], int(r["Total Batter Runs"]), int(r["Batter Balls Faced"])
    in_range = 0 <= br_ <= 120  # 8 source rows show -1 balls remaining (data anomaly): no overs-based questions
    _, batting = _ctx(r)
    out = {}
    if in_range and bb >= 1:
        out["run_rate"] = (f"What is {batting}'s current run rate in runs per over? Round to 2 decimal places.",
                           round(runs * 6 / bb, 2), "rate")
    if in_range and not dls and br_ > 0:  # rain-revised targets make chase arithmetic ambiguous
        if inn == "2" and float(r["Runs to Get"] or 0) > 0:
            out["chase_rate"] = (f"What run rate (runs per over) do {batting} need over the rest of their 20 overs to "
                                 f"reach the target? Round to 2 decimal places.",
                                 round((int(r["Target Score"]) - runs) * 6 / br_, 2), "rate")
        if inn == "1" and info.get("1", {}).get("full_20") and info["1"]["final_runs"] > runs:
            final = info["1"]["final_runs"]
            out["chase_rate"] = (f"{batting} finished their 20 overs on {final}. At what run rate (runs per over) did they "
                                 f"score over the rest of their innings from this point? Round to 2 decimal places.",
                                 round((final - runs) * 6 / br_, 2), "rate")
    if in_range and inn == "1" and 1 <= bb < 120:
        out["projection"] = (f"Take {batting}'s current run rate to be their total runs so far divided by the overs bowled "
                             f"so far. If they keep scoring at exactly that rate for the rest of their 20 overs, what total "
                             f"will they finish on? Round to the nearest whole run (halves round up).",
                             int(runs * 120 / bb + 0.5), "int")
    if br >= 1 and bf >= 1:
        m = (br // 50 + 1) * 50
        out["milestone"] = (f"{batter} wants to reach {m} runs. If {batter} keeps scoring at their own current strike "
                            f"rate (their runs per ball faced so far), how many more balls must {batter} face to reach "
                            f"{m}? Round up to a whole number of balls.", -(-(m - br) * bf // br), "int")
    if bf >= 1:
        out["strike_rate"] = (f"What is {batter}'s current strike rate (runs per 100 balls faced)? Round to 2 decimal places.",
                              round(br * 100 / bf, 2), "rate")
    if not out:  # innings opened with a wide / no-ball: nothing else is defined yet
        kind = "no-ball" if "noballs" in r["Extra Type"] else "wide"
        out["legal_balls"] = (f"The innings has opened with a {kind} from {r['Bowler']} that cost {r['Runs From Ball']} "
                              f"run(s). How many legal deliveries are still to be bowled in {batting}'s 20-over innings?",
                              br_, "int")
    return out


def _source(r, dls):
    return {"row_index": int(r.get("") or r.get("Unnamed: 0")), "match_id": int(r["Match ID"]), "date": r["Date"],
            "venue": r["Venue"], "bat_first": r["Bat First"], "bat_second": r["Bat Second"], "innings": int(r["Innings"]),
            "over": int(r["Over"]), "ball": int(r["Ball"]), "batter": r["Batter"], "bowler": r["Bowler"],
            "valid_ball": int(r["Valid Ball"]), "extra_type": r["Extra Type"], "wicket": int(r["Wicket"]),
            "innings_runs": int(r["Innings Runs"]), "innings_wickets": int(r["Innings Wickets"]),
            "balls_remaining": int(r["Balls Remaining"]),
            "target_score": int(r["Target Score"]) if r["Innings"] == "2" else None,
            "batter_runs": int(r["Total Batter Runs"]), "batter_balls_faced": int(r["Batter Balls Faced"]),
            "rain_revised_match": dls}


def build_full(csv_path, target_total=500_000, test_frac=0.1, seed=0):
    """One question for EVERY source row (least-used valid type, for balance), then a second, different question on
    random training rows until target_total. Test matches = the same held-out matches as build_splits()."""
    raw = load_matches(csv_path)
    clean = clean_matches(raw)
    mids = sorted(clean)
    random.Random(seed).shuffle(mids)  # identical test matches to build_splits(seed)
    test_m = set(mids[: max(1, int(len(mids) * test_frac))])
    rng = random.Random(seed + 1)
    rows = []
    for mid in sorted(raw):
        dls, info = mid not in clean, _innings_info(raw[mid])
        for r in raw[mid]:
            rows.append((r, info, dls, "test" if mid in test_m else "train"))
    order = list(range(len(rows)))
    rng.shuffle(order)
    counts, primary, avail = Counter(), {}, {}
    for i in order:  # greedy balance across types
        r, info, dls, _ = rows[i]
        avail[i] = row_questions(r, info, dls)
        qt = min(avail[i], key=lambda t: (counts[t], ALL_QTYPES.index(t)))
        primary[i] = qt
        counts[qt] += 1
    extra_n = max(0, target_total - len(rows))
    cands = [i for i in order if rows[i][3] == "train" and len(avail[i]) > 1]
    extra = {}
    for i in rng.sample(cands, min(extra_n, len(cands))):
        rest = [t for t in avail[i] if t != primary[i]]
        qt = min(rest, key=lambda t: (counts[t], ALL_QTYPES.index(t)))
        extra[i] = qt
        counts[qt] += 1

    def q(i, qt):
        r, info, dls, split = rows[i]
        ctx, _ = _ctx(r)
        text, gold, kind = avail[i][qt]
        src = _source(r, dls)
        return {"id": f"{r['Match ID']}-{r['Innings']}-{src['row_index']}-{qt}", "qtype": qt, "answer_kind": kind,
                "gold": gold, "split": split,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{ctx}\n\n{text}"}],
                "source": src}

    out = {"train": [], "test": []}
    for i in range(len(rows)):
        out[rows[i][3]].append(q(i, primary[i]))
        if i in extra:
            out[rows[i][3]].append(q(i, extra[i]))
    stats = {"source_dataset": "Valarmathy/CricketData", "source_rows": len(rows),
             "source_rows_train": sum(1 for x in rows if x[3] == "train"),
             "source_rows_test": sum(1 for x in rows if x[3] == "test"), "rows_with_two_questions": len(extra),
             "questions_train": len(out["train"]), "questions_test": len(out["test"]),
             "questions_total": len(out["train"]) + len(out["test"]), "questions_by_type": dict(counts),
             "matches_total": len(raw), "rain_revised_matches_in_train": len(raw) - len(clean), "test_matches": len(test_m)}
    return out["train"], out["test"], stats
