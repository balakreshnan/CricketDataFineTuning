#!/usr/bin/env python
"""Profile and validate the source dataset Valarmathy/CricketData (raw/ball_by_ball_it20.csv).

    python cricket/tools/validate_source.py <ball_by_ball_it20.csv> [--json out.json]

Standard library only. Prints a profile (matches, years, teams, venues, extras, dismissals, totals) and runs
the invariant checks documented in cricket/DATASET.md; exits non-zero if a hard check fails.
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict

EXPECTED = {"rows": 425_119, "columns": 35, "matches": 1_842,
            "first_date": "2005-02-17", "last_date": "2023-08-22"}
COLUMNS = ["", "Match ID", "Date", "Venue", "Bat First", "Bat Second", "Innings", "Over", "Ball", "Batter",
           "Non Striker", "Bowler", "Batter Runs", "Extra Runs", "Runs From Ball", "Ball Rebowled", "Extra Type",
           "Wicket", "Method", "Player Out", "Innings Runs", "Innings Wickets", "Target Score", "Runs to Get",
           "Balls Remaining", "Winner", "Chased Successfully", "Total Batter Runs", "Total Non Striker Runs",
           "Batter Balls Faced", "Non Striker Balls Faced", "Player Out Runs", "Player Out Balls Faced",
           "Bowler Runs Conceded", "Valid Ball"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    with open(a.csv, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        header = rd.fieldnames
        rows = list(rd)
    R = {"profile": {}, "checks": {}}
    P, C = R["profile"], R["checks"]

    def check(name, ok, detail, hard=True):
        C[name] = {"ok": bool(ok), "hard": hard, "detail": detail}

    # ------------------------------------------------------------ shape
    check("row_count", len(rows) == EXPECTED["rows"], f"{len(rows):,} rows (expected {EXPECTED['rows']:,})")
    check("columns", header == COLUMNS, f"{len(header)} columns; first column header is '{header[0]}' (pandas index)")
    nulls = {c: sum(1 for r in rows if r[c] in ("", "NaN", "nan", "None")) for c in header}
    P["null_counts"] = {c: n for c, n in nulls.items() if n}

    matches = defaultdict(list)
    for r in rows:
        matches[r["Match ID"]].append(r)
    dates = sorted({r["Date"] for r in rows})
    check("match_count", len(matches) == EXPECTED["matches"], f"{len(matches):,} matches")
    check("date_range", (dates[0], dates[-1]) == (EXPECTED["first_date"], EXPECTED["last_date"]),
          f"{dates[0]} -> {dates[-1]}")
    check("row_index_unique", len({r[''] for r in rows}) == len(rows),
          f"unnamed index column: {len({r[''] for r in rows}):,} distinct, min {min(int(r['']) for r in rows)}, "
          f"max {max(int(r['']) for r in rows)}")

    # ------------------------------------------------------------ per-match consistency
    const_cols = ["Date", "Venue", "Bat First", "Bat Second", "Winner", "Chased Successfully"]
    bad_const = Counter()
    for mid, rs in matches.items():
        for c in const_cols:
            if len({r[c] for r in rs}) > 1:
                bad_const[c] += 1
    check("match_constant_columns", not bad_const, f"columns not constant within a match: {dict(bad_const) or 'none'}")

    inn_sets = Counter(tuple(sorted({r["Innings"] for r in rs})) for rs in matches.values())
    P["innings_per_match"] = {"+".join(k): v for k, v in inn_sets.items()}

    # ------------------------------------------------------------ per-innings invariants
    viol = Counter()
    examples = defaultdict(list)
    totals = {}  # (mid, inn) -> (runs, wkts, legal balls, last balls remaining)
    for mid, rs in matches.items():
        by_inn = defaultdict(list)
        for r in rs:
            by_inn[r["Innings"]].append(r)
        for inn, ir in by_inn.items():
            cum_r = cum_w = legal = 0
            bat_runs, bat_balls = defaultdict(int), defaultdict(int)
            prev = (0, 0)
            for r in ir:
                br, er, rfb = int(r["Batter Runs"]), int(r["Extra Runs"]), int(r["Runs From Ball"])
                cum_r += rfb
                cum_w += int(r["Wicket"])
                legal += int(r["Valid Ball"])
                et = r["Extra Type"]
                wide = "wides" in et
                bat_runs[r["Batter"]] += br
                if not wide:
                    bat_balls[r["Batter"]] += 1
                cur = (int(r["Over"]), int(r["Ball"]))
                tests = {
                    "runs_from_ball_eq_batter_plus_extra": rfb == br + er,
                    "innings_runs_cumulative": int(r["Innings Runs"]) == cum_r,
                    "innings_wickets_cumulative": int(r["Innings Wickets"]) == cum_w,
                    "balls_remaining_eq_120_minus_legal": int(r["Balls Remaining"]) == 120 - legal,
                    "valid_ball_iff_not_wide_or_noball": int(r["Valid Ball"]) == (0 if (wide or "noballs" in et) else 1),
                    "wicket_has_method_and_player": (r["Wicket"] == "0") or (r["Method"] not in ("", "NaN") and r["Player Out"] not in ("", "NaN")),
                    "total_batter_runs_cumulative": int(r["Total Batter Runs"]) == bat_runs[r["Batter"]],
                    "batter_balls_faced_cumulative": int(r["Batter Balls Faced"]) == bat_balls[r["Batter"]],
                    "rows_ordered_by_over_ball": cur >= prev,
                    "over_in_1_20": 1 <= cur[0] <= 20,
                }
                prev = cur
                if inn == "2":
                    tests["runs_to_get_eq_target_minus_runs"] = abs(float(r["Runs to Get"] or "nan") - (int(r["Target Score"]) - cum_r)) < 1e-6
                for k, ok in tests.items():
                    if not ok:
                        viol[k] += 1
                        if len(examples[k]) < 3:
                            examples[k].append(f"row {r['']} (match {mid} inn {inn} {cur[0]}.{cur[1]})")
            totals[(mid, inn)] = (cum_r, cum_w, legal, int(ir[-1]["Balls Remaining"]))
    soft = {"total_batter_runs_cumulative", "batter_balls_faced_cumulative", "rows_ordered_by_over_ball",
            "balls_remaining_eq_120_minus_legal", "over_in_1_20"}
    for k in ["runs_from_ball_eq_batter_plus_extra", "innings_runs_cumulative", "innings_wickets_cumulative",
              "balls_remaining_eq_120_minus_legal", "valid_ball_iff_not_wide_or_noball", "wicket_has_method_and_player",
              "runs_to_get_eq_target_minus_runs", "total_batter_runs_cumulative", "batter_balls_faced_cumulative",
              "rows_ordered_by_over_ball", "over_in_1_20"]:
        check(k, viol[k] == 0, f"{viol[k]:,} violating rows" + (f"; e.g. {', '.join(examples[k])}" if viol[k] else ""),
              hard=k not in soft)

    # ------------------------------------------------------------ targets, results
    dls, chase_mismatch, no_inn2 = [], 0, 0
    for mid, rs in matches.items():
        t1 = totals.get((mid, "1"))
        tgt = int(rs[-1]["Target Score"])
        if t1 and tgt != t1[0] + 1:
            dls.append(mid)
        t2 = totals.get((mid, "2"))
        if not t2:
            no_inn2 += 1
            continue
        chased = rs[-1]["Chased Successfully"] == "1"
        if mid not in dls and chased != (t2[0] >= tgt):
            chase_mismatch += 1
    P["matches_target_not_first_innings_plus_1"] = len(dls)
    check("target_eq_first_innings_total_plus_1", len(dls) == 68,
          f"{len(dls)} matches with a revised (rain/DLS) target - expected 68; excluded from target-based questions",
          hard=False)
    check("chased_flag_matches_scores", chase_mismatch == 0,
          f"{chase_mismatch} non-revised matches where Chased Successfully disagrees with the scores", hard=False)
    P["matches_without_second_innings"] = no_inn2
    winner_vs_flag = sum(1 for rs in matches.values()
                         if (rs[-1]["Winner"] == rs[-1]["Bat Second"]) != (rs[-1]["Chased Successfully"] == "1"))
    P["matches_where_winner_disagrees_with_chased_flag"] = winner_vs_flag

    # ------------------------------------------------------------ profile
    per_year = Counter(rs[0]["Date"][:4] for rs in matches.values())
    P["matches_per_year"] = dict(sorted(per_year.items()))
    P["rows_per_year"] = dict(sorted(Counter(r["Date"][:4] for r in rows).items()))
    team_m = Counter()
    for rs in matches.values():
        team_m[rs[0]["Bat First"]] += 1
        team_m[rs[0]["Bat Second"]] += 1
    P["teams"] = len(team_m)
    P["top_teams_by_matches"] = team_m.most_common(25)
    venues = Counter(rs[0]["Venue"] for rs in matches.values())
    P["venues"] = len(venues)
    P["top_venues"] = venues.most_common(15)
    winners = Counter(rs[0]["Winner"] for rs in matches.values())
    P["winner_values"] = len(winners)
    P["top_winners"] = winners.most_common(15)
    P["chased_successfully"] = dict(Counter(rs[0]["Chased Successfully"] for rs in matches.values()))
    P["rows_per_innings"] = dict(Counter(r["Innings"] for r in rows))
    P["valid_ball"] = dict(Counter(r["Valid Ball"] for r in rows))
    P["extra_types"] = dict(Counter(r["Extra Type"] for r in rows).most_common())
    P["ball_rebowled"] = dict(Counter(r["Ball Rebowled"] for r in rows))
    P["wickets"] = sum(int(r["Wicket"]) for r in rows)
    P["dismissal_methods"] = dict(Counter(r["Method"] for r in rows if r["Wicket"] == "1").most_common())
    P["runs_from_ball"] = dict(sorted(Counter(int(r["Runs From Ball"]) for r in rows).items()))
    P["batter_runs"] = dict(sorted(Counter(int(r["Batter Runs"]) for r in rows).items()))
    P["balls_remaining_range"] = [min(int(r["Balls Remaining"]) for r in rows), max(int(r["Balls Remaining"]) for r in rows)]
    P["rows_balls_remaining_negative"] = sum(1 for r in rows if int(r["Balls Remaining"]) < 0)
    P["batters"] = len({r["Batter"] for r in rows})
    P["bowlers"] = len({r["Bowler"] for r in rows})
    bb = Counter(r["Batter"] for r in rows if "wides" not in r["Extra Type"])
    P["top_batters_by_balls_faced"] = bb.most_common(10)
    bw = Counter(r["Bowler"] for r in rows if r["Valid Ball"] == "1")
    P["top_bowlers_by_legal_balls"] = bw.most_common(10)
    t1 = [v[0] for (m, i), v in totals.items() if i == "1"]
    t2 = [v[0] for (m, i), v in totals.items() if i == "2"]
    full1 = sum(1 for (m, i), v in totals.items() if i == "1" and v[3] == 0)
    P["first_innings_total"] = {"min": min(t1), "median": st.median(t1), "mean": round(st.mean(t1), 1), "max": max(t1)}
    P["second_innings_total"] = {"min": min(t2), "median": st.median(t2), "mean": round(st.mean(t2), 1), "max": max(t2)}
    P["first_innings_full_20_overs"] = full1
    P["all_out_innings"] = sum(1 for v in totals.values() if v[1] == 10)
    hi = sorted(((v[0], m, i) for (m, i), v in totals.items()), reverse=True)[:5]
    lo = sorted(((v[0], m, i) for (m, i), v in totals.items() if v[1] == 10 or v[3] == 0))[:5]
    def desc(t):
        r = matches[t[1]][0]
        team = r["Bat First"] if t[2] == "1" else r["Bat Second"]
        opp = r["Bat Second"] if t[2] == "1" else r["Bat First"]
        return f"{team} {t[0]} v {opp}, {r['Venue']}, {r['Date']} (match {t[1]}, inn {t[2]})"
    P["highest_totals"] = [desc(t) for t in hi]
    P["lowest_completed_totals"] = [desc(t) for t in lo]
    rpm = [len(rs) for rs in matches.values()]
    P["rows_per_match"] = {"min": min(rpm), "median": st.median(rpm), "max": max(rpm)}
    P["sample_match"] = {k: matches[next(iter(matches))][0][k] for k in ("Match ID", "Date", "Venue", "Bat First", "Bat Second", "Winner")}

    # ------------------------------------------------------------ print
    print("== CHECKS")
    for k, v in C.items():
        flag = "PASS" if v["ok"] else ("FAIL" if v["hard"] else "NOTE")
        print(f"[{flag}] {k}: {v['detail']}")
    print("\n== PROFILE")
    for k, v in P.items():
        print(f"{k}: {v}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(R, f, indent=1)
    hard_fail = [k for k, v in C.items() if v["hard"] and not v["ok"]]
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
