"""Verifiable Olympics reasoning questions built from pranjalvoid/olympics-dataset (athlete_events.csv:
one row per athlete per event per Games, 1896-2016; team medals appear once per team member).

Each question gives the model a small table taken from the data and asks for something it must compute from it;
the gold answer is computed exactly with pandas. Six question types:
  athlete_medals     total medals in an athlete's career table
  athlete_games      number of distinct Games an athlete competed in (rows are per event, shuffled)
  country_golds      golds a country won at one Games; team-member rows must be collapsed to one medal per event
  sport_top_country  country topping a sport's medal table at one Games (golds, then silvers, then bronzes)
  highest_bmi        which of 5-8 athletes has the highest BMI (weight / height^2)
  medalist_age       average age of an event's medalists, 1 decimal place
Splits are by entity (athlete, country+Games, sport+Games, event+Games), so no test table appears in training.
"""
import hashlib
import random

import pandas as pd

QTYPES = ("athlete_medals", "athlete_games", "country_golds", "sport_top_country", "highest_bmi", "medalist_age")
SYSTEM_PROMPT = ("You are a sports data analyst. Use only the table you are given. Work through the problem step by "
                 "step, then give the final answer on its own line as 'Final answer: <answer>'.")
MAX_TABLE_ROWS = 120   # country_golds / sport_top_country tables: the hard types; 60 gave only ~525 train questions each


def is_test(key, test_frac):
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 1000 < test_frac * 1000


def load(csv_path):
    df = pd.read_csv(csv_path).drop_duplicates()
    df["Medal"] = df["Medal"].fillna("")
    return df


def table(df, cols, rng):
    rows = df[cols].astype(object).where(df[cols].notna(), "n/a").values.tolist()
    rng.shuffle(rows)  # order must not give the answer away
    fmt = lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v if v != "" else "-")
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                      ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows])


def q(qid, qtype, kind, gold, ctx, question, source):
    return {"id": qid, "qtype": qtype, "answer_kind": kind, "gold": gold, "source": source,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"{ctx}\n\n{question}"}]}


def gen_athlete(df, rng):
    """athlete_medals + athlete_games, one candidate each per athlete with a 4-40 row career."""
    for aid, g in df.groupby("ID"):
        if not 4 <= len(g) <= 40:
            continue
        name = g["Name"].iloc[0]
        ctx = f"Olympic record of {name} ({g['Team'].iloc[0]}). One row per event entered; Medal '-' means no medal.\n\n"
        ctx += table(g, ["Games", "City", "Sport", "Event", "Medal"], rng)
        src = {"athlete_id": int(aid), "name": name, "rows": len(g)}
        n_med = int((g["Medal"] != "").sum())
        if n_med >= 1:
            yield aid, q(f"am-{aid}", "athlete_medals", "int", n_med, ctx,
                         f"How many Olympic medals (of any colour) did {name} win in total?", src)
        n_games = g["Games"].nunique()
        if n_games >= 2:
            yield aid, q(f"ag-{aid}", "athlete_games", "int", n_games, ctx,
                         f"In how many different Olympic Games did {name} compete?", src)


def gen_country_golds(df, rng):
    med = df[df["Medal"] != ""]
    for (noc, games), g in med.groupby(["NOC", "Games"]):
        golds = g.loc[g["Medal"] == "Gold", "Event"].nunique()
        if not (8 <= len(g) <= MAX_TABLE_ROWS and golds >= 1 and g["Event"].duplicated().any()):  # needs team rows to collapse
            continue
        team = g["Team"].mode().iloc[0]
        ctx = (f"All medal rows for {team} ({noc}) at the {games} Olympics. Medals in team events are listed once "
               f"per team member.\n\n" + table(g, ["Name", "Sport", "Event", "Medal"], rng))
        yield (noc, games), q(f"cg-{noc}-{games.replace(' ', '')}", "country_golds", "int", golds, ctx,
                              f"How many gold medals did {team} win at the {games} Olympics, counting a team "
                              f"event's gold as one medal?", {"noc": noc, "games": games, "rows": len(g)})


def gen_sport_top(df, rng):
    med = df[df["Medal"] != ""]
    for (sport, games), g in med.groupby(["Sport", "Games"]):
        if not 6 <= len(g) <= MAX_TABLE_ROWS:
            continue
        per_event = g.drop_duplicates(["Event", "NOC", "Medal"])  # one medal per team
        tally = per_event.pivot_table(index="NOC", columns="Medal", values="Event", aggfunc="count", fill_value=0)
        tally = tally.reindex(columns=["Gold", "Silver", "Bronze"], fill_value=0)
        ranked = sorted(tally.itertuples(), key=lambda r: (-r.Gold, -r.Silver, -r.Bronze))
        if len(ranked) < 3 or tuple(ranked[0])[1:] == tuple(ranked[1])[1:]:
            continue  # tie for first: ambiguous
        ctx = (f"All {sport} medal rows at the {games} Olympics. Medals in team events are listed once per team "
               f"member.\n\n" + table(g, ["Name", "NOC", "Event", "Medal"], rng))
        yield (sport, games), q(f"st-{sport.replace(' ', '')}-{games.replace(' ', '')}", "sport_top_country", "noc",
                                ranked[0].Index, ctx,
                                f"Rank the countries in the {sport} medal table for these Games by gold medals, then "
                                f"silver, then bronze, counting each team medal once. Which country (NOC code) "
                                f"finishes first?", {"sport": sport, "games": games, "rows": len(g)})


def gen_bmi(df, rng, n_sets):
    ath = df.dropna(subset=["Height", "Weight"]).drop_duplicates("ID")
    ath = ath[ath["Name"].str.len() < 40]
    ids = ath["ID"].tolist()
    for i in range(n_sets):
        k = rng.randint(5, 8)
        g = ath.sample(k, random_state=rng.randint(0, 2**31))
        bmi = g["Weight"] / (g["Height"] / 100) ** 2
        top2 = bmi.sort_values(ascending=False).values[:2]
        if top2[0] - top2[1] < 0.3:
            continue  # too close to call after rounding
        if g["Name"].duplicated().any():
            continue
        ctx = "Athletes with their recorded height (cm) and weight (kg).\n\n" + table(g, ["Name", "Sport", "Height", "Weight"], rng)
        winner = g.loc[bmi.idxmax(), "Name"]
        yield tuple(sorted(g["ID"])), q(f"bmi-{i}", "highest_bmi", "name", winner, ctx,
                                        "Body mass index is weight in kg divided by the square of height in metres. "
                                        "Which athlete has the highest BMI? Give their full name as written.",
                                        {"athlete_ids": [int(x) for x in g["ID"]]})


def gen_medalist_age(df, rng):
    med = df[(df["Medal"] != "")]
    for (event, games), g in med.groupby(["Event", "Games"]):
        if not 3 <= len(g) <= 30 or g["Age"].isna().any():
            continue
        avg = round(float(g["Age"].mean()) + 1e-9, 1)
        ctx = (f"Medallists in {event} at the {games} Olympics (team events list every team member).\n\n"
               + table(g, ["Name", "NOC", "Age", "Medal"], rng))
        yield (event, games), q(f"ma-{hashlib.md5(f'{event}|{games}'.encode()).hexdigest()[:10]}", "medalist_age", "float1", avg, ctx,
                                "What is the average age of all the medallists listed (every row counts as one "
                                "person)? Round to 1 decimal place.", {"event": event, "games": games, "rows": len(g)})


def build_splits(csv_path, n_train_per_type, n_test_per_type, test_frac=0.1, seed=0):
    rng = random.Random(seed)
    df = load(csv_path)
    gens = [gen_athlete(df, rng), gen_country_golds(df, rng), gen_sport_top(df, rng),
            gen_bmi(df, rng, 4 * (n_train_per_type + n_test_per_type)), gen_medalist_age(df, rng)]
    pools = {qt: {"train": [], "test": []} for qt in QTYPES}
    for gen in gens:
        for key, item in gen:
            pools[item["qtype"]]["test" if is_test(key, test_frac) else "train"].append(item)
    train, test, stats = [], [], {"source_rows": len(df)}
    for qt in QTYPES:
        for split, want, out in (("train", n_train_per_type, train), ("test", n_test_per_type, test)):
            pool = pools[qt][split]
            rng.shuffle(pool)
            out.extend(pool[:want])
            stats[f"{qt}_{split}_available"] = len(pool)
            stats[f"{qt}_{split}"] = min(want, len(pool))
    return train, test, stats


def parse_final(text):
    """The answer after the last 'Final answer:' (outside any reasoning block), or None."""
    text = text.split("</think>")[-1]
    idx = text.lower().rfind("final answer")
    if idx < 0:
        return None
    ans = text[idx + len("final answer"):].lstrip(" :*").split("\n")[0]
    return ans.strip().strip("*`$ .").replace("\\boxed{", "").rstrip("}").strip()


def is_correct(item, text):
    ans = parse_final(text)
    if ans is None:
        return False
    kind, gold = item["answer_kind"], item["gold"]
    if kind in ("int", "float1"):
        import re
        m = re.search(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
        if not m:
            return False
        v = float(m.group())
        return abs(v - gold) < (0.051 if kind == "float1" else 1e-9)
    norm = lambda s: " ".join(s.lower().replace("(", " ").replace(")", " ").split())
    if kind == "noc":
        return norm(gold) in norm(ans).split()
    return norm(ans) == norm(gold)
