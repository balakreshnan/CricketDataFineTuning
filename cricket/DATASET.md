# Source dataset: `Valarmathy/CricketData` (T20 Internationals, ball by ball)

Documentation of the raw data used by every cricket experiment in this folder: what it contains, which matches,
teams and years it covers, how each column behaves (verified, including quirks), and how to validate it and run a
smoke test before using it. All numbers below were measured on the file itself with
[`tools/validate_source.py`](tools/validate_source.py) on 2026-10-07; the full machine-readable profile is
[`source_profile.json`](source_profile.json).

---

## 1. At a glance

| | |
|---|---|
| Hugging Face repo | [`Valarmathy/CricketData`](https://huggingface.co/datasets/Valarmathy/CricketData) (dataset) |
| Revision used | `a2518e9db3bd6e745dabc645a423eed6f053207f` (last modified 2023-10-02) |
| File | `raw/ball_by_ball_it20.csv` - **72,501,997 bytes (69 MiB)**, UTF-8 CSV with header |
| License | **CC0-1.0** (public domain dedication) |
| Granularity | **one row per delivery** (legal balls *and* extras such as wides / no-balls) |
| Rows / columns | **425,119 rows**, **35 columns** (first column = unnamed pandas index 0 … 425,118) |
| Format | **Men's T20 Internationals**, 20 overs per side |
| Matches | **1,842**, every one with both innings present |
| Dates | **2005-02-17 → 2023-08-22** (first row = the first ever men's T20I: match 211048, Australia (batting first) v New Zealand, Eden Park, won by Australia) |
| Teams / venues | **97 teams**, **183 venues** |
| People | **2,899 batters**, **2,156 bowlers** |
| Deliveries | 408,034 legal balls + 17,085 extras that were re-bowled (wides, no-balls) |
| Wickets | 23,659 dismissals (incl. 35 retirements) |
| Results | 928 chases won, 914 defended; 68 matches with a rain-revised (DLS) target; 1 tie |
| Match IDs | ESPNcricinfo match ids (e.g. `1339605` = West Indies v South Africa, Centurion, 26 Mar 2023) |

The dataset card on Hugging Face only lists task categories; there is no upstream description of the columns, so
§3 documents them from verified behaviour.

## 2. Getting the file

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("Valarmathy/CricketData", "raw/ball_by_ball_it20.csv", repo_type="dataset",
                       revision="a2518e9db3bd6e745dabc645a423eed6f053207f")
```

On Hecate it is already at
`/lustre/fsw/general_sa/bbalakreshna/clustercodes/cricket/data/raw/raw/ball_by_ball_it20.csv` (downloaded by
`prepare.py`, which re-counts the rows and refuses to continue unless there are exactly 425,119).

Read it with any CSV reader; the first header cell is empty (the pandas index the author saved). With pandas:
`pd.read_csv(path, index_col=0)`. `Extra Type` is a Python-list literal stored as text (`"['wides']"`).

## 3. Columns (35)

Rows are ordered by match, innings, over, ball. "Cumulative" means *after this delivery*, within the innings.

| # | column | type | meaning (verified) | nulls | values / notes |
|---|---|---|---|---|---|
| 0 | *(unnamed)* | int | row index 0 … 425,118, unique; we keep it as `source.row_index` | 0 | |
| 1 | `Match ID` | int | ESPNcricinfo match id | 0 | 1,842 distinct |
| 2 | `Date` | str | match date `YYYY-MM-DD` | 0 | 2005-02-17 … 2023-08-22 |
| 3 | `Venue` | str | ground name | 0 | 183 distinct; constant within a match |
| 4 | `Bat First` | str | team batting in innings 1 | 0 | constant within a match |
| 5 | `Bat Second` | str | team batting in innings 2 | 0 | constant within a match |
| 6 | `Innings` | int | 1 or 2 | 0 | 224,815 / 200,304 rows |
| 7 | `Over` | int | **1-based** over number (1 = first over) | 0 | 1 … 20 |
| 8 | `Ball` | int | ball number within the over; extras repeat the number of the legal ball they precede | 0 | 1 … 6 normally; 0 (69 rows) and 7 (158 rows) occur - see §6 |
| 9 | `Batter` | str | striker for this delivery | 0 | 2,899 distinct |
| 10 | `Non Striker` | str | non-striker | 0 | |
| 11 | `Bowler` | str | bowler | 0 | 2,156 distinct |
| 12 | `Batter Runs` | int | runs off the bat on this delivery | 0 | 0-7 (0: 184,757, 1: 149,518, 4: 41,055, 6: 17,212) |
| 13 | `Extra Runs` | int | extras on this delivery | 0 | |
| 14 | `Runs From Ball` | int | total runs from this delivery = `Batter Runs + Extra Runs` (holds on every row) | 0 | 0-8 |
| 15 | `Ball Rebowled` | 0/1 | 1 if the delivery must be re-bowled (wide / no-ball) | 0 | 1 on 17,085 rows = exactly the rows with `Valid Ball = 0` |
| 16 | `Extra Type` | str (list) | kinds of extras | 0 | `[]` 399,765; `['wides']` 15,246; `['legbyes']` 6,631; `['noballs']` 1,775; `['byes']` 1,631; combos 64; `['penalty']` 7 |
| 17 | `Wicket` | 0/1 | a dismissal (or retirement) on this delivery | 0 | 23,659 ones |
| 18 | `Method` | str | dismissal type | 401,460 (no wicket) | caught 13,378; bowled 4,917; run out 1,997; lbw 1,824; stumped 796; caught and bowled 685; retired hurt 27; hit wicket 24; retired not out 5; retired out 3; obstructing the field 2; hit the ball twice 1 |
| 19 | `Player Out` | str | dismissed batter | 401,460 | |
| 20 | `Innings Runs` | int | team score, cumulative (= running sum of `Runs From Ball`; holds on every row) | 0 | |
| 21 | `Innings Wickets` | int | wickets down, cumulative (= running sum of `Wicket`; holds on every row) | 0 | 0-10 |
| 22 | `Target Score` | int | target for the side batting second = first-innings total + 1 (revised target in 68 rain-affected matches); present on **both** innings' rows | 0 | |
| 23 | `Runs to Get` | float | innings 2 only: `Target Score − Innings Runs` (holds on every innings-2 row) | 224,815 (all innings-1 rows) | |
| 24 | `Balls Remaining` | int | legal balls left in the 20-over innings = 120 − legal balls so far (holds on every row) | 0 | −1 … 120 (8 rows at −1, §6) |
| 25 | `Winner` | str | winning team | 0 | 89 distinct; constant within a match |
| 26 | `Chased Successfully` | 0/1 | 1 if the side batting second won | 0 | per match: 928 × 1, 914 × 0 |
| 27 | `Total Batter Runs` | int | striker's score after this delivery; **on a delivery where the striker is out it is 0** (the incoming batter's tally) - the dismissed batter's score is in `Player Out Runs` | 0 | exact running total on all 401,460 non-wicket rows |
| 28 | `Total Non Striker Runs` | int | non-striker's score after this delivery | 0 | |
| 29 | `Batter Balls Faced` | int | striker's balls faced *as recorded by the source* | 0 | equals a count of non-wide deliveries on 79% of rows; on ~21% it is 1-2 higher, mostly because some **wides are counted as faced** - use as given, do not recompute |
| 30 | `Non Striker Balls Faced` | int | non-striker's balls faced, same convention | 0 | |
| 31 | `Player Out Runs` | float | score of the dismissed batter | 401,460 | |
| 32 | `Player Out Balls Faced` | float | balls faced by the dismissed batter | 401,460 | |
| 33 | `Bowler Runs Conceded` | int | runs charged to the bowler **on this delivery** (not cumulative) | 0 | |
| 34 | `Valid Ball` | 0/1 | 1 = legal delivery (counts toward the over) | 0 | 0 exactly when `Extra Type` contains wides or no-balls |

All null cells are structural (no wicket → no `Method`/`Player Out…`; innings 1 → no `Runs to Get`); there are no
other missing values.

## 4. Coverage

### 4.1 By year

| year | matches | deliveries | | year | matches | deliveries |
|---|---|---|---|---|---|---|
| 2005 | 3 | 693 | | 2015 | 52 | 12,415 |
| 2006 | 7 | 1,617 | | 2016 | 95 | 22,127 |
| 2007 | 35 | 8,236 | | 2017 | 63 | 14,416 |
| 2008 | 14 | 3,100 | | 2018 | 77 | 18,012 |
| 2009 | 47 | 11,018 | | **2019** | **205** | 47,365 |
| 2010 | 59 | 13,634 | | 2020 | 69 | 15,964 |
| 2011 | 21 | 5,036 | | **2021** | **297** | 68,426 |
| 2012 | 71 | 16,370 | | **2022** | **446** | 102,581 |
| 2013 | 50 | 11,867 | | 2023 (to 22 Aug) | 184 | 41,667 |
| 2014 | 47 | 10,575 | | **total** | **1,842** | **425,119** |

Coverage jumps from 2019, when the ICC granted T20I status to all member nations: **1,132 of the 1,842 matches
(61%) are from 2019-2023**, many between associate nations. World Cup years show as peaks (2007, 2009, 2010, 2012,
2014, 2016, 2021, 2022). 2023 is partial (data ends 22 Aug 2023).

### 4.2 Teams (97)

Top teams by matches played: Pakistan 208, India 192, New Zealand 179, West Indies 168, Sri Lanka 168, Australia 164,
England 162, South Africa 160, Bangladesh 142, Ireland 134, Zimbabwe 115, Afghanistan 109, Netherlands 84,
Scotland 72, UAE 70, Kenya 60, Uganda 58, Hong Kong 56, Nepal 51, Malaysia 49, Oman 49, Germany 46, Malta 43,
Papua New Guinea 41, Namibia 41 - the remaining 72 teams are associate members (e.g. Rwanda, Spain, Isle of Man,
China, Thailand, Belgium, Bulgaria, Serbia, Bhutan, Maldives, Lesotho, Botswana).

Most wins: Pakistan 130, India 127, New Zealand 100, South Africa 92, Australia 91, England 88, Sri Lanka 75,
West Indies 74, Afghanistan 69, Ireland 58.

### 4.3 Venues (183)

Most used: Dubai International Cricket Stadium 88, Al Amerat (Oman) 64, Shere Bangla National Stadium 53, Gahanga
(Rwanda) 49, Marsa Sports Club (Malta) 48, Sheikh Zayed Stadium 44, Harare Sports Club 38, R Premadasa Stadium 33,
Integrated Polytechnic Regional Centre (Rwanda) 33, Sharjah 31, Desert Springs (Spain) 29, ICC Academy 28, Kerava
(Finland) 27, New Wanderers 24, Gymkhana Club Ground 23.

### 4.4 Players

Most balls faced: V Kohli 2,719, Babar Azam 2,676, RG Sharma 2,596, MJ Guptill 2,441, PR Stirling 2,323, Mohammad
Rizwan 2,141, AJ Finch 2,075, DA Warner 1,952, Mohammad Hafeez 1,952, KS Williamson 1,858.
Most legal balls bowled: Shakib Al Hasan 2,373, TG Southee 2,207, Shahid Afridi 2,039, IS Sodhi 1,945, Mohammad Nabi
1,930, AU Rashid 1,928, Shadab Khan 1,903, Rashid Khan 1,874, Mustafizur Rahman 1,816, MJ Santner 1,712.
Names follow ESPNcricinfo short form (initials + surname for most players).

## 5. Match and innings statistics

| | value |
|---|---|
| deliveries per match | min 53, median 242, max 275 |
| first-innings total | min 10, median 152, mean 150.6, max 278 |
| second-innings total | min 13, median 132, mean 129.8, max 259 |
| first innings lasting the full 20 overs | 1,540 of 1,842 |
| all-out innings | 639 |
| highest totals | Afghanistan 278/3 v Ireland (Dehradun, 2019-02-23); Australia 263 v Sri Lanka (2016-09-06); Sri Lanka 260 v Kenya (2007-09-14); India 260 v Sri Lanka (2017-12-22); South Africa 259/4 v West Indies (2023-03-26, chasing 259) |
| lowest completed totals | Isle of Man 10 v Spain (2023-02-26); China 23 v Malaysia (2023-07-26); Lesotho 26 v Uganda (2021-10-19); China 26 v Thailand (2023-07-27); Thailand 30 v Malaysia (2022-07-04) |
| runs per delivery | 0: 160,394; 1: 170,365; 2: 32,230; 3: 2,392; 4: 41,659; 5: 860; 6: 17,107; 7: 111; 8: 1 |

These records match published T20I history (Afghanistan's 278/3, Isle of Man's 10 all out, South Africa's record
chase of 259), which is a useful external sanity check.

## 6. Data quality: verified invariants and known quirks

**Hold on every row** (checked by `validate_source.py`):
`Runs From Ball = Batter Runs + Extra Runs`; `Innings Runs` and `Innings Wickets` are exact running sums;
`Balls Remaining = 120 − legal balls so far`; `Valid Ball = 0` ⇔ wide or no-ball ⇔ `Ball Rebowled = 1`; every
wicket row has `Method` and `Player Out`; innings-2 `Runs to Get = Target − Innings Runs`; rows are in
(over, ball) order; `Over` ∈ 1…20; `Date`, `Venue`, teams, `Winner` and `Chased Successfully` are constant within a
match; every match has exactly two innings; the index column is unique.

**Quirks to know** (none breaks the arithmetic used by the cricket questions):

| quirk | rows / matches | impact and handling |
|---|---|---|
| Rain-revised targets: `Target Score ≠ first-innings total + 1` | 68 matches (e.g. 1273148, 682961, 1119501, 1289052, 412686) | the stored target is the revised one and overs may be reduced, so chase arithmetic over "the rest of 20 overs" is ambiguous; these matches get no `chase_rate` questions and are kept out of the test split |
| `Balls Remaining = −1` (a 7th legal ball in the 20th over) | 8 rows (matches 1321465, 306987, 1321263, 1344794, 1286675, 1384593, 412702, 1279383; all `Over 20, Ball 7`) | an over miscount in the source; these rows get no overs-based questions |
| `Ball` = 0 or 7 | 69 / 158 rows | extras recorded before ball 1 or after ball 6 of an over (plus the 8 rows above); harmless - we derive balls bowled from `Balls Remaining` |
| `Total Batter Runs` resets to 0 on the striker's dismissal | 22,729 rows | the dismissed batter's score is in `Player Out Runs`; exact running total on all other rows |
| `Batter Balls Faced` is not a strict running count | ~21% of rows differ by 1-2 (mostly +1, wides counted) | use as recorded; question text shows the row's own figures and answers are computed from those, so questions stay self-consistent |
| `Chased Successfully` / `Winner` contradict the scores | 1 match: 1268191, Belgium v Malta, 2021-07-10 (Malta 125/10 chasing 129, recorded as Malta winning) | source error; only affects outcome labels, which no cricket question uses |
| Tie | 1 match: 1321304, Norway 77 v Austria 77, Kerava, 2022-07-31 | `Winner` = Austria and `Chased Successfully` = 1 although the scores are level - the tie was decided outside the 40 overs (not in the data), so the flag means "side batting second won", not "target reached" |
| Score notation `runs/wickets` | - | when runs ≤ wickets ("1/3") the Australian `wickets/runs` reading is also plausible; question text spells such scores out |

## 7. How the cricket pipeline uses it

| | |
|---|---|
| split | by `Match ID`, seed 0: 177 held-out matches → test (41,298 rows), 1,597 + the 68 rain-revised matches → train (383,821 rows) |
| Part I dataset | 1,500 + 200 sampled deliveries × 3 questions → 4,500 train / 600 test (`prepare.py`, default mode) |
| Part II dataset | every one of the 425,119 rows → ≥ 1 question, 500,000 in total (`prepare.py --mode full`) |
| fields carried into every dataset row | `source.row_index` (column 0), match id, date, venue, teams, innings, over, ball, batter, bowler, extras, wicket, score, balls remaining, target, batter runs / balls faced, rain-revised flag |
| answers | always computed from the row's own numbers (see README §2 and §12.1) |

## 8. Validation and smoke tests

### 8.1 One-minute smoke test

```bash
python cricket/tools/validate_source.py /path/to/ball_by_ball_it20.csv --json /tmp/profile.json; echo "exit $?"
```

Expected: exit code **0**, every hard check `[PASS]`, and exactly these four `[NOTE]` lines (known quirks):

```
[NOTE] total_batter_runs_cumulative: 19,906 violating rows ...      (resets on dismissals - expected)
[NOTE] batter_balls_faced_cumulative: 106,585 violating rows ...    (source counting convention - expected)
[PASS] target_eq_first_innings_total_plus_1: 68 matches ...         (expected 68)
[NOTE] chased_flag_matches_scores: 1 non-revised matches ...        (Belgium v Malta - expected)
```

Any change in these counts means the source file changed: compare `source_profile.json` with the new profile.

### 8.2 Checklist of expected values

| check | expected |
|---|---|
| file size | 72,501,997 bytes |
| rows (excluding header) | 425,119 |
| columns | 35, first header empty |
| distinct `Match ID` | 1,842 |
| date range | 2005-02-17 … 2023-08-22 |
| matches per year | as in §4.1 (2022: 446 is the largest) |
| teams / venues / batters / bowlers | 97 / 183 / 2,899 / 2,156 |
| rows per innings | 224,815 (inn 1) / 200,304 (inn 2) |
| `Valid Ball` | 408,034 ones, 17,085 zeros |
| wickets | 23,659 |
| rain-revised targets | 68 matches |
| `Balls Remaining` range | −1 … 120 (8 rows at −1) |
| null cells | only in `Method`, `Player Out`, `Player Out Runs`, `Player Out Balls Faced` (401,460 each) and `Runs to Get` (224,815) |

### 8.3 Spot checks against real scorecards

| match id | expectation in the data | real result |
|---|---|---|
| 1168113 | innings 1 ends 278/3, Afghanistan v Ireland, Rajiv Gandhi Intl Stadium, 2019-02-23 | Afghanistan 278/3 (then a T20I record) |
| 1354803 | innings 1 ends 10/10 with 68 balls remaining, Isle of Man v Spain, 2023-02-26 | Isle of Man 10 all out in 8.4 overs |
| 1339605 | 248 deliveries; innings 1 ends 258, innings 2 ends 259/4 with 7 balls remaining | West Indies 258/5, South Africa 259/4 in 18.5 overs (record chase) |

```python
import csv
rows = [r for r in csv.DictReader(open(path, newline="", encoding="utf-8")) if r["Match ID"] == "1339605"]
print(len(rows), rows[-1]["Innings Runs"], rows[-1]["Innings Wickets"], rows[-1]["Balls Remaining"])   # 248 259 4 7
```

### 8.4 Before building a new question type

1. Prefer columns whose invariants hold on every row (§6): `Innings Runs`, `Innings Wickets`, `Balls Remaining`,
   `Target Score`, `Runs to Get`, `Runs From Ball`, `Valid Ball`.
2. Treat `Batter Balls Faced`, `Total Batter Runs` on wicket rows, `Ball`, and outcome labels with the caveats above.
3. Exclude rain-revised matches from anything that depends on the target or on 20 full overs.
4. Run a stratified pilot (`submit.sh pilot-full`) and look for truncation or a pass@k below 100% - in this project
   those exposed wording ambiguities, not model failures.
