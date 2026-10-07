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
| Teams / venues | **97 team names** (12 ICC full members, 83 associates - one listed under two names - and the ICC World XI; full list in §4.2), **183 venues** |
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

### 4.2 Teams - every team in the data (97 names)

The data covers **97 team names = 95 countries / territories + 1 renamed duplicate + 1 composite side**:

- **12 ICC Full Members** (all present): Afghanistan, Australia, Bangladesh, England, India, Ireland, New Zealand,
  Pakistan, South Africa, Sri Lanka, West Indies (a combined Caribbean team), Zimbabwe.
- **84 Associate-member team names**, i.e. **83 distinct associates**: *Swaziland* (2021) and *Eswatini* (2022) are the
  same country, renamed in 2018 - the source uses both names, so merge them if you aggregate by country.
- **ICC World XI** - a composite exhibition side (4 matches v Pakistan / West Indies, 2017-2018), not a country.
- Several associates are territories rather than sovereign states: Hong Kong, Cayman Islands, Bermuda, Gibraltar,
  Cook Islands, St Helena, and the Crown Dependencies Jersey, Guernsey and Isle of Man. England and Scotland play
  separately (no "United Kingdom" team).

By ICC region (team appearances = matches counted once per side):

| ICC region | teams | team appearances | teams |
|---|---|---|---|
| Africa | 21 | 664 | Botswana, Cameroon, Eswatini, Gambia, Ghana, Kenya, Lesotho, Malawi, Mali, Mozambique, Namibia, Nigeria, Rwanda, Seychelles, Sierra Leone, South Africa, St Helena, Swaziland, Tanzania, Uganda, Zimbabwe |
| Americas | 9 | 281 | Argentina, Bahamas, Belize, Bermuda, Canada, Cayman Islands, Panama, United States of America, West Indies |
| Asia | 21 | 1251 | Afghanistan, Bahrain, Bangladesh, Bhutan, China, Hong Kong, India, Iran, Kuwait, Malaysia, Maldives, Myanmar, Nepal, Oman, Pakistan, Qatar, Saudi Arabia, Singapore, Sri Lanka, Thailand, United Arab Emirates |
| East Asia-Pacific | 11 | 452 | Australia, Cook Islands, Fiji, Indonesia, Japan, New Zealand, Papua New Guinea, Philippines, Samoa, South Korea, Vanuatu |
| Europe | 34 | 1032 | Austria, Belgium, Bulgaria, Croatia, Cyprus, Czech Republic, Denmark, England, Estonia, Finland, France, Germany, Gibraltar, Greece, Guernsey, Hungary, Ireland, Isle of Man, Israel, Italy, Jersey, Luxembourg, Malta, Netherlands, Norway, Portugal, Romania, Scotland, Serbia, Slovenia, Spain, Sweden, Switzerland, Turkey |

All teams, by matches played (`wins` from the `Winner` column; `deliveries batted` = rows where the team batted;
first / last = calendar years of their first and last match in the data):

| # | team | ICC status | ICC region | matches | wins | win % | deliveries batted | first year | last year |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Pakistan | Full member | Asia | 208 | 130 | 62% | 24,976 | 2006 | 2023 |
| 2 | India | Full member | Asia | 192 | 127 | 66% | 22,611 | 2006 | 2023 |
| 3 | New Zealand | Full member | East Asia-Pacific | 179 | 100 | 56% | 20,729 | 2005 | 2023 |
| 4 | Sri Lanka | Full member | Asia | 168 | 75 | 45% | 19,789 | 2006 | 2023 |
| 5 | West Indies | Full member | Americas | 168 | 74 | 44% | 19,656 | 2007 | 2023 |
| 6 | Australia | Full member | East Asia-Pacific | 164 | 91 | 55% | 18,596 | 2005 | 2022 |
| 7 | England | Full member | Europe | 162 | 88 | 54% | 19,088 | 2005 | 2023 |
| 8 | South Africa | Full member | Africa | 160 | 92 | 58% | 18,491 | 2005 | 2023 |
| 9 | Bangladesh | Full member | Asia | 142 | 53 | 37% | 16,888 | 2007 | 2023 |
| 10 | Ireland | Full member | Europe | 134 | 58 | 43% | 14,976 | 2008 | 2023 |
| 11 | Zimbabwe | Full member | Africa | 115 | 36 | 31% | 13,774 | 2007 | 2023 |
| 12 | Afghanistan | Full member | Asia | 109 | 69 | 63% | 12,945 | 2010 | 2023 |
| 13 | Netherlands | Associate | Europe | 84 | 45 | 54% | 9,677 | 2008 | 2022 |
| 14 | Scotland | Associate | Europe | 72 | 34 | 47% | 8,205 | 2007 | 2023 |
| 15 | United Arab Emirates | Associate | Asia | 70 | 32 | 46% | 8,310 | 2014 | 2023 |
| 16 | Kenya | Associate | Africa | 60 | 30 | 50% | 6,578 | 2007 | 2023 |
| 17 | Uganda | Associate | Africa | 58 | 44 | 76% | 6,431 | 2019 | 2023 |
| 18 | Hong Kong | Associate | Asia | 56 | 24 | 43% | 6,432 | 2014 | 2023 |
| 19 | Nepal | Associate | Asia | 51 | 32 | 63% | 5,756 | 2014 | 2022 |
| 20 | Malaysia | Associate | Asia | 49 | 29 | 59% | 5,274 | 2019 | 2023 |
| 21 | Oman | Associate | Asia | 49 | 21 | 43% | 5,422 | 2015 | 2022 |
| 22 | Germany | Associate | Europe | 46 | 27 | 59% | 5,034 | 2019 | 2023 |
| 23 | Malta | Associate | Europe | 43 | 22 | 51% | 5,127 | 2021 | 2023 |
| 24 | Namibia | Associate | Africa | 41 | 27 | 66% | 4,709 | 2019 | 2022 |
| 25 | Papua New Guinea | Associate | East Asia-Pacific | 41 | 21 | 51% | 4,533 | 2015 | 2023 |
| 26 | Canada | Associate | Americas | 35 | 20 | 57% | 4,064 | 2010 | 2022 |
| 27 | Rwanda | Associate | Africa | 33 | 11 | 33% | 3,397 | 2021 | 2023 |
| 28 | Singapore | Associate | Asia | 33 | 11 | 33% | 3,916 | 2019 | 2022 |
| 29 | Romania | Associate | Europe | 32 | 20 | 62% | 3,653 | 2020 | 2023 |
| 30 | Tanzania | Associate | Africa | 32 | 23 | 72% | 3,438 | 2021 | 2023 |
| 31 | Denmark | Associate | Europe | 30 | 12 | 40% | 3,538 | 2019 | 2023 |
| 32 | Nigeria | Associate | Africa | 30 | 12 | 40% | 3,376 | 2019 | 2022 |
| 33 | Bahrain | Associate | Asia | 29 | 14 | 48% | 3,353 | 2020 | 2023 |
| 34 | Gibraltar | Associate | Europe | 29 | 4 | 14% | 3,704 | 2019 | 2023 |
| 35 | Bulgaria | Associate | Europe | 28 | 11 | 39% | 3,317 | 2020 | 2023 |
| 36 | Jersey | Associate | Europe | 27 | 17 | 63% | 3,179 | 2019 | 2023 |
| 37 | Botswana | Associate | Africa | 26 | 8 | 31% | 2,874 | 2019 | 2023 |
| 38 | Luxembourg | Associate | Europe | 26 | 10 | 38% | 3,215 | 2020 | 2023 |
| 39 | Spain | Associate | Europe | 25 | 18 | 72% | 2,551 | 2019 | 2023 |
| 40 | Italy | Associate | Europe | 23 | 13 | 57% | 2,634 | 2019 | 2023 |
| 41 | Austria | Associate | Europe | 22 | 12 | 55% | 2,509 | 2021 | 2023 |
| 42 | Czech Republic | Associate | Europe | 22 | 12 | 55% | 2,721 | 2020 | 2023 |
| 43 | Ghana | Associate | Africa | 22 | 9 | 41% | 2,554 | 2019 | 2022 |
| 44 | Vanuatu | Associate | East Asia-Pacific | 20 | 10 | 50% | 2,225 | 2019 | 2023 |
| 45 | Bermuda | Associate | Americas | 19 | 11 | 58% | 2,130 | 2008 | 2023 |
| 46 | Kuwait | Associate | Asia | 19 | 11 | 58% | 2,105 | 2019 | 2023 |
| 47 | Norway | Associate | Europe | 19 | 6 | 32% | 2,158 | 2019 | 2023 |
| 48 | Guernsey | Associate | Europe | 18 | 11 | 61% | 2,023 | 2019 | 2023 |
| 49 | Portugal | Associate | Europe | 18 | 14 | 78% | 2,059 | 2019 | 2023 |
| 50 | Serbia | Associate | Europe | 18 | 7 | 39% | 2,183 | 2021 | 2023 |
| 51 | France | Associate | Europe | 17 | 8 | 47% | 2,031 | 2021 | 2023 |
| 52 | Thailand | Associate | Asia | 17 | 4 | 24% | 1,817 | 2019 | 2023 |
| 53 | United States of America | Associate | Americas | 17 | 9 | 53% | 1,819 | 2019 | 2022 |
| 54 | Hungary | Associate | Europe | 16 | 7 | 44% | 1,964 | 2021 | 2023 |
| 55 | Sweden | Associate | Europe | 16 | 7 | 44% | 1,774 | 2021 | 2023 |
| 56 | Belgium | Associate | Europe | 15 | 10 | 67% | 1,741 | 2020 | 2023 |
| 57 | Finland | Associate | Europe | 15 | 7 | 47% | 1,746 | 2021 | 2023 |
| 58 | Sierra Leone | Associate | Africa | 15 | 7 | 47% | 1,490 | 2021 | 2022 |
| 59 | Isle of Man | Associate | Europe | 14 | 8 | 57% | 1,425 | 2020 | 2023 |
| 60 | Maldives | Associate | Asia | 14 | 3 | 21% | 1,639 | 2019 | 2022 |
| 61 | Qatar | Associate | Asia | 14 | 7 | 50% | 1,591 | 2019 | 2022 |
| 62 | Switzerland | Associate | Europe | 14 | 10 | 71% | 1,480 | 2021 | 2023 |
| 63 | Mozambique | Associate | Africa | 13 | 5 | 38% | 1,539 | 2021 | 2022 |
| 64 | Philippines | Associate | East Asia-Pacific | 13 | 2 | 15% | 1,466 | 2019 | 2023 |
| 65 | Bhutan | Associate | Asia | 12 | 6 | 50% | 1,427 | 2019 | 2023 |
| 66 | Japan | Associate | East Asia-Pacific | 11 | 6 | 55% | 1,253 | 2022 | 2023 |
| 67 | Argentina | Associate | Americas | 10 | 2 | 20% | 1,209 | 2021 | 2023 |
| 68 | Cameroon | Associate | Africa | 10 | 0 | 0% | 1,045 | 2021 | 2022 |
| 69 | Cayman Islands | Associate | Americas | 10 | 3 | 30% | 1,241 | 2019 | 2023 |
| 70 | Croatia | Associate | Europe | 10 | 2 | 20% | 1,142 | 2022 | 2023 |
| 71 | Cyprus | Associate | Europe | 10 | 5 | 50% | 1,214 | 2021 | 2022 |
| 72 | Lesotho | Associate | Africa | 10 | 1 | 10% | 1,005 | 2021 | 2022 |
| 73 | Saudi Arabia | Associate | Asia | 10 | 3 | 30% | 1,157 | 2020 | 2022 |
| 74 | Bahamas | Associate | Americas | 9 | 3 | 33% | 1,108 | 2021 | 2023 |
| 75 | Estonia | Associate | Europe | 8 | 0 | 0% | 972 | 2021 | 2022 |
| 76 | Malawi | Associate | Africa | 8 | 5 | 62% | 833 | 2021 | 2022 |
| 77 | Panama | Associate | Americas | 7 | 2 | 29% | 854 | 2021 | 2023 |
| 78 | Seychelles | Associate | Africa | 7 | 2 | 29% | 775 | 2021 | 2022 |
| 79 | Turkey | Associate | Europe | 7 | 1 | 14% | 760 | 2022 | 2023 |
| 80 | Belize | Associate | Americas | 6 | 1 | 17% | 729 | 2021 | 2021 |
| 81 | Gambia | Associate | Africa | 6 | 1 | 17% | 657 | 2022 | 2022 |
| 82 | Indonesia | Associate | East Asia-Pacific | 6 | 4 | 67% | 639 | 2022 | 2022 |
| 83 | Swaziland | Associate | Africa | 6 | 1 | 17% | 665 | 2021 | 2021 |
| 84 | Cook Islands | Associate | East Asia-Pacific | 5 | 3 | 60% | 561 | 2022 | 2022 |
| 85 | Eswatini | Associate | Africa | 5 | 1 | 20% | 516 | 2022 | 2022 |
| 86 | Fiji | Associate | East Asia-Pacific | 5 | 3 | 60% | 616 | 2022 | 2022 |
| 87 | Greece | Associate | Europe | 5 | 1 | 20% | 550 | 2021 | 2022 |
| 88 | Samoa | Associate | East Asia-Pacific | 5 | 0 | 0% | 629 | 2022 | 2022 |
| 89 | China | Associate | Asia | 4 | 1 | 25% | 328 | 2023 | 2023 |
| 90 | ICC World XI | Composite (not a country) | - | 4 | 1 | 25% | 475 | 2017 | 2018 |
| 91 | Mali | Associate | Africa | 4 | 0 | 0% | 340 | 2022 | 2022 |
| 92 | Myanmar | Associate | Asia | 4 | 0 | 0% | 456 | 2023 | 2023 |
| 93 | Slovenia | Associate | Europe | 4 | 0 | 0% | 458 | 2022 | 2022 |
| 94 | Israel | Associate | Europe | 3 | 1 | 33% | 378 | 2022 | 2022 |
| 95 | South Korea | Associate | East Asia-Pacific | 3 | 0 | 0% | 333 | 2022 | 2022 |
| 96 | St Helena | Associate | Africa | 3 | 1 | 33% | 258 | 2022 | 2022 |
| 97 | Iran | Associate | Asia | 1 | 0 | 0% | 131 | 2020 | 2020 |

Teams with fewer than ~10 matches give very few rows (e.g. Mali, Cameroon, Iran, Saudi Arabia, St Helena); per-team
statistics for them are noisy. The full-member sides account for 1,901 of the 3,684 team appearances.

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
