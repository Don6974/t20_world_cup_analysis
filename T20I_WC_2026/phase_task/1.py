import os
import json
import pandas as pd

# Folder containing your json files
DATA_PATH = "."   # change this

all_rows = []

for file in os.listdir(DATA_PATH):
    if not file.endswith(".json"):
        continue

    match_id = file.replace(".json", "")
    file_path = os.path.join(DATA_PATH, file)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    season = info.get("season")
    venue = info.get("venue")

    teams = info.get("teams", [])
    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")

    outcome = info.get("outcome", {})
    match_winner = outcome.get("winner")

    innings_data = data.get("innings", [])

    for inning_index, inning in enumerate(innings_data):
        batting_team = inning.get("team")
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else None

        overs = inning.get("overs", [])

        for over_data in overs:
            over_number = over_data.get("over")

            deliveries = over_data.get("deliveries", [])

            for ball_index, delivery in enumerate(deliveries):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")

                runs = delivery.get("runs", {})
                runs_batter = runs.get("batter", 0)
                runs_extras = runs.get("extras", 0)
                runs_total = runs.get("total", 0)

                extras_detail = delivery.get("extras", {})
                is_wide = 1 if "wides" in extras_detail else 0
                is_legbye = 1 if "legbyes" in extras_detail else 0

                wickets = delivery.get("wickets", [])
                is_wicket = 1 if wickets else 0

                dismissal_kind = None
                player_out = None

                if wickets:
                    dismissal_kind = wickets[0].get("kind")
                    player_out = wickets[0].get("player_out")

                row = {
                    "match_id": match_id,
                    "season": season,
                    "venue": venue,
                    "innings": inning_index + 1,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over_number,
                    "ball": ball_index + 1,
                    "batter": batter,
                    "bowler": bowler,
                    "runs_batter": runs_batter,
                    "runs_extras": runs_extras,
                    "runs_total": runs_total,
                    "is_wide": is_wide,
                    "is_legbye": is_legbye,
                    "is_wicket": is_wicket,
                    "dismissal_kind": dismissal_kind,
                    "player_out": player_out,
                    "toss_winner": toss_winner,
                    "toss_decision": toss_decision,
                    "match_winner": match_winner
                }

                all_rows.append(row)

# Convert to dataframe
df = pd.DataFrame(all_rows)

print("Total Matches:", df["match_id"].nunique())
print("Total Teams:", df["batting_team"].nunique())
print(df.groupby("batting_team")["match_id"].nunique())

# Save
df.to_csv("t20_worldcup_flattened.csv", index=False)

print("Done. Total deliveries:", len(df))


##Phase Classification
def assign_phase(over):
    if over <= 5:          # 0–5 → Overs 1–6
        return "Powerplay"
    elif over <= 9:        # 6–9 → Overs 7–10
        return "EarlyMiddle"
    elif over <= 14:       # 10–14 → Overs 11–15
        return "LateMiddle"
    else:                  # 15–19 → Overs 16–20
        return "Death"

df["phase"] = df["over"].apply(assign_phase)

df["batting_team"].unique()

df.groupby("match_id")[["batting_team"]].nunique()

# -----------------------------
# BATTER ELIGIBILITY
# -----------------------------
bat_matches = df.groupby("batter")["match_id"].nunique().reset_index(name="matches")

bat_balls = (
    df[df["is_wide"] == 0]
    .groupby("batter")
    .size()
    .reset_index(name="balls")
)

bat_elig = bat_matches.merge(bat_balls, on="batter")

bat_eligible = bat_elig[
    (bat_elig["matches"] >= 3) &
    (bat_elig["balls"] >= 45)
]["batter"]

df_bat = df[df["batter"].isin(bat_eligible)]

# -----------------------------
# BOWLER ELIGIBILITY
# -----------------------------
bowl_matches = df.groupby("bowler")["match_id"].nunique().reset_index(name="matches")

bowl_balls = (
    df[df["is_wide"] == 0]
    .groupby("bowler")
    .size()
    .reset_index(name="balls")
)

bowl_elig = bowl_matches.merge(bowl_balls, on="bowler")
bowl_elig["overs"] = bowl_elig["balls"] / 6

bowl_eligible = bowl_elig[
    (bowl_elig["matches"] >= 3) &
    (bowl_elig["overs"] >= 8)
]["bowler"]

df_bowl = df[df["bowler"].isin(bowl_eligible)]

# Batting phase baseline
phase_baseline = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby("phase")
    .agg(
        runs=("runs_batter","sum"),
        balls=("runs_batter","count")
    )
)

phase_baseline["tournament_sr"] = (
    phase_baseline["runs"] /
    phase_baseline["balls"] * 100
)
print("\nTournament Phase Baseline:")
print(phase_baseline)

#batting phase impact
player_phase = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby(["batter","phase"])
    .agg(
        runs=("runs_batter","sum"),
        balls=("runs_batter","count")
    )
    .reset_index()
)

player_phase = player_phase.merge(
    phase_baseline["tournament_sr"],
    on="phase"
)

player_phase["sr"] = player_phase["runs"] / player_phase["balls"] * 100

player_phase["phase_impact"] = (
    (player_phase["sr"] - player_phase["tournament_sr"])
    * player_phase["balls"]
)

bat_phase_score = (
    player_phase.groupby("batter")["phase_impact"]
    .sum()
    .reset_index()
)
print("\nTop 15 Batters by Phase Impact:")
print(
    bat_phase_score
    .sort_values("phase_impact", ascending=False)
    .head(15)
)
print("\nSample Player Phase Breakdown:")
print(player_phase.sort_values("phase_impact", ascending=False).head(20))

#STEP_3: Team Contribution Score (20%)
team_runs = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby(["match_id","batting_team"])["runs_batter"]
    .sum()
    .reset_index(name="team_runs")
)

player_runs = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby(["batter","match_id","batting_team"])["runs_batter"]
    .sum()
    .reset_index()
)

player_runs = player_runs.merge(team_runs, on=["match_id","batting_team"])

player_runs["run_share"] = player_runs["runs_batter"] / player_runs["team_runs"]

bat_contribution = (
    player_runs.groupby("batter")["run_share"]
    .mean()
    .reset_index()
)

#STEP 4 — Death / Pressure Score (15%)
death_stats = (
    df_bat[(df_bat["phase"] == "Death") & (df_bat["is_wide"] == 0)]
    .groupby("batter")
    .agg(
        runs=("runs_batter","sum"),
        balls=("runs_batter","count")
    )
    .reset_index()
)

death_stats["death_sr"] = death_stats["runs"] / death_stats["balls"] * 100
print("\n====================")
print("TOP 15 BY TEAM RUN SHARE")
print("====================")
print(
    bat_contribution
    .sort_values("run_share", ascending=False)
    .head(15)
)

#STEP 5 — Win Influence (10%)
win_df = df_bat[df_bat["batting_team"] == df_bat["match_winner"]]

win_runs = (
    win_df[win_df["is_wide"] == 0]
    .groupby("batter")["runs_batter"]
    .sum()
    .reset_index(name="runs_in_wins")
)

total_runs = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby("batter")["runs_batter"]
    .sum()
    .reset_index(name="total_runs")
)

win_impact = win_runs.merge(total_runs, on="batter")
win_impact["win_ratio"] = win_impact["runs_in_wins"] / win_impact["total_runs"]
print("\n====================")
print("TOP 15 DEATH STRIKE RATES")
print("====================")
print(
    death_stats
    .sort_values("death_sr", ascending=False)
    .head(15)
)

#STEP 6 — Combine Batting Scores
bat_final = bat_phase_score.merge(bat_contribution, on="batter", how="left")
bat_final = bat_final.merge(death_stats[["batter","death_sr"]], on="batter", how="left")
bat_final = bat_final.merge(win_impact[["batter","win_ratio"]], on="batter", how="left")

bat_final = bat_final.fillna(0)

# Normalise using z-score
for col in ["phase_impact","run_share","death_sr","win_ratio"]:
    bat_final[col+"_z"] = (
        (bat_final[col] - bat_final[col].mean())
        / bat_final[col].std()
    )

bat_final["final_score"] = (
    0.45*bat_final["phase_impact_z"] +
    0.20*bat_final["run_share_z"] +
    0.15*bat_final["death_sr_z"] +
    0.10*bat_final["win_ratio_z"]
)

print("\n====================")
print("TOP 15 WIN IMPACT RATIO")
print("====================")
print(
    win_impact
    .sort_values("win_ratio", ascending=False)
    .head(15)
)
print("\n====================")
print("FINAL BATTING RANKINGS")
print("====================")
print(
    bat_final
    .sort_values("final_score", ascending=False)
    .head(20)
)

#STEP 7 — Role Classification
pp_usage = (
    df_bat[df_bat["phase"] == "Powerplay"]
    .groupby("batter")
    .size()
    .reset_index(name="pp_balls")
)

total_balls = (
    df_bat.groupby("batter")
    .size()
    .reset_index(name="total_balls")
)

usage = pp_usage.merge(total_balls, on="batter")
usage["pp_ratio"] = usage["pp_balls"] / usage["total_balls"]

openers = usage[usage["pp_ratio"] >= 0.4]["batter"]
print("\n====================")
print("OPENER USAGE RATIOS")
print("====================")
print(
    usage
    .sort_values("pp_ratio", ascending=False)
    .head(15)
)

#STEP 8 — Select Final XI (Batters First)
# Top 2 openers
top_openers = (
    bat_final[bat_final["batter"].isin(openers)]
    .sort_values("final_score", ascending=False)
    .head(2)
)

remaining_batters = bat_final[
    ~bat_final["batter"].isin(top_openers["batter"])
].sort_values("final_score", ascending=False)

top_other_batters = remaining_batters.head(4)

bat_xi = pd.concat([top_openers, top_other_batters])