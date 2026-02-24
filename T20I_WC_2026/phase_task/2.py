import os
import json
import pandas as pd
import numpy as np

# =====================================================
# 1️⃣ FLATTEN JSON FILES
# =====================================================

DATA_PATH = "."
rows = []

for file in os.listdir(DATA_PATH):
    if not file.endswith(".json"):
        continue

    match_id = file.replace(".json", "")

    with open(os.path.join(DATA_PATH, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    teams = info.get("teams", [])
    outcome = info.get("outcome", {})

    for inn_idx, inning in enumerate(data.get("innings", [])):
        batting_team = inning.get("team")
        bowling_team = [t for t in teams if t != batting_team]
        bowling_team = bowling_team[0] if bowling_team else None

        for over_data in inning.get("overs", []):
            over = over_data.get("over")

            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                runs = delivery.get("runs", {})
                extras = delivery.get("extras", {})
                wickets = delivery.get("wickets", [])

                rows.append({
                    "match_id": match_id,
                    "innings": inn_idx + 1,
                    "batting_team": batting_team,
                    "bowling_team": bowling_team,
                    "over": over,
                    "ball": ball_idx + 1,
                    "batter": delivery.get("batter"),
                    "bowler": delivery.get("bowler"),
                    "runs_batter": runs.get("batter", 0),
                    "runs_total": runs.get("total", 0),
                    "is_wide": 1 if "wides" in extras else 0,
                    "is_wicket": 1 if wickets else 0,
                    "match_winner": outcome.get("winner")
                })

df = pd.DataFrame(rows)

print("Matches:", df["match_id"].nunique())
print("Teams:", df["batting_team"].nunique())

# =====================================================
# 2️⃣ PHASE CLASSIFICATION
# =====================================================

def assign_phase(over):
    if over <= 5:
        return "Powerplay"
    elif over <= 9:
        return "EarlyMiddle"
    elif over <= 14:
        return "LateMiddle"
    else:
        return "Death"

df["phase"] = df["over"].apply(assign_phase)

# =====================================================
# 3️⃣ BATTING POSITION EXTRACTION
# =====================================================

df_sorted = df.sort_values(["match_id", "innings", "over", "ball"])

bat_pos_records = []

for (match, innings, team), group in df_sorted.groupby(
    ["match_id", "innings", "batting_team"]
):
    seen = {}
    pos = 1

    for _, row in group.iterrows():
        batter = row["batter"]
        if batter not in seen:
            seen[batter] = pos
            pos += 1

    for batter, position in seen.items():
        bat_pos_records.append({
            "match_id": match,
            "innings": innings,
            "batter": batter,
            "batting_position": position
        })

bat_pos_df = pd.DataFrame(bat_pos_records)

balls_faced = (
    df[df["is_wide"] == 0]
    .groupby(["match_id","batter"])
    .size()
    .reset_index(name="balls_faced")
)

bat_pos_df = bat_pos_df.merge(
    balls_faced,
    on=["match_id","batter"],
    how="left"
)

bat_pos_df = bat_pos_df[bat_pos_df["balls_faced"] >= 5]

avg_position = (
    bat_pos_df.groupby("batter")["batting_position"]
    .mean()
    .reset_index()
)

# =====================================================
# 4️⃣ BATTER ELIGIBILITY
# =====================================================

bat_matches = df.groupby("batter")["match_id"].nunique().reset_index(name="matches")
bat_balls = (
    df[df["is_wide"] == 0]
    .groupby("batter")
    .size()
    .reset_index(name="balls")
)

bat_elig = bat_matches.merge(bat_balls, on="batter")

bat_elig = bat_elig[
    (bat_elig["matches"] >= 3) &
    (bat_elig["balls"] >= 45)
]

eligible_batters = bat_elig["batter"]
df_bat = df[df["batter"].isin(eligible_batters)]

# =====================================================
# 5️⃣ BATTING SCORING
# =====================================================

# Phase baseline
phase_base = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby("phase")
    .agg(runs=("runs_batter","sum"),
         balls=("runs_batter","count"))
)

phase_base["sr_base"] = (
    phase_base["runs"] /
    phase_base["balls"] * 100
)

# Player phase impact
player_phase = (
    df_bat[df_bat["is_wide"] == 0]
    .groupby(["batter","phase"])
    .agg(runs=("runs_batter","sum"),
         balls=("runs_batter","count"))
    .reset_index()
)

player_phase = player_phase.merge(
    phase_base["sr_base"],
    on="phase"
)

player_phase["sr"] = player_phase["runs"] / player_phase["balls"] * 100

player_phase["phase_impact"] = (
    (player_phase["sr"] - player_phase["sr_base"])
    * player_phase["balls"]
)

phase_score = (
    player_phase.groupby("batter")["phase_impact"]
    .sum()
    .reset_index()
)

# Run share
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

player_runs = player_runs.merge(
    team_runs,
    on=["match_id","batting_team"]
)

player_runs["run_share"] = (
    player_runs["runs_batter"] /
    player_runs["team_runs"]
)

run_share = (
    player_runs.groupby("batter")["run_share"]
    .mean()
    .reset_index()
)

# Death SR
death_stats = (
    df_bat[(df_bat["phase"] == "Death") &
           (df_bat["is_wide"] == 0)]
    .groupby("batter")
    .agg(runs=("runs_batter","sum"),
         balls=("runs_batter","count"))
    .reset_index()
)

death_stats = death_stats[death_stats["balls"] >= 15]
death_stats["death_sr"] = (
    death_stats["runs"] /
    death_stats["balls"] * 100
)

# Combine
bat_final = phase_score.merge(run_share, on="batter", how="left")
bat_final = bat_final.merge(death_stats[["batter","death_sr"]],
                            on="batter", how="left")
bat_final = bat_final.merge(avg_position,
                            on="batter", how="left")
bat_final = bat_final.merge(bat_elig[["batter","balls"]],
                            on="batter", how="left")

bat_final = bat_final.fillna(0)

# Z-score normalization
for col in ["phase_impact","run_share","death_sr"]:
    std = bat_final[col].std()
    if std != 0:
        bat_final[col+"_z"] = (
            (bat_final[col] - bat_final[col].mean()) / std
        )
    else:
        bat_final[col+"_z"] = 0

# Updated weights
bat_final["raw_score"] = (
    0.60*bat_final["phase_impact_z"] +
    0.25*bat_final["run_share_z"] +
    0.15*bat_final["death_sr_z"]
)

bat_final["stability"] = (
    bat_final["balls"] /
    bat_final["balls"].max()
)

bat_final["final_score"] = (
    bat_final["raw_score"] *
    (0.9 + 0.1*bat_final["stability"])
)

# =====================================================
# 6️⃣ BOWLER ELIGIBILITY
# =====================================================

bowl_matches = df.groupby("bowler")["match_id"].nunique().reset_index(name="matches")
bowl_balls = (
    df[df["is_wide"] == 0]
    .groupby("bowler")
    .size()
    .reset_index(name="balls")
)

bowl_elig = bowl_matches.merge(bowl_balls, on="bowler")
bowl_elig["overs"] = bowl_elig["balls"] / 6

bowl_elig = bowl_elig[
    (bowl_elig["matches"] >= 3) &
    (bowl_elig["overs"] >= 8)
]

eligible_bowlers = bowl_elig["bowler"]
df_bowl = df[df["bowler"].isin(eligible_bowlers)]

# =====================================================
# 7️⃣ BOWLING SCORING
# =====================================================

bowl_phase_base = (
    df_bowl[df_bowl["is_wide"] == 0]
    .groupby("phase")
    .agg(runs=("runs_total","sum"),
         balls=("runs_total","count"))
)

bowl_phase_base["econ_base"] = (
    bowl_phase_base["runs"] /
    (bowl_phase_base["balls"] / 6)
)

player_bowl = (
    df_bowl[df_bowl["is_wide"] == 0]
    .groupby(["bowler","phase"])
    .agg(runs=("runs_total","sum"),
         balls=("runs_total","count"),
         wickets=("is_wicket","sum"))
    .reset_index()
)

player_bowl = player_bowl.merge(
    bowl_phase_base["econ_base"],
    on="phase"
)

player_bowl["econ"] = (
    player_bowl["runs"] /
    (player_bowl["balls"] / 6)
)

player_bowl["econ_impact"] = (
    (player_bowl["econ_base"] -
     player_bowl["econ"])
    * player_bowl["balls"]
)

bowl_phase_score = (
    player_bowl.groupby("bowler")["econ_impact"]
    .sum()
    .reset_index()
)

wicket_rate = (
    df_bowl[df_bowl["is_wide"] == 0]
    .groupby("bowler")["is_wicket"]
    .mean()
    .reset_index(name="wicket_rate")
)

bowl_final = bowl_phase_score.merge(
    wicket_rate,
    on="bowler",
    how="left"
).fillna(0)

for col in ["econ_impact","wicket_rate"]:
    std = bowl_final[col].std()
    if std != 0:
        bowl_final[col+"_z"] = (
            (bowl_final[col] - bowl_final[col].mean()) / std
        )
    else:
        bowl_final[col+"_z"] = 0

bowl_final["final_score"] = (
    0.65*bowl_final["econ_impact_z"] +
    0.35*bowl_final["wicket_rate_z"]
)

# =====================================================
# 8️⃣ BOWLING ROLE CLASSIFICATION
# =====================================================

phase_usage = (
    df_bowl[df_bowl["is_wide"] == 0]
    .groupby(["bowler","phase"])
    .size()
    .reset_index(name="balls")
)

total_balls = (
    df_bowl[df_bowl["is_wide"] == 0]
    .groupby("bowler")
    .size()
    .reset_index(name="total")
)

phase_usage = phase_usage.merge(total_balls, on="bowler")
phase_usage["ratio"] = phase_usage["balls"] / phase_usage["total"]

phase_pivot = phase_usage.pivot_table(
    index="bowler",
    columns="phase",
    values="ratio",
    fill_value=0
).reset_index()

bowl_final = bowl_final.merge(
    phase_pivot,
    on="bowler",
    how="left"
).fillna(0)

bowl_final["middle_ratio"] = (
    bowl_final.get("EarlyMiddle",0) +
    bowl_final.get("LateMiddle",0)
)

pp_bowlers = bowl_final[bowl_final["Powerplay"] >= 0.35]["bowler"]
death_bowlers = bowl_final[bowl_final["Death"] >= 0.30]["bowler"]
middle_bowlers = bowl_final[bowl_final["middle_ratio"] >= 0.40]["bowler"]

# =====================================================
# 9️⃣ STRUCTURAL XI
# =====================================================

# Batters
openers = bat_final[bat_final["batting_position"] <= 2]["batter"]
middle = bat_final[
    (bat_final["batting_position"] > 2) &
    (bat_final["batting_position"] <= 5)
]["batter"]
finishers = bat_final[
    bat_final["batting_position"] > 5
]["batter"]

top_openers = bat_final[bat_final["batter"].isin(openers)].sort_values("final_score", ascending=False).head(2)
top_middle = bat_final[bat_final["batter"].isin(middle)].sort_values("final_score", ascending=False).head(2)
top_finisher = bat_final[bat_final["batter"].isin(finishers)].sort_values("final_score", ascending=False).head(1)

already = pd.concat([top_openers, top_middle, top_finisher])["batter"]
top_flex = bat_final[~bat_final["batter"].isin(already)].sort_values("final_score", ascending=False).head(1)

bat_xi = pd.concat([top_openers, top_middle, top_finisher, top_flex])

# =====================================================
# STRUCTURAL BOWLING XI (FIXED)
# =====================================================

selected_bowlers = pd.DataFrame()

# 1️⃣ Powerplay – select top 2
pp_candidates = (
    bowl_final[bowl_final["bowler"].isin(pp_bowlers)]
    .sort_values("final_score", ascending=False)
)

top_pp = pp_candidates.head(2)
selected_bowlers = pd.concat([selected_bowlers, top_pp])

# 2️⃣ Death – select best NOT already selected
death_candidates = (
    bowl_final[
        (bowl_final["bowler"].isin(death_bowlers)) &
        (~bowl_final["bowler"].isin(selected_bowlers["bowler"]))
    ]
    .sort_values("final_score", ascending=False)
)

top_death = death_candidates.head(1)
selected_bowlers = pd.concat([selected_bowlers, top_death])

# 3️⃣ Middle overs – select best NOT already selected
middle_candidates = (
    bowl_final[
        (bowl_final["bowler"].isin(middle_bowlers)) &
        (~bowl_final["bowler"].isin(selected_bowlers["bowler"]))
    ]
    .sort_values("final_score", ascending=False)
)

top_middle = middle_candidates.head(1)
selected_bowlers = pd.concat([selected_bowlers, top_middle])

# 4️⃣ Flex – best remaining overall
flex_candidates = (
    bowl_final[
        ~bowl_final["bowler"].isin(selected_bowlers["bowler"])
    ]
    .sort_values("final_score", ascending=False)
)

top_flex = flex_candidates.head(1)
selected_bowlers = pd.concat([selected_bowlers, top_flex])

bowl_xi = selected_bowlers

print("\n=== FINAL XI BOWLERS ===")
print(bowl_xi[["bowler","final_score"]])