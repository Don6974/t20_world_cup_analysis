import os
import json
import numpy as np
import pandas as pd

# =====================================================
# CONFIG
# =====================================================

MIN_BALLS_BAT = 40
MIN_OVERS_BOWL = 8
TOP_BOWLERS_SHORTLIST = 20

SPINNER_LIST = [
    "Shadab Khan", "Abrar Ahmed", "Mohammad Nawaz",
    "MRJ Watt", "MA Leask", "Harmeet Singh",
    "CV Varun", "AR Patel"
]

def stage(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def min_max(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

def phase(over):
    if over <= 5:
        return "PP"
    elif over <= 9:
        return "Early_Middle"
    elif over <= 14:
        return "Late_Middle"
    else:
        return "Death"

# =====================================================
# STAGE 1: LOAD DATA
# =====================================================

stage("STAGE 1: LOADING DATA")

rows = []
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for filename in os.listdir(base_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(base_dir, filename)
        with open(file_path) as f:
            data = json.load(f)

        for innings in data["innings"]:
            wickets_fallen = 0
            for over_data in innings["overs"]:
                for delivery in over_data["deliveries"]:
                    extras = delivery.get("extras", {})
                    legal = not ("wides" in extras or "noballs" in extras)

                    if delivery.get("wickets"):
                        wickets_fallen += 1

                    rows.append({
                        "match": filename,
                        "over": over_data["over"],
                        "phase": phase(over_data["over"]),
                        "batter": delivery["batter"],
                        "bowler": delivery["bowler"],
                        "batter_runs": delivery["runs"]["batter"],
                        "total_runs": delivery["runs"]["total"],
                        "byes": extras.get("byes", 0),
                        "legbyes": extras.get("legbyes", 0),
                        "legal": legal,
                        "is_wicket": len(delivery.get("wickets", [])) > 0,
                        "wickets_at_ball": wickets_fallen
                    })

df = pd.DataFrame(rows)

df["is_dot"] = df["batter_runs"] == 0
df["is_boundary"] = df["batter_runs"].isin([4,6])
df["is_rotation"] = df["batter_runs"].isin([1,2,3])

print("Matches:", df["match"].nunique())
print("Batters:", df["batter"].nunique())
print("Bowlers:", df["bowler"].nunique())

# =====================================================
# STAGE 2: BATTING BASE
# =====================================================

stage("STAGE 2: BATTING BASE")

bat = df[df["legal"]].groupby("batter").agg(    matches=("match","nunique"),
    runs=("batter_runs","sum"),
    balls=("legal","sum"),
    wickets=("is_wicket","sum")
)

print("Batters before filter:", len(bat))
bat = bat[bat["balls"] >= MIN_BALLS_BAT]
print("After balls filter:", len(bat))

bat["SR"] = bat["runs"] / bat["balls"] * 100
bat["dismissal_rate"] = bat["wickets"] / bat["balls"]

# =====================================================
# STAGE 3: PHASE METRICS
# =====================================================

stage("STAGE 3: PHASE METRICS")

phase_grp = df[df["legal"]].groupby(["batter","phase"]).agg(    balls=("legal","sum"),
    runs=("batter_runs","sum"),
    dots=("is_dot","sum"),
    boundaries=("is_boundary","sum"),
    rotation=("is_rotation","sum"),
    wickets=("is_wicket","sum")
).reset_index()

phase_grp["SR"] = phase_grp["runs"] / phase_grp["balls"] * 100
phase_grp["dot_pct"] = phase_grp["dots"] / phase_grp["balls"]
phase_grp["boundary_pct"] = phase_grp["boundaries"] / phase_grp["balls"]
phase_grp["rotation_pct"] = phase_grp["rotation"] / phase_grp["balls"]

pivot = phase_grp.pivot(index="batter", columns="phase")
pivot.columns = [f"{m}_{p}" for m,p in pivot.columns]
bat = bat.join(pivot).fillna(0)



# =====================================================
# STAGE 4: PRESSURE + CONSISTENCY
# =====================================================

stage("STAGE 4: PRESSURE + CONSISTENCY")

collapse_df = df[df["wickets_at_ball"] >= 2]
collapse = collapse_df.groupby("batter").agg(
    runs=("batter_runs","sum"),
    balls=("legal","sum")
)
collapse["collapse_SR"] = collapse["runs"]/collapse["balls"]*100
bat = bat.join(collapse["collapse_SR"])
bat["collapse_SR"] = bat["collapse_SR"].fillna(0)

innings_sr = df.groupby(["match","batter"]).agg(
    runs=("batter_runs","sum"),
    balls=("legal","sum")
).reset_index()

innings_sr["SR"] = innings_sr["runs"]/innings_sr["balls"]*100
std_dev = innings_sr.groupby("batter")["SR"].std()

bat["consistency"] = 1/(1+std_dev)
bat["consistency"] = bat["consistency"].fillna(0)

# =====================================================
# STAGE 5: BATTING ROLE SCORING
# =====================================================

stage("STAGE 5: ROLE SCORING")

bat["SR_norm"] = min_max(bat["SR"])
bat["Cons_norm"] = min_max(bat["consistency"])
bat["Pressure_norm"] = min_max(bat["collapse_SR"])

def col(name):
    return bat[name] if name in bat.columns else pd.Series(0,index=bat.index)

PP_SR = min_max(col("SR_PP"))
M1_rot = min_max(col("rotation_pct_Early_Middle"))
M2_SR = min_max(col("SR_Late_Middle"))
Death_SR = min_max(col("SR_Death"))
Death_bound = min_max(col("boundary_pct_Death"))

# Corrected anchor model (penalizes high SR)
bat["Anchor_Score"] = (
    0.35*M1_rot +
    0.30*min_max(1-col("dot_pct_Early_Middle")) +
    0.20*bat["Cons_norm"] +
    0.15*min_max(1-bat["SR_norm"])
)

bat["Opener_Score"] = 0.40*PP_SR + 0.30*bat["SR_norm"] + 0.20*bat["Cons_norm"] + 0.10*bat["Pressure_norm"]

bat["Middle_Score"] = 0.40*M2_SR + 0.30*min_max(col("boundary_pct_Late_Middle")) + 0.20*bat["Cons_norm"] + 0.10*bat["Pressure_norm"]

bat["Finisher_Score"] = 0.45*Death_SR + 0.30*Death_bound + 0.15*bat["SR_norm"] + 0.10*bat["Pressure_norm"]

print("Top Openers:", bat.sort_values("Opener_Score",ascending=False).head(5).index.tolist())
print("Top Middles:", bat.sort_values("Middle_Score",ascending=False).head(5).index.tolist())
print("Top Anchor:", bat.sort_values("Anchor_Score",ascending=False).head(5).index.tolist())
print("Top Finisher:", bat.sort_values("Finisher_Score",ascending=False).head(5).index.tolist())

# =====================================================
# STAGE 6: BOWLING
# =====================================================

stage("STAGE 6: BOWLING METRICS")

bowl = df.groupby("bowler").agg(
    runs=("total_runs","sum"),
    balls=("legal","sum"),
    wickets=("is_wicket","sum"),
    dots=("is_dot","sum")
)

bowl["overs"] = bowl["balls"]/6
bowl = bowl[bowl["overs"] >= MIN_OVERS_BOWL]

bowl["economy"] = bowl["runs"]/bowl["overs"]
bowl["wkt_rate"] = bowl["wickets"]/bowl["overs"]
bowl["dot_pct"] = bowl["dots"]/bowl["balls"]

bowl["Impact_Index"] = (
    0.40*min_max(bowl["wkt_rate"]) +
    0.35*min_max(1/bowl["economy"]) +
    0.25*min_max(bowl["dot_pct"])
)

bowl = bowl.sort_values("Impact_Index",ascending=False).head(TOP_BOWLERS_SHORTLIST)
bowl["bowling_type"] = np.where(bowl.index.isin(SPINNER_LIST),"Spinner","Pace")

print("Eligible Bowlers:", len(bowl))
print("Top Bowlers:", bowl.index.tolist())
print("Spinners:", bowl[bowl["bowling_type"]=="Spinner"].index.tolist())
print("Pacers:", bowl[bowl["bowling_type"]=="Pace"].index.tolist())

# =====================================================
# STAGE 6B: ALL-ROUNDER CLASSIFICATION
# =====================================================

stage("STAGE 6B: ALL-ROUNDER CLASSIFICATION")

eligible_batters = set(bat.index)
eligible_bowlers = set(bowl.index)

all_rounders = list(eligible_batters.intersection(eligible_bowlers))

print("Dual Eligible Players:", all_rounders)
print("Count:", len(all_rounders))

if all_rounders:

    ar_df = pd.DataFrame(index=all_rounders)

    # Normalized batting & bowling impact
    ar_df["bat_impact"] = bat.loc[all_rounders]["SR_norm"]
    ar_df["bowl_impact"] = bowl.loc[all_rounders]["Impact_Index"]

    # Batting All-Rounder (bat dominant)
    ar_df["Bat_AR_Index"] = (
        0.65 * ar_df["bat_impact"] +
        0.35 * ar_df["bowl_impact"]
    )

    # Bowling All-Rounder (bowl dominant)
    ar_df["Bowl_AR_Index"] = (
        0.65 * ar_df["bowl_impact"] +
        0.35 * ar_df["bat_impact"]
    )

    bat_all_rounders = ar_df.sort_values("Bat_AR_Index", ascending=False)
    bowl_all_rounders = ar_df.sort_values("Bowl_AR_Index", ascending=False)

    print("\nTop Batting All-Rounders:")
    print(bat_all_rounders.head())

    print("\nTop Bowling All-Rounders:")
    print(bowl_all_rounders.head())

# =====================================================
# STAGE 7: FINAL XI
# =====================================================

stage("STAGE 7: FINAL XI")

# --- LOCK BOWLING CORE ---
spinner = bowl[bowl["bowling_type"]=="Spinner"].head(1)
pacers = bowl[bowl["bowling_type"]=="Pace"].head(2)

# --- LOCK 2 ALL-ROUNDERS ---
ar_blocks = []

if len(all_rounders) >= 1:
    bat_ar_player = bat_all_rounders.index[0]
    ar_blocks.append(bat.loc[[bat_ar_player]])

if len(all_rounders) >= 2:
    # If same player ranked first in both lists, take second in bowl list
    if bowl_all_rounders.index[0] == bat_ar_player:
        bowl_ar_player = bowl_all_rounders.index[1]
    else:
        bowl_ar_player = bowl_all_rounders.index[0]

    ar_blocks.append(bat.loc[[bowl_ar_player]])


# --- CORE BATTING ROLES ---

bat_pool = bat.copy()

# Openers
openers = bat_pool.sort_values("Opener_Score", ascending=False).head(2)
bat_pool = bat_pool.drop(openers.index)

# Anchor
anchor = bat_pool.sort_values("Anchor_Score", ascending=False).head(1)
bat_pool = bat_pool.drop(anchor.index)

# Middle
middle = bat_pool.sort_values("Middle_Score", ascending=False).head(2)
bat_pool = bat_pool.drop(middle.index)

# Finisher
finisher = bat_pool.sort_values("Finisher_Score", ascending=False).head(1)
bat_pool = bat_pool.drop(finisher.index)

# --- COMBINE ---
selection_blocks = [
    openers,
    anchor,
    middle,
    finisher,
    *ar_blocks,
    spinner,
    pacers,
]

final_xi = pd.concat(selection_blocks).drop_duplicates()

print("FINAL XI:")
print(final_xi.index.tolist())
print("Total Selected:", len(final_xi))

role_map = {}

# All-Rounders
if len(ar_blocks) >= 1:
    role_map[bat_ar_player] = "Batting All-Rounder"

if len(ar_blocks) >= 2:
    role_map[bowl_ar_player] = "Bowling All-Rounder"

# Batting roles
for p in openers.index:
    role_map[p] = "Opener"

for p in anchor.index:
    role_map[p] = "Anchor"

for p in middle.index:
    role_map[p] = "Middle Order"

for p in finisher.index:
    role_map[p] = "Finisher"

# Bowling roles
for p in spinner.index:
    role_map[p] = "Spinner"

for p in pacers.index:
    role_map[p] = "Pacer"

print("\nFINAL XI WITH ROLES:")
for player in final_xi.index:
    role = role_map.get(player, "Specialist Batter")
    print(f"{player} → {role}")

print("Total Selected:", len(final_xi))