import os
import json
import numpy as np
import pandas as pd

# =====================================================
# CONFIG
# =====================================================

MIN_BALLS_BAT = 80
MIN_OVERS_BOWL = 10
MIN_DEATH_OVERS = 4

SPINNER_LIST = [
    "Shadab Khan", "Abrar Ahmed", "Mohammad Nawaz",
    "MRJ Watt", "MA Leask", "Harmeet Singh",
    "CV Varun", "AR Patel"
]

# =====================================================
# UTILITIES
# =====================================================

def stage(title):
    print("\n" + "="*80)
    print(title)
    print("="*80)

def z_score(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0, index=series.index)
    return (series - mean) / std

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

bat = df[df["legal"]].groupby("batter").agg(
    matches=("match","nunique"),
    runs=("batter_runs","sum"),
    balls=("legal","sum"),
   # wickets=("is_wicket","sum")
)
print(bat)
bat = bat[bat["balls"] >= MIN_BALLS_BAT]
bat = bat[bat["matches"] >= 3] #at least 3 matches to be considered 
print("bat after filtering:", bat)

bat["SR"] = bat["runs"] / bat["balls"] * 100

# =====================================================
# STAGE 3: PHASE METRICS
# =====================================================

stage("STAGE 3: PHASE METRICS")

phase_grp = df[df["legal"]].groupby(["batter","phase"]).agg(
    balls=("legal","sum"),
    runs=("batter_runs","sum"),
    dots=("is_dot","sum"),
    boundaries=("is_boundary","sum"),
    rotation=("is_rotation","sum")
).reset_index()

phase_grp["SR"] = phase_grp["runs"] / phase_grp["balls"] * 100
phase_grp["dot_pct"] = phase_grp["dots"] / phase_grp["balls"]
phase_grp["boundary_pct"] = phase_grp["boundaries"] / phase_grp["balls"]
phase_grp["rotation_pct"] = phase_grp["rotation"] / phase_grp["balls"]

pivot = phase_grp.pivot(index="batter", columns="phase")
pivot.columns = [f"{m}_{p}" for m,p in pivot.columns]

bat = bat.join(pivot).fillna(0)

# =====================================================
# STAGE 4: CONSISTENCY
# =====================================================

stage("STAGE 4: CONSISTENCY")

innings_sr = df.groupby(["match","batter"]).agg(
    runs=("batter_runs","sum"),
    balls=("legal","sum")
).reset_index()

innings_sr["SR"] = innings_sr["runs"]/innings_sr["balls"]*100
std_dev = innings_sr.groupby("batter")["SR"].std()

bat["consistency"] = 1/(1+std_dev)
bat["consistency"] = bat["consistency"].fillna(0)

# =====================================================
# STAGE 5: ROLE SCORING (Z-SCORE)
# =====================================================

stage("STAGE 5: ROLE SCORING")

bat["SR_z"] = z_score(bat["SR"])
bat["Cons_z"] = z_score(bat["consistency"])

def col(name):
    x = bat[name] if name in bat.columns else pd.Series(0,index=bat.index)
    print(x)
    return x

print(col("SR_PP").describe())
PP_SR = z_score(col("SR_PP"))
print("PP SR Z-Score Stats:")
print(bat.columns)
print(PP_SR.describe())
print("Top 5 PP SR Z-Scores:")
print(PP_SR.sort_values(ascending=False).head())
M1_rot = z_score(col("rotation_pct_Early_Middle"))
M2_SR = z_score(col("SR_Late_Middle"))
Death_SR = z_score(col("SR_Death"))
Death_bound = z_score(col("boundary_pct_Death"))

bat["Opener_Score"] = 0.5*PP_SR + 0.3*bat["SR_z"] + 0.2*bat["Cons_z"]
bat["Anchor_Score"] = 0.4*M1_rot + 0.3*z_score(1-col("dot_pct_Early_Middle")) + 0.3*bat["Cons_z"]
bat["Middle_Score"] = 0.5*M2_SR + 0.3*z_score(col("boundary_pct_Late_Middle")) + 0.2*bat["Cons_z"]
bat["Finisher_Score"] = 0.5*Death_SR + 0.3*Death_bound + 0.2*bat["SR_z"]

# =====================================================
# STAGE 6: BOWLING + DEATH
# =====================================================

stage("STAGE 6: BOWLING")

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
    0.4*z_score(bowl["wkt_rate"]) +
    0.3*z_score(-bowl["economy"]) +
    0.3*z_score(bowl["dot_pct"])
)

# Death specialist
death_df = df[(df["phase"]=="Death") & (df["legal"])]
death_grp = death_df.groupby("bowler").agg(
    death_runs=("total_runs","sum"),
    death_balls=("legal","sum"),
    death_wkts=("is_wicket","sum")
)

death_grp["death_overs"] = death_grp["death_balls"]/6
death_grp = death_grp[death_grp["death_overs"] >= MIN_DEATH_OVERS]

death_grp["death_econ"] = death_grp["death_runs"]/death_grp["death_overs"]
death_grp["death_wkt_rate"] = death_grp["death_wkts"]/death_grp["death_overs"]

death_grp["Death_Index"] = (
    0.5*z_score(death_grp["death_wkt_rate"]) +
    0.5*z_score(-death_grp["death_econ"])
)

bowl = bowl.join(death_grp["Death_Index"]).fillna(0)

bowl["bowling_type"] = np.where(bowl.index.isin(SPINNER_LIST),"Spinner","Pace")

# =====================================================
# STAGE 7: STRUCTURED FINAL XI WITH ROLES
# =====================================================

stage("STAGE 7: FINAL XI")

selected = []
selected_set = set()
role_map = {}

def pick(df, n, role_name):
    count = 0
    for name in df.index:
        if name not in selected_set:
            selected.append(name)
            selected_set.add(name)
            role_map[name] = role_name
            count += 1
        if count == n or len(selected) == 11:
            break

# ---- BATTERS ----
pick(bat.sort_values("Opener_Score",ascending=False), 2, "Opener")
pick(bat.sort_values("Anchor_Score",ascending=False), 1, "Anchor")
pick(bat.sort_values("Middle_Score",ascending=False), 2, "Middle Order")
pick(bat.sort_values("Finisher_Score",ascending=False), 1, "Finisher")

# ---- ALL ROUNDERS ----
ar_pool = list(set(bat.index).intersection(set(bowl.index)))

if ar_pool:
    ar_df = pd.DataFrame(index=ar_pool)
    ar_df["bat"] = bat.loc[ar_pool]["SR_z"]
    ar_df["bowl"] = bowl.loc[ar_pool]["Impact_Index"]
    ar_df["AR_Index"] = 0.65*ar_df["bat"] + 0.35*ar_df["bowl"]
    print("All-Rounder Pool:")
    print(ar_df.sort_values("AR_Index",ascending=False).head())

    pick(ar_df.sort_values("AR_Index",ascending=False), 2, "All-Rounder")

# ---- BOWLING CORE ----
pick(bowl[bowl["bowling_type"]=="Spinner"].sort_values("Impact_Index",ascending=False),
     1, "Spinner")

pick(bowl[bowl["bowling_type"]=="Pace"].sort_values("Impact_Index",ascending=False),
     2, "Pacer")

# ---- DEATH SPECIALIST ----
pick(bowl.sort_values("Death_Index",ascending=False),
     1, "Death Specialist")

selected = selected[:11]

print("\nFINAL XI WITH ROLES:")
for player in selected:
    print(f"{player}  →  {role_map.get(player, 'Utility')}")

print("Total Selected:", len(selected))


# Export Batting Table
bat.reset_index().to_csv("batting_metrics.csv", index=False)

# Export Bowling Table
bowl.reset_index().to_csv("bowling_metrics.csv", index=False)

# Export Final XI
final_xi_df = pd.DataFrame({
    "player": selected,
    "role": [role_map[p] for p in selected]
})
final_xi_df.to_csv("final_xi.csv", index=False)