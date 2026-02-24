import json
import pandas as pd
import os
import numpy as np

# =====================================================
# UTILITY
# =====================================================

def min_max(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

def stage(title):
    print("\n" + "="*65)
    print(title)
    print("="*65)

# =====================================================
# STAGE 1: LOAD DATA
# =====================================================

stage("STAGE 1: DATA LOADING & COVERAGE")

rows = []

for file in os.listdir("."):
    print(file)
    if file.endswith(".json"):
        with open(file) as f:
            data = json.load(f)
        print(file)

        for innings in data["innings"]:
            for over_data in innings["overs"]:
                for delivery in over_data["deliveries"]:
                    extras = delivery.get("extras", {})
                    legal = not ("wides" in extras or "noballs" in extras)

                    rows.append({
                        "match": file,
                        "over": over_data["over"],
                        "batter": delivery["batter"],
                        "bowler": delivery["bowler"],
                        "batter_runs": delivery["runs"]["batter"],
                        "total_runs": delivery["runs"]["total"],
                        "legal": legal,
                        "is_wicket": len(delivery.get("wickets", [])) > 0
                    })

df = pd.DataFrame(rows)

print("Matches Available:", df["match"].nunique())
print("Unique Batters:", df["batter"].nunique())
print("Unique Bowlers:", df["bowler"].nunique())

# =====================================================
# STAGE 2: 4-PHASE CLASSIFICATION
# =====================================================

stage("STAGE 2: 4-PHASE STRUCTURE")

def phase(over):
    if over <= 5:
        return "Powerplay"
    elif over <= 10:
        return "EarlyMiddle"
    elif over <= 15:
        return "LateMiddle"
    else:
        return "Death"

df["phase"] = df["over"].apply(phase)
df["is_dot"] = (df["legal"]) & (df["batter_runs"] == 0)
df["is_boundary"] = df["batter_runs"].isin([4,6])

print(df["phase"].value_counts())

# =====================================================
# STAGE 3: BATTING AGGREGATION
# =====================================================

stage("STAGE 3: BATTING AGGREGATION & ELIGIBILITY")

bat = df[df["legal"]].groupby("batter").agg(
    matches=("match","nunique"),
    runs=("batter_runs","sum"),
    balls=("legal","sum"),
    boundaries=("is_boundary","sum"),
    dots=("is_dot","sum")
)

print("Total Batters Before Filter:", len(bat))

bat = bat[bat["matches"] >= 3]
print("After Eligibility (>=3 matches):", len(bat))

bat["strike_rate"] = bat["runs"]/bat["balls"]*100
bat["runs_per_match"] = bat["runs"]/bat["matches"]
bat["boundary_pct"] = bat["boundaries"]/bat["balls"]
bat["dot_pct"] = bat["dots"]/bat["balls"]

print("Top 5 Runs per Match:")
print(bat.sort_values("runs_per_match",ascending=False)[
    ["matches","runs_per_match","strike_rate"]
].head())

# =====================================================
# STAGE 4: ROLE CLASSIFICATION (4-PHASE)
# =====================================================

stage("STAGE 4: ROLE CLASSIFICATION")

phase_exp = df[df["legal"]].groupby(["batter","phase"]).agg(
    balls=("legal","sum"),
    runs=("batter_runs","sum")
).reset_index()

total_balls = phase_exp.groupby("batter")["balls"].sum()

phase_exp["share"] = phase_exp.apply(
    lambda x: x["balls"]/total_balls[x["batter"]],
    axis=1
)

phase_exp["phase_sr"] = phase_exp["runs"]/phase_exp["balls"]*100

role_map = {}

for batter in bat.index:
    pdata = phase_exp[phase_exp["batter"]==batter]

    pp_share = pdata[pdata["phase"]=="Powerplay"]["share"].sum()
    death_share = pdata[pdata["phase"]=="Death"]["share"].sum()
    early_sr = pdata[pdata["phase"]=="EarlyMiddle"]["phase_sr"].mean()
    late_sr = pdata[pdata["phase"]=="LateMiddle"]["phase_sr"].mean()

    if pp_share > 0.45:
        role_map[batter] = "Opener"
    elif death_share > 0.35:
        role_map[batter] = "Finisher"
    elif early_sr < 125 and bat.loc[batter,"runs_per_match"] > 25:
        role_map[batter] = "Anchor"
    elif late_sr > 140:
        role_map[batter] = "MiddleHitter"
    else:
        role_map[batter] = "Middle"

bat["role"] = bat.index.map(role_map)

print(bat["role"].value_counts())

# =====================================================
# STAGE 5: BATTING SCORING
# =====================================================

stage("STAGE 5: BATTING SCORING")

bat["sr_norm"] = min_max(bat["strike_rate"])
bat["rpm_norm"] = min_max(bat["runs_per_match"])
bat["boundary_norm"] = min_max(bat["boundary_pct"])
bat["dot_control"] = min_max(1 - bat["dot_pct"])

bat["composite"] = (
    0.35*bat["sr_norm"] +
    0.30*bat["rpm_norm"] +
    0.20*bat["boundary_norm"] +
    0.15*bat["dot_control"]
)

print("Top 10 Batters:")
print(bat.sort_values("composite",ascending=False)[
    ["matches","runs_per_match","strike_rate","role","composite"]
].head(10))

# =====================================================
# STAGE 6: BOWLING AGGREGATION
# =====================================================

stage("STAGE 6: BOWLING AGGREGATION & ELIGIBILITY")

bowl = df[df["legal"]].groupby("bowler").agg(
    matches=("match","nunique"),
    runs=("total_runs","sum"),
    balls=("legal","sum"),
    wickets=("is_wicket","sum"),
    dots=("is_dot","sum")
)

print("Total Bowlers Before Filter:", len(bowl))

bowl["overs"] = bowl["balls"]/6
bowl = bowl[(bowl["matches"]>=3) & (bowl["overs"]>=8)]

print("After Eligibility:", len(bowl))

bowl["economy"] = bowl["runs"]/bowl["overs"]
bowl["wpm"] = bowl["wickets"]/bowl["matches"]
bowl["dot_pct"] = bowl["dots"]/bowl["balls"]

print("Top 5 Wickets per Match:")
print(bowl.sort_values("wpm",ascending=False)[
    ["matches","wpm","economy"]
].head())

# =====================================================
# STAGE 7: BOWLER ROLE CLASSIFICATION
# =====================================================

stage("STAGE 7: BOWLER ROLE CLASSIFICATION")

b_phase = df[df["legal"]].groupby(["bowler","phase"]).agg(
    balls=("legal","sum")
).reset_index()

tot_balls = b_phase.groupby("bowler")["balls"].sum()

b_phase["share"] = b_phase.apply(
    lambda x: x["balls"]/tot_balls[x["bowler"]],
    axis=1
)

b_role = {}

for bowler in bowl.index:
    pdata = b_phase[b_phase["bowler"]==bowler]
    pp = pdata[pdata["phase"]=="Powerplay"]["share"].sum()
    death = pdata[pdata["phase"]=="Death"]["share"].sum()

    if death > 0.30:
        b_role[bowler] = "Death"
    elif pp > 0.35:
        b_role[bowler] = "NewBall"
    else:
        b_role[bowler] = "Spinner"

bowl["role"] = bowl.index.map(b_role)

print(bowl["role"].value_counts())

# =====================================================
# STAGE 8: BOWLING SCORING
# =====================================================

stage("STAGE 8: BOWLING SCORING")

bowl["econ_norm"] = min_max(1/bowl["economy"])
bowl["wpm_norm"] = min_max(bowl["wpm"])
bowl["dot_norm"] = min_max(bowl["dot_pct"])

bowl["composite"] = (
    0.40*bowl["wpm_norm"] +
    0.35*bowl["econ_norm"] +
    0.25*bowl["dot_norm"]
)

print("Top 10 Bowlers:")
print(bowl.sort_values("composite",ascending=False)[
    ["matches","economy","wpm","role","composite"]
].head(10))

# =====================================================
# STAGE 9: FINAL XI SELECTION (NO DUPLICATES)
# =====================================================

stage("STAGE 9: FINAL XI SELECTION")

openers = bat[bat["role"]=="Opener"].sort_values("composite",ascending=False).head(2)
anchor = bat[bat["role"]=="Anchor"].sort_values("composite",ascending=False).head(1)
middle_hitter = bat[bat["role"]=="MiddleHitter"].sort_values("composite",ascending=False).head(1)
middle = bat[bat["role"]=="Middle"].sort_values("composite",ascending=False).head(1)
finisher = bat[bat["role"]=="Finisher"].sort_values("composite",ascending=False).head(1)

team_bat = pd.concat([openers, anchor, middle_hitter, middle, finisher])
team_bat = team_bat[~team_bat.index.duplicated()]

new_ball = bowl[bowl["role"]=="NewBall"].sort_values("composite",ascending=False).head(1)
death = bowl[bowl["role"]=="Death"].sort_values("composite",ascending=False).head(1)
spinners = bowl[bowl["role"]=="Spinner"].sort_values("composite",ascending=False).head(2)

team_bowl = pd.concat([new_ball, death, spinners])
team_bowl = team_bowl[~team_bowl.index.isin(team_bat.index)]

combined = pd.concat([team_bat, team_bowl])

if len(combined) < 11:
    remaining_bat = bat[~bat.index.isin(combined.index)].sort_values("composite",ascending=False)
    remaining_bowl = bowl[~bowl.index.isin(combined.index)].sort_values("composite",ascending=False)

    for player in remaining_bat.index:
        if len(combined) == 11:
            break
        combined = pd.concat([combined, bat.loc[[player]]])

    for player in remaining_bowl.index:
        if len(combined) == 11:
            break
        combined = pd.concat([combined, bowl.loc[[player]]])

final_xi = combined.head(11)

print("\nFINAL TEAM OF THE TOURNAMENT:")
print(final_xi.index.tolist())

print("\nROLE BREAKDOWN:")
print(bat.loc[final_xi.index.intersection(bat.index)][["role"]])
print(bowl.loc[final_xi.index.intersection(bowl.index)][["role"]])

print("Selected as Opener due to >45% Powerplay exposure and high SR.")