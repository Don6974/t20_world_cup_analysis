import json
import pandas as pd
import os

# =====================================================
# UTILITY
# =====================================================

def min_max(series):
    if series.max() == series.min():
        return series * 0
    return (series - series.min()) / (series.max() - series.min())

# =====================================================
# LOAD DATA
# =====================================================

folder_path = "."
rows = []

for file in os.listdir(folder_path):
    if file.endswith(".json"):
        with open(file) as f:
            data = json.load(f)

        for innings in data["innings"]:
            for over_data in innings["overs"]:
                over_number = over_data["over"]

                for delivery in over_data["deliveries"]:
                    extras = delivery.get("extras", {})
                    is_legal = not ("wides" in extras or "noballs" in extras)

                    rows.append({
                        "match": file,
                        "over": over_number,
                        "batter": delivery["batter"],
                        "bowler": delivery["bowler"],
                        "batter_runs": delivery["runs"]["batter"],
                        "total_runs": delivery["runs"]["total"],
                        "is_legal": is_legal,
                        "is_wicket": len(delivery.get("wickets", [])) > 0
                    })

df = pd.DataFrame(rows)

# =====================================================
# PHASE TAGGING
# =====================================================

def get_phase(over):
    if over <= 5:
        return "Powerplay"
    elif over <= 14:
        return "Middle"
    else:
        return "Death"

df["phase"] = df["over"].apply(get_phase)
df["is_boundary"] = df["batter_runs"].isin([4, 6])
df["is_dot"] = (df["is_legal"]) & (df["batter_runs"] == 0)

# =====================================================
# MATCH CONTEXT (PRESSURE PROXY)
# =====================================================

df["match_ball"] = df.groupby("match").cumcount() + 1
df["over_float"] = df["match_ball"] / 6
df["cumulative_total"] = df.groupby("match")["total_runs"].cumsum()

df["run_rate_so_far"] = df["cumulative_total"] / df["over_float"]
df["high_pressure"] = df["run_rate_so_far"] > 9  # pressure threshold

# =====================================================
# ================== BATTING MODEL ====================
# =====================================================

bat = df[df["is_legal"]].groupby("batter").agg(
    matches=("match", "nunique"),
    runs=("batter_runs", "sum"),
    balls=("is_legal", "sum"),
    boundaries=("is_boundary", "sum"),
    dots=("is_dot", "sum")
)

bat["strike_rate"] = bat["runs"] / bat["balls"] * 100
bat["runs_per_match"] = bat["runs"] / bat["matches"]
bat["dot_pct"] = bat["dots"] / bat["balls"]
bat["boundary_pct"] = bat["boundaries"] / bat["balls"]

# ----------------- Phase Exposure -----------------

phase_dist = df[df["is_legal"]].groupby(["batter", "phase"]).agg(
    balls=("is_legal", "sum")
).reset_index()

total_balls = phase_dist.groupby("batter")["balls"].sum()

phase_dist["share"] = phase_dist.apply(
    lambda x: x["balls"] / total_balls[x["batter"]],
    axis=1
)

role_map = {}

for batter in phase_dist["batter"].unique():
    pdata = phase_dist[phase_dist["batter"] == batter]
    pp = pdata[pdata["phase"]=="Powerplay"]["share"].sum()
    death = pdata[pdata["phase"]=="Death"]["share"].sum()

    if pp > 0.45:
        role_map[batter] = "Opener"
    elif death > 0.35:
        role_map[batter] = "Finisher"
    else:
        role_map[batter] = "Middle/Anchor"

bat["role"] = bat.index.map(role_map)

bat["anchor_flag"] = (bat["strike_rate"] < 125) & (bat["runs_per_match"] > 25)

bat["refined_role"] = bat.apply(
    lambda x:
        "Opener" if x["role"]=="Opener" else
        "Finisher" if x["role"]=="Finisher" else
        "Anchor" if x["anchor_flag"] else
        "Middle",
    axis=1
)

# ----------------- Role Eligibility -----------------

bat = bat[
    (bat["matches"] >= 3) &
    (
        ((bat["refined_role"].isin(["Opener","Anchor"])) & (bat["balls"] >= 80)) |
        ((bat["refined_role"]=="Middle") & (bat["balls"] >= 60)) |
        ((bat["refined_role"]=="Finisher") & (bat["balls"] >= 40))
    )
]

# ----------------- Pressure Stats -----------------

pressure_stats = df[
    (df["is_legal"]) & (df["high_pressure"])
].groupby("batter").agg(
    pressure_runs=("batter_runs","sum"),
    pressure_balls=("is_legal","sum")
)

pressure_stats["pressure_sr"] = (
    pressure_stats["pressure_runs"] /
    pressure_stats["pressure_balls"] * 100
)

bat = bat.join(pressure_stats, how="left").fillna(0)

# ----------------- Clutch Stats -----------------

clutch_stats = df[
    (df["phase"]=="Death") &
    (df["is_legal"])
].groupby("batter").agg(
    clutch_runs=("batter_runs","sum")
)

bat["clutch_runs"] = bat.index.map(clutch_stats["clutch_runs"]).fillna(0)

# ----------------- Normalisation -----------------

bat["sr_norm"] = min_max(bat["strike_rate"])
bat["rpm_norm"] = min_max(bat["runs_per_match"])
bat["dot_norm"] = min_max(1 - bat["dot_pct"])
bat["boundary_norm"] = min_max(bat["boundary_pct"])
bat["pressure_norm"] = min_max(bat["pressure_sr"])
bat["clutch_norm"] = min_max(bat["clutch_runs"])

death_share = phase_dist.groupby("batter").apply(
    lambda x: x[x["phase"]=="Death"]["share"].sum()
)

bat["death_bonus"] = min_max(bat.index.map(death_share).fillna(0))

# ----------------- Composite -----------------

bat["composite"] = (
    0.25*bat["sr_norm"] +
    0.25*bat["rpm_norm"] +
    0.15*bat["dot_norm"] +
    0.15*bat["boundary_norm"] +
    0.10*bat["death_bonus"] +
    0.05*bat["pressure_norm"] +
    0.05*bat["clutch_norm"]
)

# ----------------- Select Batters -----------------

team_batters = pd.concat([
    bat[bat["refined_role"]=="Opener"].sort_values("composite",ascending=False).head(2),
    bat[bat["refined_role"]=="Anchor"].sort_values("composite",ascending=False).head(1),
    bat[bat["refined_role"]=="Middle"].sort_values("composite",ascending=False).head(2),
    bat[bat["refined_role"]=="Finisher"].sort_values("composite",ascending=False).head(1)
])

# =====================================================
# ================== BOWLING MODEL ====================
# =====================================================

bowl = df[df["is_legal"]].groupby("bowler").agg(
    matches=("match","nunique"),
    runs_conceded=("total_runs","sum"),
    balls=("is_legal","sum"),
    wickets=("is_wicket","sum"),
    dots=("is_dot","sum")
)

bowl["overs"] = bowl["balls"]/6
bowl["economy"] = bowl["runs_conceded"]/bowl["overs"]
bowl["wpm"] = bowl["wickets"]/bowl["matches"]
bowl["dot_pct"] = bowl["dots"]/bowl["balls"]

bowl = bowl[(bowl["matches"]>=3) & (bowl["overs"]>=8)]

# ----------------- Death Wickets -----------------

death_wickets = df[
    (df["phase"]=="Death") &
    (df["is_wicket"])
].groupby("bowler").size()

bowl["death_wickets"] = bowl.index.map(death_wickets).fillna(0)

# ----------------- Phase Role -----------------

bowler_phase = df[df["is_legal"]].groupby(["bowler","phase"]).agg(
    balls=("is_legal","sum")
).reset_index()

total_balls_bowler = bowler_phase.groupby("bowler")["balls"].sum()

bowler_phase["share"] = bowler_phase.apply(
    lambda x: x["balls"]/total_balls_bowler[x["bowler"]],
    axis=1
)

bowler_role_map = {}

for bowler in bowl.index:
    pdata = bowler_phase[bowler_phase["bowler"]==bowler]
    pp = pdata[pdata["phase"]=="Powerplay"]["share"].sum()
    death = pdata[pdata["phase"]=="Death"]["share"].sum()

    if death > 0.30:
        bowler_role_map[bowler] = "Death"
    elif pp > 0.35:
        bowler_role_map[bowler] = "Powerplay"
    else:
        bowler_role_map[bowler] = "Middle/Spinner"

bowl["role"] = bowl.index.map(bowler_role_map)

# ----------------- Normalisation -----------------

bowl["econ_inv"] = 1/bowl["economy"]
bowl["econ_norm"] = min_max(bowl["econ_inv"])
bowl["wpm_norm"] = min_max(bowl["wpm"])
bowl["dot_norm"] = min_max(bowl["dot_pct"])
bowl["death_wicket_norm"] = min_max(bowl["death_wickets"])

# ----------------- Composite -----------------

bowl["composite"] = (
    0.30*bowl["wpm_norm"] +
    0.25*bowl["econ_norm"] +
    0.20*bowl["dot_norm"] +
    0.15*bowl["death_wicket_norm"] +
    0.10*min_max(bowl["wickets"])
)

# ----------------- Role-Enforced Selection -----------------

pp_bowlers = bowl[bowl["role"]=="Powerplay"].sort_values("composite",ascending=False).head(2)
death_bowler = bowl[bowl["role"]=="Death"].sort_values("composite",ascending=False).head(1)
middle_bowler = bowl[bowl["role"]=="Middle/Spinner"].sort_values("composite",ascending=False).head(1)

selected = set(pp_bowlers.index) | set(death_bowler.index) | set(middle_bowler.index)
remaining = bowl[~bowl.index.isin(selected)]
best_remaining = remaining.sort_values("composite",ascending=False).head(1)

team_bowlers = pd.concat([pp_bowlers, death_bowler, middle_bowler, best_remaining])

# =====================================================
# FINAL XI
# =====================================================

final_xi = pd.concat([team_batters, team_bowlers])

print("\n==============================")
print("ADVANCED ELITE TEAM OF THE TOURNAMENT")
print("==============================")
print(final_xi.index.tolist())
print(set(team_batters.index).intersection(set(team_bowlers.index)))