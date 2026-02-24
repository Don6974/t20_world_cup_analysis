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

        venue = data["info"]["venue"]

        for innings in data["innings"]:
            for over_data in innings["overs"]:
                over_number = over_data["over"]

                for delivery in over_data["deliveries"]:
                    extras = delivery.get("extras", {})
                    is_legal = not ("wides" in extras or "noballs" in extras)

                    rows.append({
                        "match": file,
                        "venue": venue,
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
df["is_boundary"] = df["batter_runs"].isin([4,6])
df["is_dot"] = (df["is_legal"]) & (df["batter_runs"]==0)

# =====================================================
# MATCH PRESSURE PROXY
# =====================================================

df["match_ball"] = df.groupby("match").cumcount() + 1
df["over_float"] = df["match_ball"] / 6
df["cumulative_total"] = df.groupby("match")["total_runs"].cumsum()
df["run_rate_so_far"] = df["cumulative_total"] / df["over_float"]
df["high_pressure"] = df["run_rate_so_far"] > 9

# =====================================================
# VENUE PROFILE
# =====================================================

venue_profile = df[df["is_legal"]].groupby(["match","venue"]).agg(
    total_runs=("total_runs","sum"),
    balls=("is_legal","sum")
).reset_index()

venue_profile["run_rate"] = venue_profile["total_runs"] / (venue_profile["balls"]/6)
venue_summary = venue_profile.groupby("venue")["run_rate"].mean()

overall_rr = venue_summary.mean()

# =====================================================
# BATTING MODEL
# =====================================================

bat = df[df["is_legal"]].groupby("batter").agg(
    matches=("match","nunique"),
    runs=("batter_runs","sum"),
    balls=("is_legal","sum"),
    boundaries=("is_boundary","sum"),
    dots=("is_dot","sum")
)

bat["strike_rate"] = bat["runs"]/bat["balls"]*100
bat["runs_per_match"] = bat["runs"]/bat["matches"]
bat["dot_pct"] = bat["dots"]/bat["balls"]
bat["boundary_pct"] = bat["boundaries"]/bat["balls"]

# Phase roles
phase_dist = df[df["is_legal"]].groupby(["batter","phase"]).agg(
    balls=("is_legal","sum")
).reset_index()

total_balls = phase_dist.groupby("batter")["balls"].sum()

phase_dist["share"] = phase_dist.apply(
    lambda x: x["balls"]/total_balls[x["batter"]], axis=1
)

role_map = {}
for batter in phase_dist["batter"].unique():
    pdata = phase_dist[phase_dist["batter"]==batter]
    pp = pdata[pdata["phase"]=="Powerplay"]["share"].sum()
    death = pdata[pdata["phase"]=="Death"]["share"].sum()

    if pp > 0.45:
        role_map[batter] = "Opener"
    elif death > 0.35:
        role_map[batter] = "Finisher"
    else:
        role_map[batter] = "Middle/Anchor"

bat["role"] = bat.index.map(role_map)
bat["anchor_flag"] = (bat["strike_rate"]<125) & (bat["runs_per_match"]>25)

bat["refined_role"] = bat.apply(
    lambda x:
        "Opener" if x["role"]=="Opener" else
        "Finisher" if x["role"]=="Finisher" else
        "Anchor" if x["anchor_flag"] else
        "Middle",
    axis=1
)

# Eligibility
bat = bat[
    (bat["matches"]>=3) &
    (
        ((bat["refined_role"].isin(["Opener","Anchor"])) & (bat["balls"]>=80)) |
        ((bat["refined_role"]=="Middle") & (bat["balls"]>=60)) |
        ((bat["refined_role"]=="Finisher") & (bat["balls"]>=40))
    )
]

# Pressure & clutch
pressure = df[(df["is_legal"]) & (df["high_pressure"])].groupby("batter")["batter_runs"].sum()
clutch = df[(df["phase"]=="Death") & (df["is_legal"])].groupby("batter")["batter_runs"].sum()

bat["pressure_runs"] = bat.index.map(pressure).fillna(0)
bat["clutch_runs"] = bat.index.map(clutch).fillna(0)

# Normalisation
bat["sr_n"] = min_max(bat["strike_rate"])
bat["rpm_n"] = min_max(bat["runs_per_match"])
bat["dot_n"] = min_max(1-bat["dot_pct"])
bat["bnd_n"] = min_max(bat["boundary_pct"])
bat["press_n"] = min_max(bat["pressure_runs"])
bat["clutch_n"] = min_max(bat["clutch_runs"])

bat["base_score"] = (
    0.25*bat["sr_n"] +
    0.25*bat["rpm_n"] +
    0.15*bat["dot_n"] +
    0.15*bat["bnd_n"] +
    0.10*bat["press_n"] +
    0.10*bat["clutch_n"]
)

# =====================================================
# BOWLING MODEL
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

# Phase role
bowler_phase = df[df["is_legal"]].groupby(["bowler","phase"]).agg(
    balls=("is_legal","sum")
).reset_index()

total_balls_b = bowler_phase.groupby("bowler")["balls"].sum()
bowler_phase["share"] = bowler_phase.apply(
    lambda x: x["balls"]/total_balls_b[x["bowler"]], axis=1
)

b_role = {}
for bowler in bowl.index:
    pdata = bowler_phase[bowler_phase["bowler"]==bowler]
    pp = pdata[pdata["phase"]=="Powerplay"]["share"].sum()
    death = pdata[pdata["phase"]=="Death"]["share"].sum()

    if death > 0.30:
        b_role[bowler] = "Death"
    elif pp > 0.35:
        b_role[bowler] = "Powerplay"
    else:
        b_role[bowler] = "Middle"

bowl["role"] = bowl.index.map(b_role)

# Normalisation
bowl["econ_n"] = min_max(1/bowl["economy"])
bowl["wpm_n"] = min_max(bowl["wpm"])
bowl["dot_n"] = min_max(bowl["dot_pct"])

bowl["base_score"] = (
    0.35*bowl["wpm_n"] +
    0.30*bowl["econ_n"] +
    0.20*bowl["dot_n"] +
    0.15*min_max(bowl["wickets"])
)

# =====================================================
# FUNCTION TO SELECT TEAM PER VENUE
# =====================================================
def select_team(venue_name):

    bat_adj = bat.copy()
    bowl_adj = bowl.copy()

    # =========================
    # AHMEDABAD LOGIC
    # =========================
    if venue_name == "Narendra Modi Stadium, Ahmedabad":

        # High scoring, dew factor, death heavy

        bat_adj["score"] = (
            bat_adj["base_score"]
            + 0.25 * bat_adj["bnd_n"]        # reward power hitting
            + 0.15 * bat_adj["press_n"]      # chasing under dew
            + 0.10 * bat_adj["clutch_n"]     # death finishing
        )

        bowl_adj["score"] = (
            bowl_adj["base_score"]
            + 0.25 * bowl_adj["wpm_n"]       # wicket takers matter
            + 0.15 * min_max(bowl_adj["wpm"])  # aggressive strike bowlers
        )

    # =========================
    # COLOMBO LOGIC
    # =========================
    elif venue_name == "R Premadasa Stadium, Colombo":

        # Slower surface, spin friendly, low scoring

        bat_adj["score"] = (
            bat_adj["base_score"]
            + 0.25 * bat_adj["dot_n"]        # reward rotation
            + 0.15 * bat_adj["rpm_n"]        # consistency
        )

        bowl_adj["score"] = (
            bowl_adj["base_score"]
            + 0.30 * bowl_adj["econ_n"]      # economy critical
            + 0.15 * bowl_adj["dot_n"]       # pressure build
        )

    # =========================
    # DEFAULT (fallback)
    # =========================
    else:
        bat_adj["score"] = bat_adj["base_score"]
        bowl_adj["score"] = bowl_adj["base_score"]

    # =========================
    # ROLE-ENFORCED BATTING
    # =========================

    team_bat = pd.concat([
        bat_adj[bat_adj["refined_role"]=="Opener"]
            .sort_values("score",ascending=False).head(2),

        bat_adj[bat_adj["refined_role"]=="Anchor"]
            .sort_values("score",ascending=False).head(1),

        bat_adj[bat_adj["refined_role"]=="Middle"]
            .sort_values("score",ascending=False).head(2),

        bat_adj[bat_adj["refined_role"]=="Finisher"]
            .sort_values("score",ascending=False).head(1),
    ])

    # =========================
    # ROLE-ENFORCED BOWLING
    # =========================

    pp = bowl_adj[bowl_adj["role"]=="Powerplay"] \
            .sort_values("score",ascending=False).head(2)

    death = bowl_adj[bowl_adj["role"]=="Death"] \
            .sort_values("score",ascending=False).head(1)

    middle = bowl_adj[bowl_adj["role"]=="Middle"] \
            .sort_values("score",ascending=False).head(1)

    selected = set(pp.index) | set(death.index) | set(middle.index)

    remaining = bowl_adj[~bowl_adj.index.isin(selected)] \
                    .sort_values("score",ascending=False).head(1)

    team_bowl = pd.concat([pp, death, middle, remaining])

    return pd.concat([team_bat, team_bowl]).index.tolist()
#######


# =====================================================
# OUTPUT BOTH TEAMS
# =====================================================

ahmedabad_team = select_team("Narendra Modi Stadium, Ahmedabad")
colombo_team = select_team("R Premadasa Stadium, Colombo")

print("\n==============================")
print("AHMEDABAD FINAL XI")
print("==============================")
print(ahmedabad_team)

print("\n==============================")
print("COLOMBO FINAL XI")
print("==============================")
print(colombo_team)