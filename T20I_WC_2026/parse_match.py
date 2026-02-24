import json
import pandas as pd

with open("1.json") as f:
    data = json.load(f)

match_info = data["info"]
match_id = f"{match_info['event']['match_number']}_{match_info['dates'][0]}"
venue = match_info["venue"]

rows = []

for innings_index, innings in enumerate(data["innings"], start=1):
    batting_team = innings["team"]

    for over_data in innings["overs"]:
        over_number = over_data["over"]
        legal_ball_count = 0

        for delivery in over_data["deliveries"]:

            batter = delivery["batter"]
            bowler = delivery["bowler"]

            runs_batter = delivery["runs"]["batter"]
            runs_total = delivery["runs"]["total"]

            extras = delivery.get("extras", {})
            is_wide = "wides" in extras
            is_noball = "noballs" in extras

            is_legal = not is_wide and not is_noball

            if is_legal:
                legal_ball_count += 1

            wicket_info = delivery.get("wickets", [])
            is_wicket = len(wicket_info) > 0
            dismissal_type = wicket_info[0]["kind"] if is_wicket else None

            rows.append({
                "match_id": match_id,
                "innings": innings_index,
                "batting_team": batting_team,
                "over": over_number,
                "legal_ball_number": legal_ball_count if is_legal else None,
                "batter": batter,
                "bowler": bowler,
                "batter_runs": runs_batter,
                "total_runs": runs_total,
                "is_legal": is_legal,
                "is_wicket": is_wicket,
                "dismissal_type": dismissal_type,
                "venue": venue
            })

df = pd.DataFrame(rows)

print("Shape:", df.shape)
print(df.head())


print("\nRuns per innings:")
print(df.groupby("innings")["total_runs"].sum())

print("\nWickets per innings:")
print(df.groupby("innings")["is_wicket"].sum())

print("\nLegal balls per innings:")
print(df[df["is_legal"]].groupby("innings").size())

#phase mapping
def get_phase(over):
    if over <= 5:
        return "Powerplay"
    elif over <= 14:
        return "Middle"
    else:
        return "Death"

df["phase"] = df["over"].apply(get_phase)

#adding boundary & dot flags
df["is_boundary"] = df["batter_runs"].isin([4, 6])

df["is_dot"] = (
    (df["is_legal"]) &
    (df["batter_runs"] == 0)
)

#BATTING SUMMARY

batting_summary = df[df["is_legal"]].groupby("batter").agg(
    runs=("batter_runs", "sum"),
    balls=("is_legal", "sum"),
    boundaries=("is_boundary", "sum"),
    dots=("is_dot", "sum")
)

batting_summary["strike_rate"] = (
    batting_summary["runs"] / batting_summary["balls"] * 100
)

batting_summary["boundary_pct"] = (
    batting_summary["boundaries"] / batting_summary["balls"]
)

batting_summary["dot_pct"] = (
    batting_summary["dots"] / batting_summary["balls"]
)

print("\nBatting Summary:")
print(batting_summary.sort_values("runs", ascending=False))


#PHASE BASED BATTING
phase_batting = df[df["is_legal"]].groupby(["batter", "phase"]).agg(
    runs=("batter_runs", "sum"),
    balls=("is_legal", "sum"),
    boundaries=("is_boundary", "sum"),
    dots=("is_dot", "sum")
).reset_index()

phase_batting["strike_rate"] = (
    phase_batting["runs"] / phase_batting["balls"] * 100
)

print("\nPhase Batting:")
print(phase_batting.sort_values(["batter", "phase"]))

bowling_summary = df[df["is_legal"]].groupby("bowler").agg(
    runs_conceded=("total_runs", "sum"),
    balls=("is_legal", "sum"),
    wickets=("is_wicket", "sum"),
    dots=("is_dot", "sum")
)

bowling_summary["overs"] = bowling_summary["balls"] / 6
bowling_summary["economy"] = bowling_summary["runs_conceded"] / bowling_summary["overs"]
bowling_summary["dot_pct"] = bowling_summary["dots"] / bowling_summary["balls"]

print("\nBowling Summary:")
print(bowling_summary.sort_values("economy"))

#BATTING IMPACT SCORE
match_avg_runs = df.groupby("innings")["total_runs"].sum().mean()

batting_summary["impact"] = (
    batting_summary["runs"] - (match_avg_runs / 6)
)

print("\nBatting Impact Score:")
print(batting_summary.sort_values("impact", ascending=False))


#BOWLING IMPACT SCORE

match_avg_runs = df.groupby("innings")["total_runs"].sum().mean()

batting_summary["impact"] = (
    batting_summary["runs"] - (match_avg_runs / 6)
)

print("\nBatting Impact Score:")
print(batting_summary.sort_values("impact", ascending=False))