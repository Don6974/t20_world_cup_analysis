import requests
import json
from datetime import datetime
from tqdm import tqdm

BASE_URL = "https://site.web.api.espn.com/apis/site/v2/sports/cricket/scoreboard"
HEADERS = {"User-Agent": "Mozilla/5.0"}

AFG_MATCH_IDS = [
66187828,
66187846,
66187876,
66187898,
]

def fetch_match(match_id):
url = f"{BASE_URL}?event={match_id}"
r = requests.get(url, headers=HEADERS)
r.raise_for_status()
return r.json()

def build_match(raw):
event = raw["events"][0]
comp = event["competitions"][0]

if name == "main":
for match_id in tqdm(AFG_MATCH_IDS):
raw = fetch_match(match_id)
structured = build_match(raw)