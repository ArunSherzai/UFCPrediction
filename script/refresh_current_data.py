import json
import math
import os
import re
import sys
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from dataCleaning import clean_and_prepare_data
from model_utils import train_all_models


BASE_URL = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFCPredictionBot/1.0)"}

DATA_DIR = SCRIPT_DIR.parent / "data"
RAW_PATH = DATA_DIR / "raw_total_fight_data.csv"
FIGHTER_DETAILS_PATH = DATA_DIR / "row_fighter_details.csv"
MAIN_DATA_PATH = DATA_DIR / "data.csv"
REFRESH_METADATA_PATH = DATA_DIR / "refresh_metadata.json"
SCRAPE_CACHE_PATH = DATA_DIR / "new_fights_cache.csv"
SCRAPE_PROGRESS_PATH = DATA_DIR / "new_fights_progress.json"
BOOKMAKER_ODDS_PATH = DATA_DIR / "bookmaker_odds.csv"
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"

RAW_COLUMNS = [
    "R_fighter", "B_fighter", "R_KD", "B_KD", "R_SIG_STR.", "B_SIG_STR.", "R_SIG_STR_pct", "B_SIG_STR_pct",
    "R_TOTAL_STR.", "B_TOTAL_STR.", "R_TD", "B_TD", "R_TD_pct", "B_TD_pct", "R_SUB_ATT", "B_SUB_ATT",
    "R_REV", "B_REV", "R_CTRL", "B_CTRL", "win_by", "last_round", "last_round_time", "Format",
    "Referee", "date", "location", "Fight_type", "Winner"
]

BOOKMAKER_ODDS_COLUMNS = [
    "event_name",
    "event_date",
    "bookmaker_name",
    "market",
    "fighter_a",
    "fighter_b",
    "fighter_a_decimal_odds",
    "fighter_b_decimal_odds",
    "fighter_a_implied_prob",
    "fighter_b_implied_prob",
    "bookmaker_reason",
    "odds_timestamp",
]


def fetch(url, retries=3, sleep_s=1.0):
    last_error = None
    for _ in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_s)
    raise RuntimeError(f"Fetch failed for {url}: {last_error}")


def parse_date(value):
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def normalize_fighter_name(name):
    cleaned = str(name or "").strip().lower()
    cleaned = cleaned.replace(".", "")
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace("-", " ")
    return " ".join(cleaned.split())


def load_existing_raw():
    raw = pd.read_csv(RAW_PATH, sep=";")
    raw["parsed_date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw


def get_latest_existing_date(raw_df):
    latest = raw_df["parsed_date"].max()
    if pd.isna(latest):
        raise ValueError("Konnte max. Datum in raw_total_fight_data.csv nicht bestimmen.")
    return latest.normalize()


def scrape_completed_events():
    soup = BeautifulSoup(fetch(f"{BASE_URL}/statistics/events/completed?page=all"), "html.parser")
    events = []
    for row in soup.select("tr.b-statistics__table-row"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        link = row.select_one("a[href]")
        if not link:
            continue
        text = cells[0].get_text(" ", strip=True)
        location = cells[1].get_text(" ", strip=True)
        match = re.match(r"(.+?)\s+([A-Z][a-z]{2}\.\s+\d{2},\s+\d{4}|[A-Z][a-z]+\s+\d{2},\s+\d{4})$", text)
        if not match:
            continue
        events.append(
            {
                "event_name": match.group(1).strip(),
                "date": pd.to_datetime(match.group(2), errors="coerce"),
                "location": location,
                "event_url": link["href"],
            }
        )
    return pd.DataFrame(events).sort_values("date").reset_index(drop=True)


def parse_person_result(soup):
    people = []
    for div in soup.select("div.b-fight-details__person"):
        txt = div.get_text(" ", strip=True)
        m = re.match(r"([WLDNC]+)\s+(.+?)(?:\s+\".*)?$", txt)
        if not m:
            continue
        people.append((m.group(1), m.group(2).strip()))
    return people


def parse_text_item_map(soup):
    out = {}
    for item in soup.select("i.b-fight-details__text-item, p.b-fight-details__text"):
        txt = " ".join(item.stripped_strings)
        if ":" in txt:
            parts = re.findall(r"([A-Za-z .]+):\s*([^:]+?)(?=(?:\s+[A-Za-z .]+:)|$)", txt)
            if parts:
                for key, value in parts:
                    out[key.strip()] = value.strip()
            else:
                key, value = txt.split(":", 1)
                out[key.strip()] = value.strip()
    return out


def parse_total_row(soup):
    rows = [row for row in soup.select("tr.b-fight-details__table-row") if row.select("td")]
    if not rows:
        raise ValueError("Keine Statistikreihen auf Fight-Detail-Seite gefunden.")
    first = rows[0]
    values = []
    for td in first.select("td"):
        texts = [p.get_text(" ", strip=True) for p in td.select("p")]
        values.append(texts)
    return values


def scrape_fight(fight_url, event_date, location):
    soup = BeautifulSoup(fetch(fight_url), "html.parser")
    info = parse_text_item_map(soup)
    total_row = parse_total_row(soup)
    people = parse_person_result(soup)
    fight_title = soup.select_one(".b-fight-details__fight-title").get_text(" ", strip=True)

    fighter_names = total_row[0]
    if len(fighter_names) != 2:
        raise ValueError(f"Unerwartete Fighter-Zelle in {fight_url}: {fighter_names}")
    red_name, blue_name = fighter_names[0], fighter_names[1]

    result_lookup = {name: result for result, name in people}
    red_result = result_lookup.get(red_name)
    blue_result = result_lookup.get(blue_name)
    if red_result == "W":
        winner = "Red"
    elif blue_result == "W":
        winner = "Blue"
    else:
        winner = None

    fighter_links = [a["href"] for a in soup.select("a.b-link.b-fight-details__person-link")]
    fighter_link_map = {}
    for anchor in soup.select("a.b-link.b-fight-details__person-link"):
        fighter_link_map[anchor.get_text(" ", strip=True)] = anchor["href"]

    def pair_at(index):
        values = total_row[index]
        return values[0] if len(values) > 0 else np.nan, values[1] if len(values) > 1 else np.nan

    r_kd, b_kd = pair_at(1)
    r_sig, b_sig = pair_at(2)
    r_sig_pct, b_sig_pct = pair_at(3)
    r_total, b_total = pair_at(4)
    r_td, b_td = pair_at(5)
    r_td_pct, b_td_pct = pair_at(6)
    r_sub, b_sub = pair_at(7)
    r_rev, b_rev = pair_at(8)
    r_ctrl, b_ctrl = pair_at(9)

    return {
        "R_fighter": red_name,
        "B_fighter": blue_name,
        "R_fighter_url": fighter_link_map.get(red_name),
        "B_fighter_url": fighter_link_map.get(blue_name),
        "R_KD": r_kd,
        "B_KD": b_kd,
        "R_SIG_STR.": r_sig,
        "B_SIG_STR.": b_sig,
        "R_SIG_STR_pct": r_sig_pct,
        "B_SIG_STR_pct": b_sig_pct,
        "R_TOTAL_STR.": r_total,
        "B_TOTAL_STR.": b_total,
        "R_TD": r_td,
        "B_TD": b_td,
        "R_TD_pct": r_td_pct,
        "B_TD_pct": b_td_pct,
        "R_SUB_ATT": r_sub,
        "B_SUB_ATT": b_sub,
        "R_REV": r_rev,
        "B_REV": b_rev,
        "R_CTRL": r_ctrl,
        "B_CTRL": b_ctrl,
        "win_by": info.get("Method"),
        "last_round": info.get("Round"),
        "last_round_time": info.get("Time"),
        "Format": info.get("Time format"),
        "Referee": info.get("Referee"),
        "date": event_date.strftime("%B %d, %Y"),
        "location": location,
        "Fight_type": fight_title,
        "Winner": winner,
    }


def load_scrape_progress():
    if SCRAPE_PROGRESS_PATH.exists():
        with open(SCRAPE_PROGRESS_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"completed_event_urls": []}


def save_scrape_progress(progress):
    with open(SCRAPE_PROGRESS_PATH, "w", encoding="utf-8") as handle:
        json.dump(progress, handle, ensure_ascii=True, indent=2)


def append_fights_cache(new_rows):
    frame = pd.DataFrame(new_rows)
    if frame.empty:
        return
    if SCRAPE_CACHE_PATH.exists():
        frame.to_csv(SCRAPE_CACHE_PATH, mode="a", header=False, index=False)
    else:
        frame.to_csv(SCRAPE_CACHE_PATH, index=False)


def scrape_new_fights(latest_existing_date, max_workers=10):
    events = scrape_completed_events()
    new_events = events[events["date"] > latest_existing_date].copy()
    progress = load_scrape_progress()
    completed_event_urls = set(progress.get("completed_event_urls", []))
    cached_rows = pd.read_csv(SCRAPE_CACHE_PATH).to_dict("records") if SCRAPE_CACHE_PATH.exists() else []
    fights = list(cached_rows)

    for _, event in new_events.iterrows():
        if event["event_url"] in completed_event_urls:
            continue
        print(f"Scrape Event: {event['event_name']} ({event['date'].date()})")
        soup = BeautifulSoup(fetch(event["event_url"]), "html.parser")
        fight_links = []
        for row in soup.select("tr.b-fight-details__table-row"):
            cells = row.select("td")
            links = row.select("a[href]")
            if len(cells) < 10 or not links:
                continue
            fight_links.append(links[0]["href"])

        new_rows = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(scrape_fight, fight_link, event["date"], event["location"]): fight_link
                for fight_link in fight_links
            }
            for future in as_completed(future_map):
                fight_link = future_map[future]
                try:
                    new_rows.append(future.result())
                except Exception as exc:
                    print(f"  Fight failed: {fight_link} -> {exc}")

        fights.extend(new_rows)
        append_fights_cache(new_rows)
        completed_event_urls.add(event["event_url"])
        save_scrape_progress({"completed_event_urls": sorted(completed_event_urls)})

    return pd.DataFrame(fights)


def scrape_fighter_details(fighter_urls):
    rows = []
    for fighter_name, fighter_url in fighter_urls.items():
        if not fighter_url:
            continue
        try:
            soup = BeautifulSoup(fetch(fighter_url), "html.parser")
            stats = {}
            for item in soup.select("li.b-list__box-list-item"):
                txt = " ".join(item.stripped_strings)
                if ":" in txt:
                    key, value = txt.split(":", 1)
                    stats[key.strip()] = value.strip()
            rows.append(
                {
                    "fighter_name": fighter_name,
                    "Height": stats.get("Height"),
                    "Weight": stats.get("Weight"),
                    "Reach": stats.get("Reach"),
                    "Stance": stats.get("STANCE"),
                    "DOB": stats.get("DOB"),
                    "SLpM": stats.get("SLpM"),
                    "Str_Acc": stats.get("Str. Acc."),
                    "SApM": stats.get("SApM"),
                    "Str_Def": stats.get("Str. Def"),
                    "TD_Avg": stats.get("TD Avg."),
                    "TD_Acc": stats.get("TD Acc."),
                    "TD_Def": stats.get("TD Def."),
                    "Sub_Avg": stats.get("Sub. Avg."),
                }
            )
        except Exception as exc:
            print(f"  Fighter failed: {fighter_name} -> {exc}")
        time.sleep(0.1)
    return pd.DataFrame(rows)


def load_existing_bookmaker_odds():
    if not BOOKMAKER_ODDS_PATH.exists():
        return pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)
    odds = pd.read_csv(BOOKMAKER_ODDS_PATH)
    for column in BOOKMAKER_ODDS_COLUMNS:
        if column not in odds.columns:
            odds[column] = np.nan
    return odds[BOOKMAKER_ODDS_COLUMNS].copy()


def decimal_implied_probability(odds_value):
    try:
        decimal_odds = float(odds_value)
        if decimal_odds <= 0:
            return np.nan
        return round(1.0 / decimal_odds, 6)
    except Exception:
        return np.nan


def american_to_decimal(odds_value):
    try:
        text = str(odds_value).strip()
        if text in {"", "-", "+-", "nan", "None", "+0", "0"}:
            return np.nan
        american = int(text)
        if american > 0:
            return round(1 + (american / 100.0), 6)
        return round(1 + (100.0 / abs(american)), 6)
    except Exception:
        return np.nan


def scrape_covers_odds():
    url = "https://www.covers.com/sport/mma/ufc/odds"
    soup = BeautifulSoup(fetch(url), "html.parser")
    rows = []

    for event_heading in soup.find_all("h2"):
        event_name = " ".join(event_heading.get_text(" ", strip=True).split())
        if not event_name.startswith(("UFC", "MVP")):
            continue

        table = event_heading.find_next("table")
        if table is None:
            continue

        event_date = np.nan
        date_div = event_heading.find_next("div")
        if date_div is not None:
            date_text = " ".join(date_div.get_text(" ", strip=True).split())
            if " ET" in date_text or "GMT" in date_text:
                parsed = pd.to_datetime(date_text, errors="coerce")
                if pd.notna(parsed):
                    event_date = parsed.strftime("%Y-%m-%d")
        if pd.isna(event_date):
            event_date = pd.to_datetime(event_name, errors="coerce")
            event_date = event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else np.nan

        headers = table.find_all("th")
        if len(headers) < 5:
            continue
        sportsbook_names = ["Open"]
        for th in headers[2:5]:
            img = th.find("img")
            sportsbook_names.append(img.get("alt", "Bookmaker") if img else "Bookmaker")

        fight_rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) != 5:
                continue
            fighter_img = cells[0].find("img")
            if fighter_img is None:
                continue
            fighter_name = fighter_img.get("alt")
            odds = [" ".join(cell.get_text(" ", strip=True).split()) for cell in cells[1:5]]
            fight_rows.append({"fighter_name": fighter_name, "odds": odds})

        for idx in range(0, len(fight_rows), 2):
            pair = fight_rows[idx:idx + 2]
            if len(pair) != 2:
                continue
            fighter_a = pair[0]["fighter_name"]
            fighter_b = pair[1]["fighter_name"]
            for book_idx, bookmaker_name in enumerate(sportsbook_names[1:], start=1):
                fighter_a_decimal = american_to_decimal(pair[0]["odds"][book_idx])
                fighter_b_decimal = american_to_decimal(pair[1]["odds"][book_idx])
                if pd.isna(fighter_a_decimal) and pd.isna(fighter_b_decimal):
                    continue
                rows.append(
                    {
                        "event_name": event_name,
                        "event_date": event_date,
                        "bookmaker_name": bookmaker_name,
                        "market": "h2h",
                        "fighter_a": fighter_a,
                        "fighter_b": fighter_b,
                        "fighter_a_decimal_odds": fighter_a_decimal,
                        "fighter_b_decimal_odds": fighter_b_decimal,
                        "fighter_a_implied_prob": decimal_implied_probability(fighter_a_decimal),
                        "fighter_b_implied_prob": decimal_implied_probability(fighter_b_decimal),
                        "bookmaker_reason": np.nan,
                        "odds_timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)
    return pd.DataFrame(rows, columns=BOOKMAKER_ODDS_COLUMNS)


def fetch_bookmaker_odds():
    scraped = scrape_covers_odds()
    if not scraped.empty:
        return scraped

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        return pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)

    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    response = requests.get(ODDS_API_URL, params=params, headers=HEADERS, timeout=30)
    try:
        response.raise_for_status()
    except requests.HTTPError:
        return pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)
    payload = response.json()
    rows = []

    for event in payload:
        event_name = event.get("commence_time")
        event_date = pd.to_datetime(event.get("commence_time"), errors="coerce")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        if not home_team or not away_team:
            continue

        bookmakers = event.get("bookmakers") or []
        for bookmaker in bookmakers:
            markets = bookmaker.get("markets") or []
            for market in markets:
                if market.get("key") != "h2h":
                    continue
                outcomes = market.get("outcomes") or []
                if len(outcomes) != 2:
                    continue
                outcome_map = {outcome.get("name"): outcome.get("price") for outcome in outcomes}
                fighter_a_odds = outcome_map.get(home_team)
                fighter_b_odds = outcome_map.get(away_team)
                rows.append(
                    {
                        "event_name": event.get("home_team") + " vs " + event.get("away_team"),
                        "event_date": event_date.strftime("%Y-%m-%d") if pd.notna(event_date) else np.nan,
                        "bookmaker_name": bookmaker.get("title"),
                        "market": market.get("key"),
                        "fighter_a": home_team,
                        "fighter_b": away_team,
                        "fighter_a_decimal_odds": fighter_a_odds,
                        "fighter_b_decimal_odds": fighter_b_odds,
                        "fighter_a_implied_prob": decimal_implied_probability(fighter_a_odds),
                        "fighter_b_implied_prob": decimal_implied_probability(fighter_b_odds),
                        "bookmaker_reason": np.nan,
                        "odds_timestamp": pd.to_datetime(bookmaker.get("last_update"), errors="coerce").strftime("%Y-%m-%dT%H:%M:%SZ")
                        if bookmaker.get("last_update") else np.nan,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=BOOKMAKER_ODDS_COLUMNS)
    return pd.DataFrame(rows, columns=BOOKMAKER_ODDS_COLUMNS)


def refresh_bookmaker_odds():
    existing = load_existing_bookmaker_odds()
    fetched = fetch_bookmaker_odds()
    if fetched.empty:
        if not BOOKMAKER_ODDS_PATH.exists():
            existing.to_csv(BOOKMAKER_ODDS_PATH, index=False)
        return existing

    combined = pd.concat([existing, fetched], ignore_index=True)
    combined["event_date"] = pd.to_datetime(combined["event_date"], errors="coerce")
    combined["odds_timestamp"] = pd.to_datetime(combined["odds_timestamp"], errors="coerce")
    combined["fighter_a_norm"] = combined["fighter_a"].map(normalize_fighter_name)
    combined["fighter_b_norm"] = combined["fighter_b"].map(normalize_fighter_name)
    combined = combined.sort_values(["event_date", "odds_timestamp"], ascending=[False, False], na_position="last")
    combined = combined.drop_duplicates(
        subset=["event_date", "bookmaker_name", "market", "fighter_a_norm", "fighter_b_norm"],
        keep="first",
    )
    combined = combined[BOOKMAKER_ODDS_COLUMNS]
    combined.to_csv(BOOKMAKER_ODDS_PATH, index=False)
    return combined


def to_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return 0


def parse_percent(value):
    value = str(value).strip()
    if value in {"---", "--", "nan", "None", ""}:
        return np.nan
    return float(value.replace("%", ""))


def parse_of(value):
    value = str(value).strip()
    match = re.match(r"(\d+)\s+of\s+(\d+)", value)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def parse_ctrl_to_seconds(value):
    value = str(value).strip()
    if value in {"---", "--", "nan", "None", ""}:
        return 0
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + int(seconds)


def feet_to_cm(value):
    value = str(value).strip()
    match = re.match(r"(\d+)'\s*(\d+)\"", value)
    if not match:
        return np.nan
    inches = int(match.group(1)) * 12 + int(match.group(2))
    return round(inches * 2.54, 2)


def lbs_to_int(value):
    value = str(value).strip()
    match = re.match(r"(\d+)", value)
    return int(match.group(1)) if match else np.nan


def parse_bio_frame():
    bios = pd.read_csv(FIGHTER_DETAILS_PATH)
    bios = bios.drop_duplicates(subset=["fighter_name"], keep="last")
    bios["Height_cms"] = bios["Height"].apply(feet_to_cm)
    bios["Reach_cms"] = bios["Reach"].apply(feet_to_cm)
    bios["Weight_lbs_num"] = bios["Weight"].apply(lbs_to_int)
    bios["DOB_dt"] = pd.to_datetime(bios["DOB"], errors="coerce")
    return bios.set_index("fighter_name")


def default_state():
    return {
        "fight_count": 0,
        "sum_kd": 0.0,
        "sum_opp_kd": 0.0,
        "sum_sig_landed": 0.0,
        "sum_sig_att": 0.0,
        "sum_opp_sig_landed": 0.0,
        "sum_opp_sig_att": 0.0,
        "sum_total_landed": 0.0,
        "sum_total_att": 0.0,
        "sum_opp_total_landed": 0.0,
        "sum_opp_total_att": 0.0,
        "sum_td_landed": 0.0,
        "sum_td_att": 0.0,
        "sum_opp_td_landed": 0.0,
        "sum_opp_td_att": 0.0,
        "sum_sub_att": 0.0,
        "sum_opp_sub_att": 0.0,
        "sum_rev": 0.0,
        "sum_opp_rev": 0.0,
        "sum_ctrl": 0.0,
        "sum_opp_ctrl": 0.0,
        "total_time_fought": 0.0,
        "total_rounds_fought": 0,
        "total_title_bouts": 0,
        "current_win_streak": 0,
        "current_lose_streak": 0,
        "longest_win_streak": 0,
        "wins": 0,
        "losses": 0,
        "draw": 0,
        "win_by_Decision_Majority": 0,
        "win_by_Decision_Split": 0,
        "win_by_Decision_Unanimous": 0,
        "win_by_KO/TKO": 0,
        "win_by_Submission": 0,
        "win_by_TKO_Doctor_Stoppage": 0,
    }


def safe_avg(num, den):
    if den == 0:
        return np.nan
    return num / den


def safe_ratio(num, den):
    if den == 0:
        return np.nan
    return (num / den) * 100.0


def make_prefight_row(prefix, fighter_name, fight_date, state, bios):
    row = {
        f"{prefix}fighter": fighter_name,
        f"{prefix}avg_KD": safe_avg(state["sum_kd"], state["fight_count"]),
        f"{prefix}avg_opp_KD": safe_avg(state["sum_opp_kd"], state["fight_count"]),
        f"{prefix}avg_SIG_STR_pct": safe_ratio(state["sum_sig_landed"], state["sum_sig_att"]),
        f"{prefix}avg_opp_SIG_STR_pct": safe_ratio(state["sum_opp_sig_landed"], state["sum_opp_sig_att"]),
        f"{prefix}avg_TD_pct": safe_ratio(state["sum_td_landed"], state["sum_td_att"]),
        f"{prefix}avg_opp_TD_pct": safe_ratio(state["sum_opp_td_landed"], state["sum_opp_td_att"]),
        f"{prefix}avg_SUB_ATT": safe_avg(state["sum_sub_att"], state["fight_count"]),
        f"{prefix}avg_opp_SUB_ATT": safe_avg(state["sum_opp_sub_att"], state["fight_count"]),
        f"{prefix}avg_REV": safe_avg(state["sum_rev"], state["fight_count"]),
        f"{prefix}avg_opp_REV": safe_avg(state["sum_opp_rev"], state["fight_count"]),
        f"{prefix}avg_SIG_STR_att": safe_avg(state["sum_sig_att"], state["fight_count"]),
        f"{prefix}avg_SIG_STR_landed": safe_avg(state["sum_sig_landed"], state["fight_count"]),
        f"{prefix}avg_opp_SIG_STR_att": safe_avg(state["sum_opp_sig_att"], state["fight_count"]),
        f"{prefix}avg_opp_SIG_STR_landed": safe_avg(state["sum_opp_sig_landed"], state["fight_count"]),
        f"{prefix}avg_TOTAL_STR_att": safe_avg(state["sum_total_att"], state["fight_count"]),
        f"{prefix}avg_TOTAL_STR_landed": safe_avg(state["sum_total_landed"], state["fight_count"]),
        f"{prefix}avg_opp_TOTAL_STR_att": safe_avg(state["sum_opp_total_att"], state["fight_count"]),
        f"{prefix}avg_opp_TOTAL_STR_landed": safe_avg(state["sum_opp_total_landed"], state["fight_count"]),
        f"{prefix}avg_TD_att": safe_avg(state["sum_td_att"], state["fight_count"]),
        f"{prefix}avg_TD_landed": safe_avg(state["sum_td_landed"], state["fight_count"]),
        f"{prefix}avg_opp_TD_att": safe_avg(state["sum_opp_td_att"], state["fight_count"]),
        f"{prefix}avg_opp_TD_landed": safe_avg(state["sum_opp_td_landed"], state["fight_count"]),
        f"{prefix}avg_CTRL_time(seconds)": safe_avg(state["sum_ctrl"], state["fight_count"]),
        f"{prefix}avg_opp_CTRL_time(seconds)": safe_avg(state["sum_opp_ctrl"], state["fight_count"]),
        f"{prefix}total_time_fought(seconds)": state["total_time_fought"],
        f"{prefix}total_rounds_fought": state["total_rounds_fought"],
        f"{prefix}total_title_bouts": state["total_title_bouts"],
        f"{prefix}current_win_streak": state["current_win_streak"],
        f"{prefix}current_lose_streak": state["current_lose_streak"],
        f"{prefix}longest_win_streak": state["longest_win_streak"],
        f"{prefix}wins": state["wins"],
        f"{prefix}losses": state["losses"],
        f"{prefix}draw": state["draw"],
        f"{prefix}win_by_Decision_Majority": state["win_by_Decision_Majority"],
        f"{prefix}win_by_Decision_Split": state["win_by_Decision_Split"],
        f"{prefix}win_by_Decision_Unanimous": state["win_by_Decision_Unanimous"],
        f"{prefix}win_by_KO/TKO": state["win_by_KO/TKO"],
        f"{prefix}win_by_Submission": state["win_by_Submission"],
        f"{prefix}win_by_TKO_Doctor_Stoppage": state["win_by_TKO_Doctor_Stoppage"],
    }

    if fighter_name in bios.index:
        bio = bios.loc[fighter_name]
        row[f"{prefix}Stance"] = bio.get("Stance")
        row[f"{prefix}Height_cms"] = bio.get("Height_cms")
        row[f"{prefix}Reach_cms"] = bio.get("Reach_cms")
        row[f"{prefix}Weight_lbs"] = bio.get("Weight_lbs_num")
        dob = bio.get("DOB_dt")
        if pd.notna(dob):
            row[f"{prefix}age"] = round((fight_date - dob).days / 365.25, 2)
        else:
            row[f"{prefix}age"] = np.nan
    else:
        row[f"{prefix}Stance"] = np.nan
        row[f"{prefix}Height_cms"] = np.nan
        row[f"{prefix}Reach_cms"] = np.nan
        row[f"{prefix}Weight_lbs"] = np.nan
        row[f"{prefix}age"] = np.nan

    return row


def update_state(state, result, method, kd, sig, total, td, sub_att, rev, ctrl, opp_kd, opp_sig, opp_total, opp_td, opp_sub_att, opp_rev, opp_ctrl, rounds, fight_seconds, title_bout):
    sig_landed, sig_att = parse_of(sig)
    opp_sig_landed, opp_sig_att = parse_of(opp_sig)
    total_landed, total_att = parse_of(total)
    opp_total_landed, opp_total_att = parse_of(opp_total)
    td_landed, td_att = parse_of(td)
    opp_td_landed, opp_td_att = parse_of(opp_td)

    state["fight_count"] += 1
    state["sum_kd"] += to_int(kd)
    state["sum_opp_kd"] += to_int(opp_kd)
    state["sum_sig_landed"] += sig_landed
    state["sum_sig_att"] += sig_att
    state["sum_opp_sig_landed"] += opp_sig_landed
    state["sum_opp_sig_att"] += opp_sig_att
    state["sum_total_landed"] += total_landed
    state["sum_total_att"] += total_att
    state["sum_opp_total_landed"] += opp_total_landed
    state["sum_opp_total_att"] += opp_total_att
    state["sum_td_landed"] += td_landed
    state["sum_td_att"] += td_att
    state["sum_opp_td_landed"] += opp_td_landed
    state["sum_opp_td_att"] += opp_td_att
    state["sum_sub_att"] += to_int(sub_att)
    state["sum_opp_sub_att"] += to_int(opp_sub_att)
    state["sum_rev"] += to_int(rev)
    state["sum_opp_rev"] += to_int(opp_rev)
    state["sum_ctrl"] += parse_ctrl_to_seconds(ctrl)
    state["sum_opp_ctrl"] += parse_ctrl_to_seconds(opp_ctrl)
    state["total_time_fought"] += fight_seconds
    state["total_rounds_fought"] += rounds
    state["total_title_bouts"] += int(bool(title_bout))

    if result == "win":
        state["wins"] += 1
        state["current_win_streak"] += 1
        state["current_lose_streak"] = 0
        state["longest_win_streak"] = max(state["longest_win_streak"], state["current_win_streak"])
        method_value = str(method or "")
        if "Decision - Majority" in method_value:
            state["win_by_Decision_Majority"] += 1
        elif "Decision - Split" in method_value:
            state["win_by_Decision_Split"] += 1
        elif "Decision - Unanimous" in method_value or "U-DEC" in method_value:
            state["win_by_Decision_Unanimous"] += 1
        elif "KO" in method_value or "TKO" in method_value:
            if "Doctor" in method_value:
                state["win_by_TKO_Doctor_Stoppage"] += 1
            else:
                state["win_by_KO/TKO"] += 1
        elif "Submission" in method_value or "SUB" in method_value:
            state["win_by_Submission"] += 1
    elif result == "loss":
        state["losses"] += 1
        state["current_lose_streak"] += 1
        state["current_win_streak"] = 0
    else:
        state["draw"] += 1
        state["current_win_streak"] = 0
        state["current_lose_streak"] = 0


def build_main_dataset_from_raw():
    raw = pd.read_csv(RAW_PATH, sep=";")
    raw["parsed_date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.sort_values(["parsed_date", "R_fighter", "B_fighter"]).reset_index(drop=True)

    bios = parse_bio_frame()
    states = defaultdict(default_state)
    rows = []

    for _, fight in raw.iterrows():
        fight_date = fight["parsed_date"]
        title_bout = "title" in str(fight["Fight_type"]).lower()
        weight_class = str(fight["Fight_type"]).replace("Title Bout", "").replace("Bout", "").strip()
        weight_class = re.sub(r"\s+", " ", weight_class) or np.nan

        red_name = fight["R_fighter"]
        blue_name = fight["B_fighter"]
        red_state = states[red_name]
        blue_state = states[blue_name]

        row = {
            "R_fighter": red_name,
            "B_fighter": blue_name,
            "Referee": fight.get("Referee"),
            "date": fight_date.strftime("%Y-%m-%d") if pd.notna(fight_date) else np.nan,
            "location": fight.get("location"),
            "Winner": (
                "Red" if str(fight.get("Winner")) == str(red_name) else
                "Blue" if str(fight.get("Winner")) == str(blue_name) else
                fight.get("Winner")
            ),
            "title_bout": title_bout,
            "weight_class": weight_class,
        }
        row.update(make_prefight_row("R_", red_name, fight_date, red_state, bios))
        row.update(make_prefight_row("B_", blue_name, fight_date, blue_state, bios))
        rows.append(row)

        total_rounds = to_int(fight.get("last_round"))
        time_in_round = parse_ctrl_to_seconds(fight.get("last_round_time"))
        max_round_seconds = 300
        fight_seconds = max(0, (max(total_rounds, 1) - 1) * max_round_seconds + time_in_round)

        winner_value = row["Winner"]
        red_result = "win" if winner_value == "Red" else "loss" if winner_value == "Blue" else "draw"
        blue_result = "win" if winner_value == "Blue" else "loss" if winner_value == "Red" else "draw"

        update_state(
            red_state, red_result, fight.get("win_by"), fight.get("R_KD"), fight.get("R_SIG_STR."), fight.get("R_TOTAL_STR."),
            fight.get("R_TD"), fight.get("R_SUB_ATT"), fight.get("R_REV"), fight.get("R_CTRL"), fight.get("B_KD"),
            fight.get("B_SIG_STR."), fight.get("B_TOTAL_STR."), fight.get("B_TD"), fight.get("B_SUB_ATT"), fight.get("B_REV"),
            fight.get("B_CTRL"), total_rounds, fight_seconds, title_bout
        )
        update_state(
            blue_state, blue_result, fight.get("win_by"), fight.get("B_KD"), fight.get("B_SIG_STR."), fight.get("B_TOTAL_STR."),
            fight.get("B_TD"), fight.get("B_SUB_ATT"), fight.get("B_REV"), fight.get("B_CTRL"), fight.get("R_KD"),
            fight.get("R_SIG_STR."), fight.get("R_TOTAL_STR."), fight.get("R_TD"), fight.get("R_SUB_ATT"), fight.get("R_REV"),
            fight.get("R_CTRL"), total_rounds, fight_seconds, title_bout
        )

    pd.DataFrame(rows).to_csv(MAIN_DATA_PATH, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-date", default=None, help="YYYY-MM-DD; überschreibt den lokalen Cutoff.")
    return parser.parse_args()


def refresh():
    args = parse_args()
    existing_raw = load_existing_raw()
    if args.since_date:
        latest_existing_date = pd.to_datetime(args.since_date).normalize()
        existing_raw = existing_raw[existing_raw["parsed_date"] <= latest_existing_date].copy()
        if SCRAPE_CACHE_PATH.exists():
            SCRAPE_CACHE_PATH.unlink()
        if SCRAPE_PROGRESS_PATH.exists():
            SCRAPE_PROGRESS_PATH.unlink()
    else:
        latest_existing_date = get_latest_existing_date(existing_raw)
    print(f"Lokaler Raw-Datenstand bis: {latest_existing_date.date()}")

    new_fights = scrape_new_fights(latest_existing_date)
    if new_fights.empty:
        print("Keine neuen Kämpfe gefunden.")
    else:
        print(f"Neue Kämpfe gescraped: {len(new_fights)}")
        raw_append = new_fights[[col for col in RAW_COLUMNS if col in new_fights.columns]].copy()
        merged_raw = pd.concat([existing_raw[RAW_COLUMNS], raw_append], ignore_index=True)
        merged_raw = merged_raw.drop_duplicates(subset=["R_fighter", "B_fighter", "date", "Fight_type"], keep="last")
        merged_raw.to_csv(RAW_PATH, sep=";", index=False)

        existing_bios = pd.read_csv(FIGHTER_DETAILS_PATH).drop_duplicates(subset=["fighter_name"], keep="last")
        known = set(existing_bios["fighter_name"].dropna())
        fighter_urls = {}
        for row in new_fights.to_dict("records"):
            if row["R_fighter"] not in known:
                fighter_urls[row["R_fighter"]] = row.get("R_fighter_url")
            if row["B_fighter"] not in known:
                fighter_urls[row["B_fighter"]] = row.get("B_fighter_url")
        new_bios = scrape_fighter_details(fighter_urls)
        if not new_bios.empty:
            merged_bios = pd.concat([existing_bios, new_bios], ignore_index=True)
            merged_bios = merged_bios.drop_duplicates(subset=["fighter_name"], keep="last")
            merged_bios.to_csv(FIGHTER_DETAILS_PATH, index=False)

    build_main_dataset_from_raw()
    clean_and_prepare_data()
    artifacts = train_all_models()
    odds_frame = refresh_bookmaker_odds()

    cleaned = pd.read_csv(DATA_DIR / "cleaned_ufc_data.csv")
    metadata = {
        "refreshed_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "latest_event_date": pd.to_datetime(cleaned["date"], errors="coerce").max().strftime("%Y-%m-%d"),
        "fight_rows": int(len(cleaned)),
        "bookmaker_odds_rows": int(len(odds_frame)),
        "winner_balanced_accuracy": artifacts["summary"]["winner_advanced"]["balanced_accuracy"],
    }
    with open(REFRESH_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    refresh()
