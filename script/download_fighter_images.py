"""
UFC Fighter Image Downloader

- Lädt Bilder für alle Fighter aus dem Datensatz.
- Nutzt Wikipedia/Wikimedia als frei zugängliche Quelle.
- Verwendet manuelle Titel-Overrides für bekannte Problemfälle.

Ausführen:
    python3 script/download_fighter_images.py
"""

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_ufc_data.csv"
SAVE_DIR = BASE_DIR / "data" / "fighter_images"
JSON_PATH = SAVE_DIR / "fighter_photos.json"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFCBot/2.0)"}

# Bekannte Wikipedia-Titel, die nicht sauber aus dem Namen ableitbar sind.
TITLE_OVERRIDES = {
    "Jon Jones": "Jon Jones (fighter)",
    "Alex Pereira": "Alex Pereira (fighter)",
    "Max Holloway": "Max Holloway (fighter)",
    "Robert Whittaker": "Robert Whittaker (fighter)",
    "Valentina Shevchenko": "Valentina Shevchenko (fighter)",
    "Amanda Nunes": "Amanda Nunes",
    "Justin Gaethje": "Justin Gaethje",
    "Michael Chandler": "Michael Chandler (fighter)",
    "Leon Edwards": "Leon Edwards (fighter)",
    "Sean Strickland": "Sean Strickland",
    "Tom Aspinall": "Tom Aspinall",
    "Paddy Pimblett": "Paddy Pimblett",
    "Colby Covington": "Colby Covington",
    "Sean O'Malley": "Sean O'Malley",
    "Dricus Du Plessis": "Dricus du Plessis",
    "Yoel Romero": "Yoel Romero",
    "Cain Velasquez": "Cain Velasquez",
}


def sanitize(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("'", "")
        .replace("/", "_")
        .replace(".", "")
        .lower()
        + ".jpg"
    )


def load_fighter_names() -> list[str]:
    df = pd.read_csv(DATA_PATH, usecols=["R_fighter", "B_fighter"])
    fighters = set(df["R_fighter"].dropna()).union(set(df["B_fighter"].dropna()))
    return sorted(name.strip() for name in fighters if isinstance(name, str) and name.strip())


def request_json(url: str, params: dict | None = None) -> dict | None:
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


def get_page_image_url(title: str) -> str | None:
    data = request_json(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "pithumbsize": 600,
            "format": "json",
            "redirects": 1,
        },
    )
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        thumb = page.get("thumbnail", {})
        source = thumb.get("source")
        if source:
            return source
    return None


def get_summary_image_url(title: str) -> str | None:
    data = request_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
    if not data:
        return None
    thumb = data.get("thumbnail", {})
    source = thumb.get("source")
    if source:
        return source
    original = data.get("originalimage", {})
    return original.get("source")


def get_html_image_url(title: str) -> str | None:
    url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code != 200:
            return None
        html = response.text

        og_match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        if og_match:
            return og_match.group(1)

        infobox_match = re.search(r'<a class="mw-file-description" href="([^"]+)"', html)
        if infobox_match:
            href = infobox_match.group(1)
            if href.startswith("//"):
                return "https:" + href
            if href.startswith("/"):
                return "https://en.wikipedia.org" + href
            return href
    except Exception:
        return None
    return None


def search_titles(query: str, limit: int = 8) -> list[str]:
    data = request_json(
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        },
    )
    if not data:
        return []
    return [item.get("title") for item in data.get("query", {}).get("search", []) if item.get("title")]


def build_title_candidates(name: str) -> list[str]:
    candidates = []
    if name in TITLE_OVERRIDES:
        candidates.append(TITLE_OVERRIDES[name])

    candidates.extend(
        [
            name,
            f"{name} (fighter)",
            f"{name} (mixed martial artist)",
            f"{name} (martial artist)",
        ]
    )

    search_queries = [
        f'intitle:"{name}" UFC',
        f'"{name}" mixed martial artist',
        f'"{name}" fighter',
        name,
    ]
    for query in search_queries:
        candidates.extend(search_titles(query))

    # Duplikate entfernen, Reihenfolge beibehalten.
    seen = set()
    unique = []
    for title in candidates:
        if title not in seen:
            seen.add(title)
            unique.append(title)
    return unique


def resolve_image_url(name: str) -> str | None:
    for title in build_title_candidates(name):
        url = get_page_image_url(title)
        if url:
            return url
        url = get_summary_image_url(title)
        if url:
            return url
        url = get_html_image_url(title)
        if url:
            return url
    return None


def download_image(url: str, destination: Path) -> bool:
    try:
        response = requests.get(url, headers=HEADERS, timeout=20)
        if response.status_code != 200 or len(response.content) < 5000:
            return False
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            return False
        destination.write_bytes(response.content)
        return True
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--names", nargs="*", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    fighters = args.names if args.names else load_fighter_names()
    if args.limit:
        fighters = fighters[: args.limit]
    results = {}
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as handle:
            results = json.load(handle)

    print(f"Zielordner: {SAVE_DIR}")
    print(f"Fighter im Datensatz: {len(fighters)}")
    print(f"Bereits gemappt: {len(results)}\n")

    success_count = 0
    skipped_count = 0
    failed = []

    for index, name in enumerate(fighters, start=1):
        filename = sanitize(name)
        destination = SAVE_DIR / filename
        rel_path = f"../data/fighter_images/{filename}"

        if destination.exists() and destination.stat().st_size > 5000:
            results[name] = rel_path
            skipped_count += 1
            continue

        print(f"[{index}/{len(fighters)}] {name} ... ", end="", flush=True)
        url = resolve_image_url(name)
        if url and download_image(url, destination):
            results[name] = rel_path
            success_count += 1
            print("OK")
        else:
            failed.append(name)
            print("kein Bild")

        if index % 25 == 0:
            with open(JSON_PATH, "w", encoding="utf-8") as handle:
                json.dump(results, handle, indent=2, ensure_ascii=False)
        time.sleep(0.15)

    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    print("\nFertig.")
    print(f"Neu geladen: {success_count}")
    print(f"Vorhanden übersprungen: {skipped_count}")
    print(f"Gemappt gesamt: {len(results)}")
    print(f"Ohne Bild: {len(failed)}")
    if failed:
        print("Beispiele ohne Bild:", ", ".join(failed[:25]))


if __name__ == "__main__":
    main()
