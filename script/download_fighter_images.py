"""
UFC Fighter Image Downloader v3
Nutzt Wikipedia Pageimages API - holt das Hauptfoto jeder Wikipedia-Seite.
Ausführen: python3 download_fighter_images.py
"""

import os
import json
import requests
from pathlib import Path
import time

SAVE_DIR = Path(__file__).parent.parent / "data" / "fighter_images"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UFCBot/1.0)"}

# "Dataset-Name": "Wikipedia-Seitentitel" (genauer Titel der Wikipedia-Seite)
FIGHTERS = {
    "Ilia Topuria":          "Ilia Topuria",
    "Charles Oliveira":      "Charles Oliveira",
    "Islam Makhachev":       "Islam Makhachev",
    "Jon Jones":             "Jon Jones (fighter)",
    "Alex Pereira":          "Alex Pereira (fighter)",
    "Conor McGregor":        "Conor McGregor",
    "Khabib Nurmagomedov":   "Khabib Nurmagomedov",
    "Max Holloway":          "Max Holloway (fighter)",
    "Dustin Poirier":        "Dustin Poirier",
    "Justin Gaethje":        "Justin Gaethje",
    "Stipe Miocic":          "Stipe Miocic",
    "Francis Ngannou":       "Francis Ngannou",
    "Kamaru Usman":          "Kamaru Usman",
    "Israel Adesanya":       "Israel Adesanya",
    "Robert Whittaker":      "Robert Whittaker (fighter)",
    "Valentina Shevchenko":  "Valentina Shevchenko (fighter)",
    "Amanda Nunes":          "Amanda Nunes",
    "Nate Diaz":             "Nate Diaz",
    "Jorge Masvidal":        "Jorge Masvidal",
    "Michael Chandler":      "Michael Chandler (fighter)",
    "Leon Edwards":          "Leon Edwards (fighter)",
    "Sean Strickland":       "Sean Strickland",
    "Tom Aspinall":          "Tom Aspinall",
    "Paddy Pimblett":        "Paddy Pimblett",
    "Colby Covington":       "Colby Covington",
    "Sean O'Malley":         "Sean O'Malley (fighter)",
    "Dricus Du Plessis":     "Dricus du Plessis",
    "Anthony Pettis":        "Anthony Pettis",
    "Yoel Romero":           "Yoel Romero",
    "Cain Velasquez":        "Cain Velasquez",
}

def get_image_url(wiki_title: str) -> str | None:
    """Wikipedia Pageimages API → gibt URL des Hauptfotos zurück."""
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": wiki_title,
        "prop": "pageimages",
        "pithumbsize": 500,
        "format": "json",
        "redirects": 1,
    }
    try:
        r = requests.get(api, params=params, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                thumb = page.get("thumbnail", {})
                if thumb:
                    return thumb.get("source")
    except Exception as e:
        print(f"  Fehler: {e}")
    return None


def download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and len(r.content) > 5000:
            ct = r.headers.get("Content-Type", "")
            if "image" in ct or url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                dest.write_bytes(r.content)
                return True
    except Exception as e:
        print(f"  Download-Fehler: {e}")
    return False


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("'", "").replace("/", "_").lower() + ".jpg"


def main():
    # Bereits vorhandene laden
    json_path = SAVE_DIR / "fighter_photos.json"
    results = {}
    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)

    print(f"📂 Ziel: {SAVE_DIR}")
    print(f"🔍 Lade Bilder für {len(FIGHTERS)} Kämpfer ...\n")

    for name, wiki_title in FIGHTERS.items():
        fname = sanitize(name)
        dest  = SAVE_DIR / fname
        rel   = f"../data/fighter_images/{fname}"

        if dest.exists() and dest.stat().st_size > 5000:
            print(f"  ✅ {name}: vorhanden")
            results[name] = rel
            continue

        print(f"  🔎 {name} ...", end=" ", flush=True)
        url = get_image_url(wiki_title)

        if url:
            ok = download(url, dest)
            if ok:
                kb = dest.stat().st_size // 1024
                print(f"✅ {kb} KB  [{url[:60]}...]")
                results[name] = rel
            else:
                print(f"❌ Download fehlgeschlagen  (URL: {url[:60]})")
        else:
            print(f"❌ kein Bild auf Wikipedia")

        time.sleep(0.2)

    # JSON speichern
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    total = len(FIGHTERS)
    found = len(results)
    print(f"\n📄 Mapping gespeichert: {json_path}")
    print(f"✅ {found}/{total} Bilder erfolgreich gespeichert.")
    if found < total:
        missing = [n for n in FIGHTERS if n not in results]
        print(f"❌ Fehlend: {', '.join(missing)}")


if __name__ == "__main__":
    main()
