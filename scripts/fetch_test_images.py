"""Download a small set of CC-licensed portraits from Wikimedia Commons.

These images are local evaluation fixtures only. They are not redistributed with
the package (``data/`` is git-ignored except for the attribution file).

Usage::

    python scripts/fetch_test_images.py [--dest data/input]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

UA = "portrait-engine-test-fixtures/0.1 (local research use)"
API = "https://commons.wikimedia.org/w/api.php"

# Chosen for: real photographs, single visible face, frontal/semi-frontal,
# varied skin tone, age and lighting, and a couple with visible teeth.
TITLES = [
    "File:Closeup of a young blonde girl face looking up.jpg",
    "File:Woman (Imagicity 820).jpg",
    "File:Young Man (Imagicity 823).jpg",
    "File:Laughing woman with teeth.jpg",
    "File:Grinning Laotian showing teeth.jpg",
    "File:Old Man (Imagicity 231).jpg",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


def _strip_html(value: str | None) -> str:
    return re.sub("<[^>]+>", "", value or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/input")
    ap.add_argument("--width", type=int, default=1400, help="max thumbnail width")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": "|".join(TITLES),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": str(args.width),
        }
    )
    pages = json.loads(_get(f"{API}?{query}"))["query"]["pages"]

    rows = []
    for page in pages.values():
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        title = page["title"][len("File:") :]
        slug = re.sub(r"[^a-z0-9]+", "_", title.rsplit(".", 1)[0].lower()).strip("_") + ".jpg"
        dest = os.path.join(args.dest, slug)
        if not os.path.exists(dest):
            with open(dest, "wb") as handle:
                handle.write(_get(info.get("thumburl") or info["url"]))
            time.sleep(1.5)
        rows.append(
            (
                slug,
                title,
                _strip_html(meta.get("LicenseShortName", {}).get("value")),
                _strip_html(meta.get("Artist", {}).get("value")),
                "https://commons.wikimedia.org/wiki/"
                + urllib.parse.quote(page["title"].replace(" ", "_")),
            )
        )
        print(f"{dest}  ({os.path.getsize(dest) // 1024} KB)")

    rows.sort()
    with open(os.path.join(args.dest, "ATTRIBUTION.md"), "w") as handle:
        handle.write(
            "# Test image attribution\n\n"
            "All test portraits come from Wikimedia Commons and are used only as local\n"
            "evaluation fixtures. They are not redistributed as part of this package.\n"
            "Regenerate with `python scripts/fetch_test_images.py`.\n\n"
            "| file | source page | license | author |\n|---|---|---|---|\n"
        )
        for slug, title, lic, artist, url in rows:
            handle.write(f"| `{slug}` | [{title}]({url}) | {lic} | {artist} |\n")


if __name__ == "__main__":
    main()
