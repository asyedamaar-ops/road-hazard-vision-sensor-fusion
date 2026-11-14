"""Optional context step: list Indian cities with recent road-collapse or excavation news.

    python news_risk_scan.py

Reads the Google News RSS feed for the last 14 days and keeps headlines that
mention both a hazard word and one of the watched cities. The output is a list
of "risk zones". It is a standalone script and is NOT connected to the camera
and accelerometer loop.

Written by my teammate Mushir (original: india_risk_scraper.py in
github.com/mushir2004/Sinkhole_Detection_Project). This version only separates
the download from the headline filter, so the filter can be tested offline.
"""
import re
import urllib.request
import xml.etree.ElementTree as ET

FEED_URL = ('https://news.google.com/rss/search?q=road+collapse+OR+sinkhole+OR+pipeline+burst+OR+'
            'metro+construction+OR+excavation+in+india+when:14d&hl=en-IN&gl=IN&ceid=IN:en')

CITIES = [
    'Chennai', 'Coimbatore', 'Madurai', 'Trichy', 'Salem',
    'Bengaluru', 'Mumbai', 'Delhi', 'Hyderabad', 'Pune',
    'Kolkata', 'Ahmedabad', 'Gurugram', 'Noida',
]

# hazards and active underground work
HAZARD_WORDS = [
    'collapse', 'sinkhole', 'caved', 'crater', 'burst', 'leak', 'crack',
    'construction', 'metro', 'excavation', 'digging', 'sewage',
]


def _mentions(word, text):
    return re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE) is not None


def find_risk_zones(rss_xml, max_items=20):
    """Return [(city, headline, date)] for headlines that name a hazard and a watched city."""
    root = ET.fromstring(rss_xml)
    hits = []
    for item in root.findall('.//item')[:max_items]:
        title = item.findtext('title') or ''
        date = item.findtext('pubDate') or ''
        if not any(_mentions(word, title) for word in HAZARD_WORDS):
            continue
        for city in CITIES:
            if _mentions(city, title):
                hits.append((city, title, date))
    return hits


def fetch_feed(url=FEED_URL, timeout=15):
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def main():
    try:
        hits = find_risk_zones(fetch_feed())
    except Exception as exc:
        raise SystemExit(f'Could not read the news feed: {exc}')

    for city, title, date in hits:
        print(f'{city}: {title} ({date})')
    zones = sorted({city for city, _, _ in hits})
    print(f'Active risk zones: {zones}')


if __name__ == '__main__':
    main()
