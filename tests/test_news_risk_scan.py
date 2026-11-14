"""The headline filter is tested on a small hand-written feed, so no network is needed."""
from news_risk_scan import find_risk_zones

FEED = b"""<?xml version="1.0"?>
<rss><channel>
  <item><title>Road caved in near Chennai metro construction site</title><pubDate>Mon, 09 Nov 2026</pubDate></item>
  <item><title>Sinkhole swallows car in Pune after pipeline burst</title><pubDate>Tue, 10 Nov 2026</pubDate></item>
  <item><title>Cricket: Mumbai win the final</title><pubDate>Tue, 10 Nov 2026</pubDate></item>
  <item><title>Excavation work blamed for cracks in Jaipur homes</title><pubDate>Wed, 11 Nov 2026</pubDate></item>
  <item><title>Leak repaired on Delhi-Noida water line</title><pubDate>Thu, 12 Nov 2026</pubDate></item>
</channel></rss>"""


def test_keeps_headlines_with_a_hazard_and_a_watched_city():
    hits = find_risk_zones(FEED)
    assert [(city, date) for city, _, date in hits] == [
        ('Chennai', 'Mon, 09 Nov 2026'),
        ('Pune', 'Tue, 10 Nov 2026'),
        ('Delhi', 'Thu, 12 Nov 2026'),      # one headline can name two cities
        ('Noida', 'Thu, 12 Nov 2026'),
    ]


def test_ignores_city_without_hazard_and_hazard_without_city():
    cities = {city for city, _, _ in find_risk_zones(FEED)}
    assert 'Mumbai' not in cities            # a city, but sports news
    assert 'Jaipur' not in cities            # a hazard, but not a watched city


def test_max_items_limits_how_many_headlines_are_read():
    assert [city for city, _, _ in find_risk_zones(FEED, max_items=1)] == ['Chennai']


def test_words_are_matched_whole():
    feed = b"<rss><channel><item><title>Metropolitan budget for Salem announced</title></item></channel></rss>"
    assert find_risk_zones(feed) == []       # "Metropolitan" is not "metro"
