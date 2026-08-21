from datetime import datetime, timezone

from ai_trend_radar.models import Item, Trend
from ai_trend_radar.trends import arrange_must_reads


def item(uid: str, source: str, kind: str) -> Item:
    return Item(
        uid=uid,
        source=source,
        kind=kind,
        title=uid,
        url=f"https://example.com/{uid}",
        published_at=datetime.now(timezone.utc),
    )


def test_must_reads_stay_on_topic_and_prefer_independent_evidence_family():
    paper_a = item("paper-a", "Hugging Face Papers", "paper")
    paper_b = item("paper-b", "arXiv", "paper")
    release = item("release", "GitHub Releases", "release")
    unrelated_blog = item("unrelated", "Vendor Blog", "blog")
    trend = Trend(
        1,
        "Serving",
        3.0,
        0.4,
        4,
        8,
        3,
        [paper_a, paper_b, release, unrelated_blog],
        relevant_urls=[paper_a.url, paper_b.url, release.url],
    )

    selected = arrange_must_reads(trend, 2)

    assert selected == [paper_a, release]
    assert trend.items[:2] == selected
    assert unrelated_blog.metadata["selected_must_read"] is False


def test_source_diversity_never_admits_an_off_topic_item():
    paper_a = item("paper-a", "Hugging Face Papers", "paper")
    paper_b = item("paper-b", "arXiv", "paper")
    unrelated_release = item("unrelated", "GitHub Releases", "release")
    trend = Trend(
        2,
        "Quantization",
        3.0,
        0.4,
        3,
        6,
        3,
        [paper_a, paper_b, unrelated_release],
        relevant_urls=[paper_a.url, paper_b.url],
    )

    assert arrange_must_reads(trend, 2) == [paper_a, paper_b]
