from factories import blurred, darkened, metrics_for, noisy, product_image

from vqa.quality.scoring import (
    DEFAULT_CONFIG,
    build_report,
    composite_score,
    detect_issues,
    technical_score,
    verdict_for,
)
from vqa.types import Issue, SemanticAttributes


def _codes(report):
    return {issue.code for issue in report.issues}


def test_clean_studio_shot_passes_without_issues():
    report = build_report(metrics_for(product_image()))
    assert report.verdict == "pass"
    assert report.issues == []
    assert report.score > 80


def test_blurred_shot_is_flagged_and_fails():
    report = build_report(metrics_for(blurred(product_image())))
    assert "blurry" in _codes(report)
    assert report.verdict == "fail"


def test_dark_shot_is_flagged_as_underexposed():
    assert "underexposed" in _codes(build_report(metrics_for(darkened(product_image()))))


def test_noisy_shot_is_flagged_but_not_double_counted_as_clutter():
    codes = _codes(build_report(metrics_for(noisy(product_image()))))
    assert "noisy" in codes
    assert "cluttered_background" not in codes


def test_low_resolution_is_reported_on_the_original_dimensions():
    assert "low_resolution" in _codes(build_report(metrics_for(product_image(size=320))))


def test_every_issue_carries_an_actionable_remedy():
    report = build_report(metrics_for(blurred(product_image(size=300))))
    assert report.issues
    for issue in report.issues:
        assert issue.remedy and issue.message
        assert issue.severity in {"low", "medium", "high"}


def test_issues_are_sorted_by_severity():
    report = build_report(metrics_for(darkened(blurred(product_image(size=300)))))
    order = [issue.severity for issue in report.issues]
    assert order == sorted(order, key=lambda s: {"high": 0, "medium": 1, "low": 2}[s])


def test_worst_subscore_drags_the_technical_score_down():
    clean = metrics_for(product_image())
    blurry = metrics_for(blurred(product_image()))
    assert technical_score(blurry) < technical_score(clean) - 0.25


def test_verdict_rules():
    high = [Issue("blurry", "high", "m", "r")]
    low = [Issue("off_center", "low", "m", "r")]
    assert verdict_for(95.0, [], DEFAULT_CONFIG) == "pass"
    assert verdict_for(95.0, high, DEFAULT_CONFIG) == "fail"
    assert verdict_for(95.0, low, DEFAULT_CONFIG) == "review"
    assert verdict_for(30.0, [], DEFAULT_CONFIG) == "fail"


def test_semantic_signal_shifts_the_composite_score():
    metrics = metrics_for(product_image())
    good = SemanticAttributes(model="t", scores={
        "in_focus": 0.99, "professional_lighting": 0.98, "clean_background": 0.99,
        "well_framed": 0.97, "studio_quality": 0.98,
    })
    bad = SemanticAttributes(model="t", scores={k: 0.02 for k in good.scores})
    assert composite_score(metrics, good) > composite_score(metrics, bad)


def test_score_is_bounded():
    for image in (product_image(), blurred(product_image()), darkened(product_image())):
        score = build_report(metrics_for(image)).score
        assert 0.0 <= score <= 100.0


def test_detect_issues_is_pure():
    metrics = metrics_for(blurred(product_image()))
    assert [i.code for i in detect_issues(metrics)] == [i.code for i in detect_issues(metrics)]
