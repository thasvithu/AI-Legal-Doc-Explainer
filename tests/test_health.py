from src.utils.health import run_health_check


def test_health_check_basic():
    report = run_health_check()
    assert "components" in report
    assert isinstance(report["ok"], bool)
