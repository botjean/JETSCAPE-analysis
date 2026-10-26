"""NEW review-only exclusions; all historical test sources remain byte-exact."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        reason = None
        if item.name == "test_actual_event1483_with_calibrated_transport":
            reason = "Raw event integration is intentionally outside this source package"
        elif item.name == "test_actual_known_failure_and_unchanged_old_successes":
            reason = "Requires omitted raw-event fixture and executed native runtime"
        elif "test_seed_scaling.py" in str(item.path) and "captured" in item.fixturenames:
            reason = "Captured event-derived seed file is not redistributed"
        elif "test_precision.py" in str(item.path) and "rows" in item.fixturenames:
            reason = "Requires omitted raw-event precision fixture"
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
