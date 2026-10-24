"""Loud accounting for the integration lane.

The default run (``pytest tests/``) deselects ``integration``-marked tests to
hold the 30s-per-100-tests budget. Deselection must never be silent: every
default run states, in its own summary, how many integration tests did not
run and the command that runs them. The integration lane is a mandatory
pre-fire staging step, so a green default run is necessary, not sufficient.
"""


def pytest_collection_modifyitems(config, items):
    config._integration_count = sum(1 for item in items if item.get_closest_marker("integration"))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    count = getattr(config, "_integration_count", 0)
    if count and config.option.markexpr != "integration":
        terminalreporter.section("integration lane not run", red=True, bold=True)
        terminalreporter.line(
            f"{count} integration test(s) deselected by the default marker "
            "filter. Run `pytest -m integration` before staging any fire."
        )
