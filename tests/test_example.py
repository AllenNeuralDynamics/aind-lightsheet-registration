"""Example test module."""

import aind_lightsheet_registration


def test_version():
    """Test that version is defined."""
    assert aind_lightsheet_registration.__version__ is not None
    assert isinstance(aind_lightsheet_registration.__version__, str)
