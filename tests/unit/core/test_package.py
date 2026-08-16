"""Phase 0: Smoke tests — verifies the package installs and basic metadata."""
import ahf


def test_package_version_exists():
    assert hasattr(ahf, "__version__")


def test_package_version_format():
    parts = ahf.__version__.split(".")
    assert len(parts) >= 3, f"Version should be semver, got: {ahf.__version__}"


def test_package_imports():
    """All top-level subpackages should import without error."""
    import ahf.adapters  # noqa: F401
    import ahf.core  # noqa: F401
    import ahf.domain  # noqa: F401
    import ahf.entrypoints  # noqa: F401
    import ahf.rl  # noqa: F401
    import ahf.signals  # noqa: F401
    import ahf.utils  # noqa: F401
