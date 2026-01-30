"""
ThinkRL CLI
===========

Command-line interface for ThinkRL.

Author: EllanorAI
"""


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running as module."""
    if name == "app":
        from thinkrl.cli.main import app

        return app
    elif name == "main":
        from thinkrl.cli.main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app", "main"]
