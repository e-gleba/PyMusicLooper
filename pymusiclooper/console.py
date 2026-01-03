from rich.console import Console

rich_console = Console()

# Option group definitions (immutable tuples)
_BASIC_OPTIONS = ("--path", "--url")
_LOOP_OPTIONS = (
    "--min-duration-multiplier",
    "--min-loop-duration",
    "--max-loop-duration",
    "--approx-loop-position",
    "--brute-force",
    "--disable-pruning",
)
_EXPORT_OPTIONS = ("--output-dir", "--format")
_BATCH_OPTIONS = ("--recursive", "--flatten")


def _option_groups(*extra_basic: str) -> list[dict]:
    """Build option groups for CLI help styling."""
    return [
        {"name": "Basic options", "options": [*_BASIC_OPTIONS, *extra_basic]},
        {"name": "Advanced loop options", "options": list(_LOOP_OPTIONS)},
        {"name": "Export options", "options": list(_EXPORT_OPTIONS)},
        {"name": "Batch options", "options": list(_BATCH_OPTIONS)},
    ]


_OPTION_GROUPS = {
    "pymusiclooper play": _option_groups(),
    "pymusiclooper split-audio": _option_groups(),
    "pymusiclooper tag": _option_groups("--tag-names", "--tag-offset"),
    "pymusiclooper export-points": _option_groups(
        "--export-to", "--alt-export-top", "--fmt"
    ),
    "pymusiclooper extend": _option_groups(
        "--extended-length", "--fade-length", "--disable-fade-out"
    ),
}

_COMMAND_GROUPS = {
    "pymusiclooper": [
        {"name": "Play Commands", "commands": ["play", "play-tagged"]},
        {
            "name": "Export Commands",
            "commands": ["export-points", "split-audio", "tag", "extend"],
        },
    ]
}
