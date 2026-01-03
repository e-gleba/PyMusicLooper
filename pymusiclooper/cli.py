import functools
import logging
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

import rich_click as click
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.traceback import install as rich_traceback_handler
from rich_click.patch import patch as rich_click_patch

rich_click_patch()
from click_option_group import RequiredMutuallyExclusiveOptionGroup, optgroup
from click_params import URL as UrlParamType
from yt_dlp.utils import YoutubeDLError

from pymusiclooper import __version__
from pymusiclooper.console import _COMMAND_GROUPS, _OPTION_GROUPS, rich_console
from pymusiclooper.core import MusicLooper
from pymusiclooper.exceptions import AudioLoadError, LoopNotFoundError
from pymusiclooper.handler import BatchHandler, LoopExportHandler, LoopHandler
from pymusiclooper.utils import download_audio, get_outputdir, mk_outputdir

# CLI styling
click.rich_click.OPTION_GROUPS = _OPTION_GROUPS
click.rich_click.COMMAND_GROUPS = _COMMAND_GROUPS
click.rich_click.USE_RICH_MARKUP = True


# Environment variable helpers
def _env_flag(name: str) -> bool:
    return os.getenv(name) is not None


def _set_env(*names: str) -> None:
    for name in names:
        os.environ[name] = "1"


@contextmanager
def _processing_spinner():
    """Context manager for processing spinner."""
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=rich_console,
        transient=True,
    ) as progress:
        progress.add_task("Processing", total=None)
        yield


def _handle_errors(func):
    """Decorator for consistent error handling across commands."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except YoutubeDLError:
            pass  # Already logged
        except (AudioLoadError, LoopNotFoundError, Exception) as e:
            if _env_flag("PML_DEBUG"):
                rich_console.print_exception(suppress=[click])
            else:
                logging.error(e)

    return wrapper


def _resolve_audio_path(kwargs: dict, output_dir_required: bool = False) -> None:
    """Resolve URL to local path if needed, setup output directory."""
    if kwargs.get("url"):
        if output_dir_required:
            kwargs["output_dir"] = mk_outputdir(Path.cwd(), kwargs.get("output_dir"))
            kwargs["path"] = download_audio(kwargs["url"], kwargs["output_dir"])
        else:
            kwargs["path"] = download_audio(kwargs["url"], tempfile.gettempdir())
    elif output_dir_required:
        kwargs["output_dir"] = get_outputdir(kwargs["path"], kwargs.get("output_dir"))


# Composable option decorators
def common_path_options(f):
    decorators = [
        optgroup.group(
            "audio path",
            cls=RequiredMutuallyExclusiveOptionGroup,
            help="the path to the audio track(s) to load",
        ),
        optgroup.option(
            "--path",
            type=click.Path(exists=True),
            default=None,
            help=r"Path to the audio file(s). [dim cyan]\[mutually exclusive with --url][/]",
        ),
        optgroup.option(
            "--url",
            type=UrlParamType,
            default=None,
            help=r"Link to the youtube video or yt-dlp supported stream. [dim cyan]\[mutually exclusive with --path][/]",
        ),
    ]
    return functools.reduce(lambda fn, dec: dec(fn), reversed(decorators), f)


def common_loop_options(f):
    decorators = [
        click.option(
            "--min-duration-multiplier",
            type=click.FloatRange(0, 1, min_open=True, max_open=True),
            default=0.35,
            show_default=True,
            help="Minimum loop duration as multiplier of total duration.",
        ),
        click.option(
            "--min-loop-duration",
            type=click.FloatRange(min=0, min_open=True),
            default=None,
            help="Minimum loop duration in seconds. [dim](overrides --min-duration-multiplier)[/]",
        ),
        click.option(
            "--max-loop-duration",
            type=click.FloatRange(min=0, min_open=True),
            default=None,
            help="Maximum loop duration in seconds.",
        ),
        click.option(
            "--approx-loop-position",
            type=click.FloatRange(min=0),
            nargs=2,
            default=None,
            help="Approximate loop start and end in seconds. [dim]([cyan]+/-2s[/] search window)[/]",
        ),
        click.option(
            "--brute-force",
            is_flag=True,
            default=False,
            help=r"Check entire track instead of detected beats. [dim yellow](Warning: slow)[/]",
        ),
        click.option(
            "--disable-pruning",
            is_flag=True,
            default=False,
            help="Disable filtering of initial loop points.",
        ),
    ]
    return functools.reduce(lambda fn, dec: dec(fn), reversed(decorators), f)


def common_export_options(f):
    decorators = [
        click.option(
            "--output-dir",
            "-o",
            type=click.Path(exists=False, writable=True, file_okay=False),
            help="Output directory for exported files.",
        ),
        click.option(
            "--recursive",
            "-r",
            is_flag=True,
            default=False,
            help="Process directories recursively.",
        ),
        click.option(
            "--flatten",
            "-f",
            is_flag=True,
            default=False,
            help="Flatten output directory structure. [dim yellow](Warning: overwrites duplicates)[/]",
        ),
    ]
    return functools.reduce(lambda fn, dec: dec(fn), reversed(decorators), f)


@click.group("pymusiclooper", epilog="Docs: https://github.com/arkrow/PyMusicLooper")
@click.option("--debug", "-d", is_flag=True, help="Enable debug mode.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option(
    "--interactive", "-i", is_flag=True, help="Enable interactive loop selection."
)
@click.option(
    "--samples",
    "-s",
    is_flag=True,
    help="Display loop points in samples instead of mm:ss.sss.",
)
@click.version_option(__version__, prog_name="pymusiclooper")
def cli_main(debug, verbose, interactive, samples):
    """Automatically find and use loop points for seamless music repetition."""
    if debug:
        _set_env("PML_DEBUG")
        warnings.simplefilter("default")
        rich_traceback_handler(console=rich_console, suppress=[click])
    else:
        warnings.filterwarnings("ignore")

    if verbose:
        _set_env("PML_VERBOSE")
    if interactive:
        _set_env("PML_INTERACTIVE_MODE")
    if samples:
        _set_env("PML_DISPLAY_SAMPLES")

    level = logging.INFO if verbose else logging.ERROR
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[
            RichHandler(
                level=level,
                console=rich_console,
                rich_tracebacks=True,
                show_path=debug,
                show_time=False,
                tracebacks_suppress=[click],
            )
        ],
    )


@cli_main.command()
@common_path_options
@common_loop_options
@_handle_errors
def play(**kwargs):
    """Play audio on repeat with best discovered loop points."""
    _resolve_audio_path(kwargs)

    with _processing_spinner():
        handler = LoopHandler(**kwargs)

    interactive = _env_flag("PML_INTERACTIVE_MODE")
    pair = handler.choose_loop_pair(interactive_mode=interactive)

    start_fmt = handler.format_time(pair.loop_start)
    end_fmt = handler.format_time(pair.loop_end)

    rich_console.print(
        f"\nLooping from [green]{end_fmt}[/] → [green]{start_fmt}[/]; similarity: {pair.score:.2%}"
    )
    rich_console.print("(Press [red]Ctrl+C[/] to stop)")

    handler.play_looping(pair.loop_start, pair.loop_end)


@cli_main.command()
@click.option(
    "--path", type=click.Path(exists=True), required=True, help="Path to audio file."
)
@click.option(
    "--tag-names",
    type=str,
    nargs=2,
    default=None,
    help="Loop metadata tag names, e.g. --tag-names LOOP_START LOOP_END",
)
@click.option(
    "--tag-offset/--no-tag-offset",
    default=None,
    help="Parse second tag as relative/absolute length. Default: auto-detect.",
)
@_handle_errors
def play_tagged(path, tag_names, tag_offset):
    """Play audio using loop points from file metadata tags."""
    looper = MusicLooper(path)
    start_tag, end_tag = tag_names or (None, None)
    loop_start, loop_end = looper.read_tags(start_tag, end_tag, tag_offset)

    fmt = (lambda s: s) if _env_flag("PML_DISPLAY_SAMPLES") else looper.samples_to_ftime

    rich_console.print(
        f"\nLooping from [green]{fmt(loop_end)}[/] → [green]{fmt(loop_start)}[/]"
    )
    rich_console.print("(Press [red]Ctrl+C[/] to stop)")

    looper.play_looping(loop_start, loop_end)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option(
    "--format",
    type=click.Choice(("WAV", "FLAC", "OGG", "MP3"), case_sensitive=False),
    default="WAV",
    show_default=True,
    help="Audio format for exported files.",
)
@_handle_errors
def split_audio(**kwargs):
    """Split audio into intro, loop, and outro sections."""
    kwargs["split_audio"] = True
    _run_export_handler(**kwargs)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option(
    "--format",
    type=click.Choice(("WAV", "FLAC", "OGG", "MP3"), case_sensitive=False),
    default="MP3",
    show_default=True,
    help="Audio format for output file.",
)
@click.option(
    "--extended-length",
    type=float,
    required=True,
    help="Desired extended length in seconds.",
)
@click.option(
    "--fade-length",
    type=float,
    default=5,
    show_default=True,
    help="Fade out length in seconds.",
)
@click.option(
    "--disable-fade-out",
    is_flag=True,
    default=False,
    help="Include full outro without fading.",
)
@_handle_errors
def extend(**kwargs):
    """Create extended version by looping to specific length."""
    _run_export_handler(**kwargs)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option(
    "--export-to",
    type=click.Choice(("STDOUT", "TXT"), case_sensitive=False),
    default="STDOUT",
    show_default=True,
    help="Export destination.",
)
@click.option(
    "--fmt",
    type=click.Choice(("SAMPLES", "SECONDS", "TIME"), case_sensitive=False),
    default="SAMPLES",
    show_default=True,
    help="Loop point format.",
)
@click.option(
    "--alt-export-top", type=int, default=0, help="Export top N points (-1 for all)."
)
@_handle_errors
def export_points(**kwargs):
    """Export loop points to terminal or text file."""
    export_to = kwargs.pop("export_to", "STDOUT").upper()
    kwargs["to_stdout"] = export_to == "STDOUT"
    kwargs["to_txt"] = export_to == "TXT"
    _run_export_handler(**kwargs)


@cli_main.command()
@common_path_options
@common_loop_options
@common_export_options
@click.option(
    "--tag-names",
    type=str,
    required=True,
    nargs=2,
    help="Loop metadata tag names, e.g. --tag-names LOOP_START LOOP_END",
)
@click.option(
    "--tag-offset/--no-tag-offset",
    default=None,
    help="Export second tag as relative/absolute length.",
)
@_handle_errors
def tag(**kwargs):
    """Add loop point metadata tags to audio file copy."""
    _run_export_handler(**kwargs)


def _run_export_handler(**kwargs):
    """Common export handler logic."""
    _resolve_audio_path(kwargs, output_dir_required=True)

    if Path(kwargs["path"]).is_file():
        with _processing_spinner():
            handler = LoopExportHandler(**kwargs)
        handler.run()
    else:
        BatchHandler(**kwargs).run()


if __name__ == "__main__":
    cli_main()
