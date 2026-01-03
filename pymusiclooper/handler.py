import logging
import os
import sys
from contextlib import contextmanager
from typing import List, Literal, Optional, Tuple

from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

from pymusiclooper.analysis import LoopPair
from pymusiclooper.console import rich_console
from pymusiclooper.core import MusicLooper
from pymusiclooper.exceptions import AudioLoadError, LoopNotFoundError
from pymusiclooper.utils import DEFAULT_OUTPUT_DIR


class LoopHandler:
    def __init__(
        self,
        *,
        path: str,
        min_duration_multiplier: float,
        min_loop_duration: float | None = None,
        max_loop_duration: float | None = None,
        approx_loop_position: tuple[float, float] | None = None,
        brute_force: bool = False,
        disable_pruning: bool = False,
        _progressbar: Progress | None = None,
        **kwargs,
    ):
        self.filepath = path
        self._musiclooper = MusicLooper(filepath=path)
        self._progressbar = _progressbar
        self.interactive_mode = os.getenv("PML_INTERACTIVE_MODE") is not None
        self.in_samples = os.getenv("PML_DISPLAY_SAMPLES") is not None

        # Unpack approximate loop position
        approx_start, approx_end = approx_loop_position or (None, None)

        logging.info(f'Loaded "{path}". Analyzing...')

        self.loop_pair_list = self._musiclooper.find_loop_pairs(
            min_duration_multiplier=min_duration_multiplier,
            min_loop_duration=min_loop_duration,
            max_loop_duration=max_loop_duration,
            approx_loop_start=approx_start,
            approx_loop_end=approx_end,
            brute_force=brute_force,
            disable_pruning=disable_pruning,
        )

    @property
    def musiclooper(self) -> MusicLooper:
        """Returns the handler's MusicLooper instance."""
        return self._musiclooper

    def get_all_loop_pairs(self) -> List[LoopPair]:
        """Returns discovered loop points as a list of LoopPair objects."""
        return self.loop_pair_list

    def format_time(self, samples: int) -> int | str:
        return (
            samples if self.in_samples else self.musiclooper.samples_to_ftime(samples)
        )

    def play_looping(self, loop_start: int, loop_end: int) -> None:
        self.musiclooper.play_looping(loop_start, loop_end)

    def choose_loop_pair(self, interactive_mode: bool = False) -> LoopPair:
        idx = 0
        if self.loop_pair_list and interactive_mode:
            with _hideprogressbar(self._progressbar):
                idx = self._interactive_handler()
        return self.loop_pair_list[idx]

    def _build_table(self, show_top: int) -> Table:
        """Build a Rich table displaying loop pair candidates."""
        total = len(self.loop_pair_list)
        shown = min(show_top, total)

        caption = (
            "\nEnter 'more' to display additional loop points, 'all' to display all, or 'reset' for default."
            if show_top < total
            else ""
        )

        table = Table(
            title=f"Discovered loop points\n({shown}/{total} displayed)",
            caption=caption,
        )

        columns = [
            ("Index", "right", "cyan"),
            ("Loop Start", "left", "magenta"),
            ("Loop End", "left", "green"),
            ("Length", "left", "white"),
            ("Note Distance", "left", "yellow"),
            ("Loudness Difference", "left", "blue"),
            ("Score", "right", "red"),
        ]
        for name, justify, style in columns:
            table.add_column(
                name, justify=justify, style=style, no_wrap=(name == "Index")
            )

        fmt = self.format_time
        for idx, pair in enumerate(self.loop_pair_list[:show_top]):
            table.add_row(
                str(idx),
                str(fmt(pair.loop_start)),
                str(fmt(pair.loop_end)),
                str(fmt(pair.loop_end - pair.loop_start)),
                f"{pair.note_distance:.4f}",
                f"{pair.loudness_difference:.4f}",
                f"{pair.score:.2%}",
            )
        return table

    def _interactive_handler(self, show_top: int = 25) -> int:
        """Interactive loop selection with preview support."""
        total = len(self.loop_pair_list)

        rich_console.print(f'Processing: "{self.filepath}"')
        rich_console.print(self._build_table(show_top))
        rich_console.print()

        while True:
            try:
                user_input = (
                    rich_console.input(
                        "Enter index to select (append [cyan]p[/] to preview, e.g. [cyan]0p[/]): "
                    )
                    .strip()
                    .lower()
                )

                # Handle display commands
                match user_input:
                    case "more":
                        return self._interactive_handler(show_top * 2)
                    case "all":
                        return self._interactive_handler(total)
                    case "reset":
                        return self._interactive_handler()

                # Parse index and preview flag
                preview = user_input.endswith("p")
                idx = int(user_input.rstrip("p"))

                if not 0 <= idx < total:
                    raise IndexError

                if preview:
                    self._preview_loop(idx)
                    continue

                return idx

            except (ValueError, IndexError):
                rich_console.print(f"Please enter a number in range [0, {total - 1}].")
            except KeyboardInterrupt:
                rich_console.print("\n[red]Operation cancelled.[/]")
                sys.exit()

    def _preview_loop(self, idx: int) -> None:
        """Preview a loop pair with 5-second lead-in."""
        pair = self.loop_pair_list[idx]
        rich_console.print(
            f"Previewing loop [cyan]#{idx}[/] | Press [red]Ctrl+C[/] to stop:"
        )

        offset = self.musiclooper.seconds_to_samples(5)
        start_from = max(0, pair.loop_end - offset)
        self.musiclooper.play_looping(
            pair.loop_start, pair.loop_end, start_from=start_from
        )


class LoopExportHandler(LoopHandler):
    def __init__(
        self,
        *,
        path: str,
        min_duration_multiplier: float,
        output_dir: str,
        min_loop_duration: Optional[float] = None,
        max_loop_duration: Optional[float] = None,
        approx_loop_position: Optional[tuple] = None,
        brute_force: bool = False,
        disable_pruning: bool = False,
        split_audio: bool = False,
        format: Literal["WAV", "FLAC", "OGG", "MP3"] = "WAV",
        to_txt: bool = False,
        to_stdout: bool = False,
        fmt: Literal["SAMPLES", "SECONDS", "TIME"] = "SAMPLES",
        alt_export_top: int = 0,
        tag_names: Optional[Tuple[str, str]] = None,
        tag_offset: Optional[bool] = None,
        batch_mode: bool = False,
        extended_length: float = 0,
        fade_length: float = 0,
        disable_fade_out: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            min_duration_multiplier=min_duration_multiplier,
            min_loop_duration=min_loop_duration,
            max_loop_duration=max_loop_duration,
            approx_loop_position=approx_loop_position,
            brute_force=brute_force,
            disable_pruning=disable_pruning,
            **kwargs,
        )
        self.output_directory = output_dir
        self.split_audio = split_audio
        self.format = format
        self.to_txt = to_txt
        self.to_stdout = to_stdout
        self.fmt = fmt.lower()
        self.alt_export_top = alt_export_top
        self.tag_names = tag_names
        self.tag_offset = tag_offset
        self.batch_mode = batch_mode
        self.extended_length = extended_length
        self.disable_fade_out = disable_fade_out
        self.fade_length = fade_length
        self._is_autocreated_outdir = False

    def run(self):
        self.loop_pair_list = self.get_all_loop_pairs()
        chosen_loop_pair = self.choose_loop_pair(self.interactive_mode)
        loop_start = chosen_loop_pair.loop_start
        loop_end = chosen_loop_pair.loop_end

        # Runners that do not need an output directory
        if self.to_stdout:
            self.stdout_export_runner(loop_start, loop_end)

        # TODO: refactor into a context manager instead
        try:
            # Runners that need an output directory
            if (
                self.tag_names
                or self.to_txt
                or self.split_audio
                or self.extended_length
            ) and not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)
                self._is_autocreated_outdir = True

            if self.tag_names is not None:
                self.tag_runner(loop_start, loop_end)

            if self.to_txt:
                self.txt_export_runner(loop_start, loop_end)

            if self.split_audio:
                self.split_audio_runner(loop_start, loop_end)

            if self.extended_length:
                self.extend_track_runner(loop_start, loop_end)
        finally:
            if (
                self._is_autocreated_outdir
                and os.path.exists(self.output_directory)
                and len(os.listdir(self.output_directory)) == 0
            ):
                os.rmdir(self.output_directory)

    def split_audio_runner(self, loop_start: int, loop_end: int):
        try:
            self.musiclooper.export(
                loop_start,
                loop_end,
                format=self.format,
                output_dir=self.output_directory,
            )
            message = f'Successfully exported "{self.musiclooper.filename}" intro/loop/outro sections to "{self.output_directory}"'
            if self.batch_mode:
                logging.info(message)
            else:
                rich_console.print(message)
        # Usually: unknown file format specified; raised by soundfile
        except ValueError as e:
            logging.error(e)

    def extend_track_runner(self, loop_start: int, loop_end: int):
        # Add a progress bar since it could take some time to export
        # Do not enable if batch mode is active, since it already has a progress bar
        if not self.batch_mode:
            progress = Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=rich_console,
                transient=True,
            )
            progress.add_task(
                f"Exporting an extended version of {self.musiclooper.filename}...",
                total=None,
            )
            progress.start()
        try:
            output_path = self.musiclooper.extend(
                loop_start,
                loop_end,
                format=self.format,
                output_dir=self.output_directory,
                extended_length=self.extended_length,
                disable_fade_out=self.disable_fade_out,
                fade_length=self.fade_length,
            )
            message = f'Successfully exported an extended version of "{self.musiclooper.filename}" to "{output_path}"'
            if self.batch_mode:
                logging.info(message)
            else:
                progress.stop()
                rich_console.print(message)
        # Usually: unknown file format specified; raised by soundfile
        except ValueError as e:
            logging.error(e)

    def txt_export_runner(self, loop_start: int, loop_end: int):
        if self.alt_export_top != 0:
            self.alt_export_runner(mode="TXT")
        else:
            self.musiclooper.export_txt(
                self._fmt(loop_start),
                self._fmt(loop_end),
                output_dir=self.output_directory,
            )
            out_path = os.path.join(self.output_directory, "loop.txt")
            message = f'Successfully added "{self.musiclooper.filename}" loop points to "{out_path}"'
            if self.batch_mode:
                logging.info(message)
            else:
                rich_console.print(message)

    def stdout_export_runner(self, loop_start: int, loop_end: int):
        if self.alt_export_top != 0:
            self.alt_export_runner(mode="STDOUT")
        else:
            rich_console.print(
                f'\nLoop points for "{self.musiclooper.filename}":\n'
                f"LOOP_START: {self._fmt(loop_start)}\n"
                f"LOOP_END: {self._fmt(loop_end)}\n"
            )

    def alt_export_runner(self, mode: Literal["STDOUT", "TXT"]):
        pair_list_slice = (
            self.loop_pair_list
            if self.alt_export_top < 0
            or self.alt_export_top >= len(self.loop_pair_list)
            else self.loop_pair_list[: self.alt_export_top]
        )

        def fmt_line(pair: LoopPair):
            return f"{self._fmt(pair.loop_start)} {self._fmt(pair.loop_end)} {pair.note_distance} {pair.loudness_difference} {pair.score}\n"

        formatted_lines = [fmt_line(pair) for pair in pair_list_slice]
        if mode == "STDOUT":
            rich_console.out(*formatted_lines, sep="", end="")
        elif mode == "TXT":
            out_path = os.path.join(
                self.output_directory, f"{self.musiclooper.filename}.alt_export.txt"
            )
            with open(out_path, mode="w") as f:
                f.writelines(formatted_lines)

    def tag_runner(self, loop_start: int, loop_end: int):
        loop_start_tag, loop_end_tag = self.tag_names
        loop_start, loop_end = self.musiclooper.export_tags(
            loop_start,
            loop_end,
            loop_start_tag,
            loop_end_tag,
            is_offset=self.tag_offset,
            output_dir=self.output_directory,
        )
        message = f'Exported {loop_start_tag}: {loop_start} and {loop_end_tag}: {loop_end} of "{self.musiclooper.filename}" to a copy in "{self.output_directory}"'
        if self.batch_mode:
            logging.info(message)
        else:
            rich_console.print(message)

    def _fmt(self, samples: int):
        if self.fmt == "seconds":
            return str(self.musiclooper.samples_to_seconds(samples))
        elif self.fmt == "time":
            return str(self.musiclooper.samples_to_ftime(samples))
        else:
            return str(samples)


class BatchHandler:
    def __init__(
        self,
        *,
        path: str,
        output_dir: str,
        recursive: bool = False,
        flatten: bool = False,
        **kwargs,
    ):
        """Processes all audio files in a directory with `LoopExportHandler`.

        Args:
            path (str): Path to directory.
            output_dir (str): Output directory to use for exports.
            recursive (bool, optional): Process directories recursively. Defaults to False.
            flatten (bool, optional): Flatten the output directory structure instead of preserving it when processing it recursively. Defaults to False.
            kwargs: Additional `kwargs` are passed onto `LoopExportHandler`.
        """
        self.directory_path = os.path.abspath(path)
        self.output_directory = output_dir
        self.recursive = recursive
        self.flatten = flatten
        self.kwargs = kwargs
        self._created_dirs = []

    def run(self):
        files = self.get_files_in_directory(
            self.directory_path, recursive=self.recursive
        )

        if len(files) == 0:
            raise FileNotFoundError(f'No files found in "{self.directory_path}"')

        output_dirs = (
            None
            if self.flatten
            else self.clone_file_tree_structure(files, self.output_directory)
        )

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            MofNCompleteColumn(),
            console=rich_console,
        ) as progress:
            pbar = progress.add_task("Processing...", total=len(files))
            for file_idx, file_path in enumerate(files):
                progress.update(
                    pbar,
                    advance=1,
                    description=(
                        f'Processing "{os.path.relpath(file_path, self.directory_path)}"'
                    ),
                )
                task_kwargs = {
                    **self.kwargs,
                    "_progressbar": progress,
                    "path": file_path,
                    "output_dir": (
                        self.output_directory if self.flatten else output_dirs[file_idx]
                    ),
                }
                try:
                    self._batch_export_helper(**task_kwargs)
                finally:
                    self._cleanup_empty_created_dirs()

    def clone_file_tree_structure(
        self, in_files: List[str], output_directory: str
    ) -> List[str]:
        common_path = os.path.commonpath(in_files)
        output_dirs = [
            os.path.join(
                os.path.abspath(output_directory),
                os.path.dirname(os.path.relpath(file, start=common_path)),
            )
            for file in in_files
        ]
        for out_dir in output_dirs:
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                self._created_dirs.append(out_dir)
        return output_dirs

    @staticmethod
    def get_files_in_directory(dir_path: str, recursive: bool = False) -> List[str]:
        return (
            [
                os.path.join(directory, filename)
                for directory, sub_dir_list, file_list in os.walk(dir_path)
                for filename in file_list
            ]
            if recursive
            else [
                os.path.join(dir_path, f)
                for f in os.listdir(dir_path)
                if os.path.isfile(os.path.join(dir_path, f))
            ]
        )

    @staticmethod
    def _batch_export_helper(**kwargs):
        try:
            export_handler = LoopExportHandler(**kwargs, batch_mode=True)
            export_handler.run()
        except (AudioLoadError, LoopNotFoundError) as e:
            logging.error(e)
        except Exception as e:
            logging.error(e)

    def _cleanup_empty_created_dirs(self):
        dirs_to_check = self._created_dirs

        if os.path.basename(self.output_directory) == DEFAULT_OUTPUT_DIRECTORY_NAME:
            dirs_to_check += [self.output_directory]

        for directory in dirs_to_check:
            if os.path.exists(directory) and len(os.listdir(directory)) == 0:
                os.removedirs(directory)


@contextmanager
def _hideprogressbar(progress: Progress):
    """
    Intended to pause and hide the progress bar while a prompt is active.

    Based on @abrahammurciano's answer in rich issue #1535
    https://github.com/Textualize/rich/issues/1535#issuecomment-1745297594
    """
    # Handle edge case where a progressbar might not exist
    if progress is None:
        try:
            yield
        finally:
            pass
        return
    transient = progress.live.transient  # save the old value
    progress.live.transient = True
    progress.stop()
    try:
        yield
    finally:
        # make space for the progress to use so it doesn't overwrite any previous lines
        print("\n" * (len(progress.tasks) - 2))
        progress.live.transient = transient  # restore the old value
        progress.start()
