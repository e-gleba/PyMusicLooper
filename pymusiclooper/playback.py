"""Module for playback through the terminal"""

import importlib
import logging
import signal
import threading
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn

from pymusiclooper.console import rich_console


@lru_cache(maxsize=1)
def sd():
    """Lazy-load sounddevice module."""
    return importlib.import_module("sounddevice")


class PlaybackHandler:
    """Handler class for looping audio playback through the terminal."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.progressbar = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[time_field]}"),
            TextColumn("{task.fields[loop_field]}"),
            console=rich_console,
            transient=True,
            refresh_per_second=2,
        )
        self.stream = None
        self.looping = True
        self.loop_counter = 0
        self.current_frame = 0

    @contextmanager
    def _signal_handler(self):
        """Temporarily override SIGINT with custom loop interrupt handler."""
        original = signal.signal(signal.SIGINT, self._loop_interrupt_handler)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, original)

    def play_looping(
        self,
        playback_data: np.ndarray,
        samplerate: int,
        n_channels: int,
        loop_start: int,
        loop_end: int,
        start_from: int = 0,
    ) -> None:
        """Plays audio with a loop between loop_start and loop_end. Ctrl+C to interrupt."""

        total_samples = playback_data.shape[0]

        # Validate loop bounds
        if not (0 <= loop_start < loop_end < total_samples):
            raise ValueError(
                f"Invalid loop bounds: start={loop_start}, end={loop_end}, total={total_samples}"
            )

        # Reset state
        self.loop_counter = 0
        self.looping = True
        self.current_frame = start_from
        self.event.clear()

        def callback(outdata, frames, time, status):
            pos = self.current_frame

            # Check for loop crossing
            if self.looping and pos + frames > loop_end:
                pre = loop_end - pos
                post = frames - pre
                outdata[:pre] = playback_data[pos:loop_end]
                outdata[pre:] = playback_data[loop_start : loop_start + post]
                self.current_frame = loop_start + post
                self.loop_counter += 1
            else:
                end = min(pos + frames, total_samples)
                chunk = end - pos
                outdata[:chunk] = playback_data[pos:end]
                if chunk < frames:
                    outdata[chunk:] = 0
                    raise sd().CallbackStop()
                self.current_frame = end

        try:
            self.stream = sd().OutputStream(
                samplerate=samplerate,
                channels=n_channels,
                callback=callback,
                finished_callback=self.event.set,
            )

            with self.stream, self.progressbar, self._signal_handler():
                pbar = self.progressbar.add_task(
                    "Now Playing...",
                    total=total_samples,
                    loop_field="",
                    time_field="",
                )

                # Poll with 0.5s timeout (Windows threading workaround)
                while not self.event.wait(0.5):
                    t = self.current_frame / samplerate
                    self.progressbar.update(
                        pbar,
                        completed=self.current_frame,
                        time_field=f"{int(t // 60):02d}:{int(t % 60):02d}",
                        loop_field=(
                            f"[dim] | Loop #{self.loop_counter}[/]"
                            if self.loop_counter
                            else ""
                        ),
                    )

        except Exception as e:
            logging.error(e)

    def _loop_interrupt_handler(self, *args):
        if self.looping:
            self.looping = False
            rich_console.print(
                "[dim italic yellow](Looping disabled. Ctrl+C again to stop.)[/]"
            )
        else:
            self.event.set()
            if self.stream:
                self.stream.stop()
                self.stream.close()
            rich_console.print("[dim]Playback interrupted.[/]")
