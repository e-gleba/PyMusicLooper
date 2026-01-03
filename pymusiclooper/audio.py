from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

from pymusiclooper.exceptions import AudioLoadError


class MLAudio:
    """Wrapper class for loading audio files for PyMusicLooper."""

    __slots__ = (
        "filepath",
        "filename",
        "rate",
        "total_duration",
        "trim_offset",
        "audio",
        "playback_audio",
        "n_channels",
        "length",
    )

    def __init__(self, filepath: str | Path) -> None:
        """Load and initialize audio data from filepath."""
        path = Path(filepath)

        try:
            raw_audio, sr = librosa.load(path, sr=None, mono=False)
        except Exception as e:
            raise AudioLoadError(
                f"{path.name} could not be loaded. Invalid audio data or unsupported format."
            ) from e

        if raw_audio.size == 0:
            raise AudioLoadError(f'No audio data could be loaded from "{path}".')

        mono = librosa.to_mono(raw_audio)

        if mono.min() == 0 == mono.max():
            raise AudioLoadError(f'"{path}" contains only silence.')

        # Normalize and trim
        mono /= np.abs(mono).max()
        trimmed, (trim_start, _) = librosa.effects.trim(mono, top_db=40)

        # Store core attributes
        self.filepath = str(path)
        self.filename = path.name
        self.rate = sr
        self.total_duration = librosa.get_duration(y=raw_audio, sr=sr)
        self.trim_offset = trim_start
        self.audio = trimmed

        # Prepare playback audio: shape (samples, channels)
        self.n_channels = 1 if raw_audio.ndim == 1 else raw_audio.shape[0]
        self.playback_audio = np.atleast_2d(raw_audio).T
        self.length = self.playback_audio.shape[0]

    def apply_trim_offset(self, frame: int) -> int:
        if not self.trim_offset:
            return frame
        samples = librosa.frames_to_samples(frame) + self.trim_offset
        return librosa.samples_to_frames(samples)

    def samples_to_frames(self, samples: int) -> int:
        return librosa.samples_to_frames(samples)

    def samples_to_seconds(self, samples: int) -> float:
        return librosa.samples_to_time(samples, sr=self.rate)

    def frames_to_samples(self, frame: int) -> int:
        return librosa.frames_to_samples(frame)

    def seconds_to_frames(self, seconds: float, apply_trim_offset: bool = False) -> int:
        if apply_trim_offset:
            seconds -= librosa.samples_to_time(self.trim_offset, sr=self.rate)
        return librosa.time_to_frames(seconds, sr=self.rate)

    def seconds_to_samples(self, seconds: float) -> int:
        return librosa.time_to_samples(seconds, sr=self.rate)

    def _format_time(self, seconds: float) -> str:
        return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"

    def frames_to_ftime(self, frame: int) -> str:
        return self._format_time(librosa.frames_to_time(frame, sr=self.rate))

    def samples_to_ftime(self, samples: int) -> str:
        return self._format_time(librosa.samples_to_time(samples, sr=self.rate))
