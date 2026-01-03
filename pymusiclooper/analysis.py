import logging
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np
from numba import njit

from pymusiclooper.audio import MLAudio
from pymusiclooper.exceptions import LoopNotFoundError


@dataclass
class LoopPair:
    """A data class that encapsulates the loop point related data.
    Contains:
        loop_start: int (exact loop start position in samples)
        loop_end: int (exact loop end position in samples)
        note_distance: float
        loudness_difference: float
        score: float. Defaults to 0.
    """

    _loop_start_frame_idx: int
    _loop_end_frame_idx: int
    note_distance: float
    loudness_difference: float
    score: float = 0
    loop_start: int = 0
    loop_end: int = 0


def find_best_loop_points(
    mlaudio: MLAudio,
    min_duration_multiplier: float = 0.35,
    min_loop_duration: Optional[float] = None,
    max_loop_duration: Optional[float] = None,
    approx_loop_start: Optional[float] = None,
    approx_loop_end: Optional[float] = None,
    brute_force: bool = False,
    disable_pruning: bool = False,
) -> List[LoopPair]:
    """Finds the best loop points for a given audio track."""
    runtime_start = time.perf_counter()
    s2f = mlaudio.seconds_to_frames  # Local alias for repeated calls

    # Duration bounds (in frames)
    total_frames = s2f(mlaudio.total_duration)
    min_dur = (
        s2f(min_loop_duration)
        if min_loop_duration
        else s2f(int(min_duration_multiplier * mlaudio.total_duration))
    )
    max_dur = s2f(max_loop_duration) if max_loop_duration else total_frames
    min_dur = max(1, min_dur)

    approx_mode = approx_loop_start is not None and approx_loop_end is not None

    if approx_mode or brute_force:
        chroma, power_db, _, _ = _analyze_audio(mlaudio, skip_beat_analysis=True)
        bpm = 120.0

        if approx_mode:
            start_frame = s2f(approx_loop_start, apply_trim_offset=True)
            end_frame = s2f(approx_loop_end, apply_trim_offset=True)
            window = s2f(2)  # +/- 2 seconds

            min_dur = (end_frame - window) - (start_frame + window) - 1
            max_dur = (end_frame + window) - (start_frame - window) + 1

            beats = np.concatenate(
                [
                    np.arange(
                        max(0, start_frame - window),
                        min(total_frames, start_frame + window),
                    ),
                    np.arange(
                        max(0, end_frame - window),
                        min(total_frames, end_frame + window),
                    ),
                ]
            )
        else:  # brute_force
            beats = np.arange(chroma.shape[-1], dtype=int)
            n_iter = int(beats.size**2 * (1 - min_dur / chroma.shape[-1]))
            logging.info(f"Brute force: {beats.size} frames, ~{n_iter} iterations")
            logging.info("**NOTICE** Processing may take several minutes.")
    else:
        chroma, power_db, bpm, beats = _analyze_audio(mlaudio)
        logging.info(f"Detected {beats.size} beats at {bpm:.0f} bpm")

    logging.info(f"Initial processing: {time.perf_counter() - runtime_start:.3f}s")

    # Find candidate pairs
    t0 = time.perf_counter()
    candidate_pairs = [
        LoopPair(
            _loop_start_frame_idx=start,
            _loop_end_frame_idx=end,
            note_distance=note_dist,
            loudness_difference=loud_diff,
        )
        for start, end, note_dist, loud_diff in _find_candidate_pairs(
            chroma, power_db, beats, min_dur, max_dur
        )
    ]

    logging.info(
        f"Found {len(candidate_pairs)} candidates in {time.perf_counter() - t0:.3f}s"
    )

    if not candidate_pairs:
        raise LoopNotFoundError(
            f'No loop points found for "{mlaudio.filename}" with current parameters.'
        )

    filtered = _assess_and_filter_loop_pairs(
        mlaudio, chroma, bpm, candidate_pairs, disable_pruning
    )

    if len(filtered) > 1:
        _prioritize_duration(filtered)

    # Adjust to nearest zero crossings
    for pair in filtered:
        if mlaudio.trim_offset > 0:
            pair._loop_start_frame_idx = int(
                mlaudio.apply_trim_offset(pair._loop_start_frame_idx)
            )
            pair._loop_end_frame_idx = int(
                mlaudio.apply_trim_offset(pair._loop_end_frame_idx)
            )

        pair.loop_start = nearest_zero_crossing(
            mlaudio.playback_audio,
            mlaudio.rate,
            mlaudio.frames_to_samples(pair._loop_start_frame_idx),
        )
        pair.loop_end = nearest_zero_crossing(
            mlaudio.playback_audio,
            mlaudio.rate,
            mlaudio.frames_to_samples(pair._loop_end_frame_idx),
        )

    if not filtered:
        raise LoopNotFoundError(
            f'No loop points found for "{mlaudio.filename}" with current parameters.'
        )

    logging.info(f"Filtered to {len(filtered)} best candidates")
    logging.info(f"Total runtime: {time.perf_counter() - runtime_start:.3f}s")

    return filtered


def _analyze_audio(
    mlaudio: MLAudio, skip_beat_analysis: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Performs the main audio analysis required."""
    S = librosa.stft(y=mlaudio.audio)
    S_power = S.real**2 + S.imag**2  # Faster than np.abs(S)**2

    freqs = librosa.fft_frequencies(sr=mlaudio.rate)
    S_weighted = librosa.perceptual_weighting(S=S_power, frequencies=freqs)

    mel_spec = librosa.feature.melspectrogram(
        S=S_weighted, sr=mlaudio.rate, n_mels=128, fmax=8000
    )
    chroma = librosa.feature.chroma_stft(S=S_power)
    power_db = librosa.power_to_db(S_weighted, ref=np.median)

    if skip_beat_analysis:
        return chroma, power_db, None, None

    try:
        onset_env = librosa.onset.onset_strength(S=mel_spec)

        pulse = librosa.beat.plp(onset_envelope=onset_env)
        beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
        bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env)

        beats = np.union1d(beats, beats_plp)  # Already returns sorted unique
        bpm = bpm.item() if isinstance(bpm, np.ndarray) else bpm

    except Exception as e:
        raise LoopNotFoundError(
            f'Beat analysis failed for "{mlaudio.filename}". Cannot continue.'
        ) from e

    return chroma, power_db, bpm, beats


@njit(fastmath=True, cache=True)
def _db_diff(power_db_f1: np.ndarray, power_db_f2: np.ndarray) -> float:
    return abs(power_db_f1.max() - power_db_f2.max())


@njit(fastmath=True, cache=True)
def _norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.abs(a) ** 2, axis=0))


@njit(cache=True, fastmath=True)
def _find_candidate_pairs(
    chroma: np.ndarray,
    power_db: np.ndarray,
    beats: np.ndarray,
    min_loop_duration: int,
    max_loop_duration: int,
) -> List[Tuple[int, int, float, float]]:
    """Generates a list of all valid candidate loop pairs."""

    NOTE_THRESH = 0.0875
    LOUDNESS_THRESH = 0.5

    # Precompute at beat indices (contiguous memory)
    chroma_beats = np.ascontiguousarray(chroma[:, beats])
    power_beats = power_db[:, beats]
    n_beats = len(beats)
    n_chroma = chroma_beats.shape[0]

    # Precompute all thresholds and power maxes upfront
    deviation = np.empty(n_beats)
    power_max = np.empty(n_beats)

    for i in range(n_beats):
        acc = 0.0
        pmax = -np.inf
        for j in range(n_chroma):
            val = chroma_beats[j, i] * NOTE_THRESH
            acc += val * val
            if power_beats[j, i] > pmax:
                pmax = power_beats[j, i]
        deviation[i] = np.sqrt(acc)
        power_max[i] = pmax

    candidate_pairs = []

    for end_idx in range(n_beats):
        loop_end = beats[end_idx]
        threshold = deviation[end_idx]
        pmax_end = power_max[end_idx]

        # Valid start range: loop_end - max_dur <= start <= loop_end - min_dur
        min_valid_start = loop_end - max_loop_duration
        max_valid_start = loop_end - min_loop_duration

        for start_idx in range(n_beats):
            loop_start = beats[start_idx]

            if loop_start > max_valid_start:
                break  # Sorted: all subsequent are too close
            if loop_start < min_valid_start:
                continue  # Too far apart

            # Inline norm: manual loop is faster in Numba than np.dot for small arrays
            dist_sq = 0.0
            for j in range(n_chroma):
                d = chroma_beats[j, end_idx] - chroma_beats[j, start_idx]
                dist_sq += d * d
            note_distance = np.sqrt(dist_sq)

            if note_distance <= threshold:
                loudness_diff = abs(pmax_end - power_max[start_idx])

                if loudness_diff <= LOUDNESS_THRESH:
                    candidate_pairs.append(
                        (
                            loop_start,
                            loop_end,
                            note_distance,
                            loudness_diff,
                        )
                    )

    return candidate_pairs


def _assess_and_filter_loop_pairs(
    mlaudio: MLAudio,
    chroma: np.ndarray,
    bpm: float,
    candidate_pairs: List[LoopPair],
    disable_pruning: bool = False,
) -> List[LoopPair]:
    """Assigns scores to each loop pair and prunes the candidate list."""

    # Calculate test duration in frames (~12 beats)
    seconds_to_test = 12 / (bpm / 60)
    test_offset = mlaudio.samples_to_frames(int(seconds_to_test * mlaudio.rate))

    # Only adjust for very short tracks where test window exceeds entire track
    if test_offset > chroma.shape[-1]:
        test_offset = chroma.shape[-1] // 4

    # Prune if needed
    pairs = (
        _prune_candidates(candidate_pairs)
        if len(candidate_pairs) >= 100 and not disable_pruning
        else candidate_pairs
    )

    weights = _weights(test_offset, start=max(2, test_offset // 12), stop=1)

    # Score and assign in single pass
    for pair in pairs:
        pair.score = _calculate_loop_score(
            pair._loop_start_frame_idx,
            pair._loop_end_frame_idx,
            chroma,
            test_duration=test_offset,
            weights=weights,
        )

    return sorted(pairs, key=lambda p: p.score, reverse=True)


def _prune_candidates(
    candidate_pairs: List[LoopPair],
    keep_top_notes: float = 75,
    keep_top_loudness: float = 50,
    acceptable_loudness: float = 0.25,
) -> List[LoopPair]:
    """Prunes candidate pairs based on loudness and note distance thresholds."""

    epsilon = 1e-3

    db_diff = np.array([p.loudness_difference for p in candidate_pairs])
    note_dist = np.array([p.note_distance for p in candidate_pairs])

    # Filter out near-silent samples for percentile calculation
    db_valid = db_diff[db_diff > epsilon]
    note_valid = note_dist[note_dist > epsilon]

    # Use percentile if enough samples, else fall back to max
    db_thresh = (
        np.percentile(db_valid, keep_top_loudness)
        if db_valid.size > 3
        else db_diff.max()
    )
    note_thresh = (
        np.percentile(note_valid, keep_top_notes)
        if note_valid.size > 3
        else note_dist.max()
    )

    # Lower values are better
    mask = (db_diff <= max(acceptable_loudness, db_thresh)) & (note_dist <= note_thresh)

    from itertools import compress

    return list(compress(candidate_pairs, mask))


def _prioritize_duration(pair_list: List[LoopPair]) -> None:
    """Promotes the longest high-scoring loop to the front of the list (in-place)."""

    if len(pair_list) < 2:
        return

    db_diff = np.array([p.loudness_difference for p in pair_list])
    scores = np.array([p.score for p in pair_list])

    db_thresh = np.median(db_diff)
    score_thresh = max(np.percentile(scores, 90), pair_list[0].score - 1e-4)

    # Find longest duration among top-scoring, low-loudness pairs
    best_idx, best_dur = 0, 0

    for idx, pair in enumerate(pair_list):
        if pair.score < score_thresh:
            break  # List is sorted by score; no need to continue

        duration = pair.loop_end - pair.loop_start
        if duration > best_dur and pair.loudness_difference <= db_thresh:
            best_idx, best_dur = idx, duration

    # Move best to front if not already there
    if best_idx:
        pair_list.insert(0, pair_list.pop(best_idx))


def _calculate_loop_score(
    b1: int,
    b2: int,
    chroma: np.ndarray,
    test_duration: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Returns the best cosine similarity score (lookahead vs lookbehind) for two loop points."""

    rev_weights = weights[::-1] if weights is not None else None

    return max(
        _calculate_subseq_beat_similarity(b1, b2, chroma, test_duration, weights),
        _calculate_subseq_beat_similarity(b1, b2, chroma, -test_duration, rev_weights),
    )


def _calculate_subseq_beat_similarity(
    b1_start: int,
    b2_start: int,
    chroma: np.ndarray,
    test_end_offset: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculates cosine similarity of subsequent/preceding frames at two positions."""

    chroma_len = chroma.shape[-1]
    test_len = abs(test_end_offset)

    # Compute slice bounds
    if test_end_offset < 0:
        offset = min(test_len, b1_start, b2_start)
        s1, s2 = slice(b1_start - offset, b1_start), slice(b2_start - offset, b2_start)
    else:
        offset = min(test_len, chroma_len - b1_start, chroma_len - b2_start)
        s1, s2 = slice(b1_start, b1_start + offset), slice(b2_start, b2_start + offset)

    b1_slice, b2_slice = chroma[:, s1], chroma[:, s2]

    # Cosine similarity per frame
    dot = np.einsum("ij,ij->j", b1_slice, b2_slice)
    norms = np.linalg.norm(b1_slice, axis=0) * np.linalg.norm(b2_slice, axis=0)
    cosine_sim = dot / np.maximum(norms, 1e-10)

    # Pad with zeros if shorter than expected (penalizes truncated tests)
    if offset < test_len:
        cosine_sim = np.pad(cosine_sim, (0, test_len - offset))

    return np.average(cosine_sim, weights=weights)


def _weights(length: int, start: int = 100, stop: int = 1):
    return np.geomspace(start, stop, num=length)


@njit(cache=True, fastmath=True)
def nearest_zero_crossing(audio: np.ndarray, rate: int, sample_idx: int) -> int:
    """Finds the nearest rising zero crossing with minimal local amplitude.

    Simplified algorithm:
    - Detects actual sign changes (zero crossings)
    - Prefers rising crossings (negative→positive) for smoother splices
    - Scores by: local amplitude + distance from target
    """
    n_samples, n_channels = audio.shape
    window_size = max(2, rate // 100)  # ~10ms window
    half_win = window_size // 2

    start = max(0, sample_idx - half_win)
    end = min(n_samples, sample_idx + half_win + 1)

    if end - start < 2:
        return sample_idx

    # Mix to mono for crossing detection
    mono = audio[start:end, 0].copy()
    for ch in range(1, n_channels):
        mono += audio[start:end, ch]

    center = sample_idx - start
    best_idx, best_score = -1, np.inf

    for i in range(1, len(mono)):
        prev, curr = mono[i - 1], mono[i]

        # Skip non-crossings (same sign)
        if prev * curr > 0:
            continue

        # Score components (lower = better)
        amplitude = (abs(prev) + abs(curr)) / n_channels  # Local amplitude
        distance = 0.3 * abs(i - center) / half_win  # Distance penalty
        falling = 0.15 if prev > 0 else 0.0  # Prefer rising crossings

        score = amplitude + distance + falling

        if score < best_score:
            best_score, best_idx = score, i

    # Reject if no crossing found or score too poor
    if best_idx < 0 or best_score > 0.8 * n_channels:
        return sample_idx

    return start + best_idx
