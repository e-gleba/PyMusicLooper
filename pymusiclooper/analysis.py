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
    """Finds the best loop points for a given audio track, given the constraints specified

    Args:
        mlaudio (MLAudio): The MLAudio object to use for analysis
        min_duration_multiplier (float, optional): The minimum duration of a loop as a multiplier of track duration. Defaults to 0.35.
        min_loop_duration (float, optional): The minimum duration of a loop (in seconds). Defaults to None.
        max_loop_duration (float, optional): The maximum duration of a loop (in seconds). Defaults to None.
        approx_loop_start (float, optional): The approximate location of the desired loop start (in seconds). If specified, must specify approx_loop_end as well. Defaults to None.
        approx_loop_end (float, optional): The approximate location of the desired loop end (in seconds). If specified, must specify approx_loop_start as well. Defaults to None.
        brute_force (bool, optional): Checks the entire track instead of the detected beats (disclaimer: runtime may be significantly longer). Defaults to False.
        disable_pruning (bool, optional): Returns all the candidate loop points without filtering. Defaults to False.
    Raises:
        LoopNotFoundError: raised in case no loops were found

    Returns:
        List[LoopPair]: A list of `LoopPair` objects containing the loop points related data. See the `LoopPair` class for more info.
    """
    runtime_start = time.perf_counter()
    min_loop_duration = (
        mlaudio.seconds_to_frames(min_loop_duration)
        if min_loop_duration is not None
        else mlaudio.seconds_to_frames(
            int(min_duration_multiplier * mlaudio.total_duration)
        )
    )
    max_loop_duration = (
        mlaudio.seconds_to_frames(max_loop_duration)
        if max_loop_duration is not None
        else mlaudio.seconds_to_frames(mlaudio.total_duration)
    )

    # Loop points must be at least 1 frame apart
    min_loop_duration = max(1, min_loop_duration)

    if approx_loop_start is not None and approx_loop_end is not None:
        # Skipping the unnecessary beat analysis (in this case) speeds up the analysis runtime by ~2x
        # and significantly reduces the total memory consumption
        chroma, power_db, _, _ = _analyze_audio(mlaudio, skip_beat_analysis=True)
        # Set bpm to a general average of 120
        bpm = 120.0

        approx_loop_start = mlaudio.seconds_to_frames(
            approx_loop_start, apply_trim_offset=True
        )
        approx_loop_end = mlaudio.seconds_to_frames(
            approx_loop_end, apply_trim_offset=True
        )

        n_frames_to_check = mlaudio.seconds_to_frames(2)

        # Adjust min and max loop duration checks to the specified range
        min_loop_duration = (
            (approx_loop_end - n_frames_to_check)
            - (approx_loop_start + n_frames_to_check)
            - 1
        )
        max_loop_duration = (
            (approx_loop_end + n_frames_to_check)
            - (approx_loop_start - n_frames_to_check)
            + 1
        )

        # Override the beats to check with the specified approx points +/- 2 seconds
        beats = np.concatenate(
            [
                np.arange(
                    start=max(0, approx_loop_start - n_frames_to_check),
                    stop=min(
                        mlaudio.seconds_to_frames(mlaudio.total_duration),
                        approx_loop_start + n_frames_to_check,
                    ),
                ),
                np.arange(
                    start=max(0, approx_loop_end - n_frames_to_check),
                    stop=min(
                        mlaudio.seconds_to_frames(mlaudio.total_duration),
                        approx_loop_end + n_frames_to_check,
                    ),
                ),
            ]
        )
    elif brute_force:
        # Similarly skip beat analysis, as the results will not be used
        chroma, power_db, _, _ = _analyze_audio(mlaudio, skip_beat_analysis=True)
        bpm = 120.0
        beats = np.arange(start=0, stop=chroma.shape[-1], step=1, dtype=int)
        logging.info(f"Overriding number of frames to check with: {beats.size}")
        logging.info(
            f"Estimated iterations required using brute force: {int(beats.size*beats.size*(1-(min_loop_duration/chroma.shape[-1])))}"
        )
        logging.info(
            "**NOTICE** The program may appear frozen, but processing will continue in the background. This operation may take several minutes to complete."
        )
    else:  # normal mode of operation
        chroma, power_db, bpm, beats = _analyze_audio(mlaudio)
        logging.info(f"Detected {beats.size} beats at {bpm:.0f} bpm")

    logging.info(
        "Finished initial audio processing in {:.3f}s".format(
            time.perf_counter() - runtime_start
        )
    )

    initial_pairs_start_time = time.perf_counter()

    # Since numba jitclass cannot be cached, the pair data must be stored temporarily in a list of tuple
    # (instead of a list of LoopPairs directly) and then loaded into a list of LoopPair objects using list comprehension
    unproc_candidate_pairs = _find_candidate_pairs(
        chroma, power_db, beats, min_loop_duration, max_loop_duration
    )
    candidate_pairs = [
        LoopPair(
            _loop_start_frame_idx=tup[0],
            _loop_end_frame_idx=tup[1],
            note_distance=tup[2],
            loudness_difference=tup[3],
        )
        for tup in unproc_candidate_pairs
    ]

    n_candidate_pairs = len(candidate_pairs) if candidate_pairs is not None else 0
    logging.info(
        f"Found {n_candidate_pairs} possible loop points in"
        f" {(time.perf_counter() - initial_pairs_start_time):.3f}s"
    )

    if not candidate_pairs:
        raise LoopNotFoundError(
            f'No loop points found for "{mlaudio.filename}" with current parameters.'
        )

    filtered_candidate_pairs = _assess_and_filter_loop_pairs(
        mlaudio, chroma, bpm, candidate_pairs, disable_pruning
    )

    # prefer longer loops for highly similar sequences
    if len(filtered_candidate_pairs) > 1:
        _prioritize_duration(filtered_candidate_pairs)

    # Set the exact loop start and end in samples and adjust them
    # to the nearest zero crossing. Avoids audio popping/clicking while looping
    # as much as possible.
    for pair in filtered_candidate_pairs:
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

    if not filtered_candidate_pairs:
        raise LoopNotFoundError(
            f"No loop points found for {mlaudio.filename} with current parameters."
        )

    logging.info(
        f"Filtered to {len(filtered_candidate_pairs)} best candidate loop points"
    )
    logging.info(f"Total analysis runtime: {time.perf_counter() - runtime_start:.3f}s")

    return filtered_candidate_pairs


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

    ACCEPTABLE_NOTE_DEVIATION = 0.0875
    ACCEPTABLE_LOUDNESS_DIFFERENCE = 0.5

    # Precompute arrays at beat indices (avoids repeated fancy indexing)
    chroma_beats = chroma[:, beats]
    power_beats = power_db[:, beats]
    n_beats = len(beats)

    # Deviation thresholds per beat (inlined _norm)
    deviation = np.sqrt(np.sum((chroma_beats * ACCEPTABLE_NOTE_DEVIATION) ** 2, axis=0))

    candidate_pairs = []

    for end_idx in range(n_beats):
        loop_end = beats[end_idx]
        chroma_end = chroma_beats[:, end_idx]
        power_max_end = power_beats[:, end_idx].max()  # Hoist out of inner loop
        threshold = deviation[end_idx]

        for start_idx in range(n_beats):
            loop_start = beats[start_idx]
            loop_length = loop_end - loop_start

            if loop_length < min_loop_duration:
                break
            if loop_length > max_loop_duration:
                continue

            # Inlined _norm: np.dot is faster than np.sum(x**2)
            diff = chroma_end - chroma_beats[:, start_idx]
            note_distance = np.sqrt(np.dot(diff, diff))

            if note_distance <= threshold:
                # Inlined _db_diff
                loudness_diff = abs(power_max_end - power_beats[:, start_idx].max())

                if loudness_diff <= ACCEPTABLE_LOUDNESS_DIFFERENCE:
                    candidate_pairs.append(
                        (
                            int(loop_start),
                            int(loop_end),
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
    """Assigns the scores to each loop pair and prunes the list of candidate loop pairs

    Args:
        mlaudio (MLAudio): MLAudio object of the track being analyzed
        chroma (np.ndarray): The chroma spectrogram
        bpm (float): The estimated bpm/tempo of the track
        candidate_pairs (List[LoopPair]): The list of candidate loop pairs found
        disable_pruning (bool, optional): Returns all the candidate loop points without filtering. Defaults to False.

    Returns:
        List[LoopPair]: A scored and filtered list of valid loop candidate pairs
    """
    beats_per_second = bpm / 60
    num_test_beats = 12
    seconds_to_test = num_test_beats / beats_per_second
    test_offset = mlaudio.samples_to_frames(int(seconds_to_test * mlaudio.rate))

    # adjust offset for very short tracks to 25% of its length
    if test_offset > chroma.shape[-1]:
        test_offset = chroma.shape[-1] // 4

    # Prune candidates if there are too many
    if len(candidate_pairs) >= 100 and not disable_pruning:
        pruned_candidate_pairs = _prune_candidates(candidate_pairs)
    else:
        pruned_candidate_pairs = candidate_pairs

    weights = _weights(test_offset, start=max(2, test_offset // num_test_beats), stop=1)

    pair_score_list = [
        _calculate_loop_score(
            int(pair._loop_start_frame_idx),
            int(pair._loop_end_frame_idx),
            chroma,
            test_duration=test_offset,
            weights=weights,
        )
        for pair in pruned_candidate_pairs
    ]
    # Add cosine similarity as score
    for pair, score in zip(pruned_candidate_pairs, pair_score_list):
        pair.score = score

    # re-sort based on new score
    pruned_candidate_pairs = sorted(
        pruned_candidate_pairs, reverse=True, key=lambda x: x.score
    )
    return pruned_candidate_pairs


def _prune_candidates(
    candidate_pairs: List[LoopPair],
    keep_top_notes: float = 75,
    keep_top_loudness: float = 50,
    acceptable_loudness=0.25,
) -> List[LoopPair]:
    db_diff_array = np.array([pair.loudness_difference for pair in candidate_pairs])
    note_dist_array = np.array([pair.note_distance for pair in candidate_pairs])

    # Minimum value used to avoid issues with tracks with lots of silence
    epsilon = 1e-3
    min_adjusted_db_diff_array = db_diff_array[db_diff_array > epsilon]
    min_adjusted_note_dist_array = note_dist_array[note_dist_array > epsilon]

    # Avoid index errors by having at least 3 elements when performing percentile-based pruning
    # Otherwise, skip by setting the value to the highest available
    if min_adjusted_db_diff_array.size > 3:
        db_threshold = np.percentile(min_adjusted_db_diff_array, keep_top_loudness)
    else:
        db_threshold = np.max(db_diff_array)

    if min_adjusted_note_dist_array.size > 3:
        note_dist_threshold = np.percentile(
            min_adjusted_note_dist_array, keep_top_notes
        )
    else:
        note_dist_threshold = np.max(note_dist_array)

    # Lower values are better
    indices_that_meet_cond = np.flatnonzero(
        (db_diff_array <= max(acceptable_loudness, db_threshold))
        & (note_dist_array <= note_dist_threshold)
    )
    return [candidate_pairs[idx] for idx in indices_that_meet_cond]


def _prioritize_duration(pair_list: List[LoopPair]) -> List[LoopPair]:
    db_diff_array = np.array([pair.loudness_difference for pair in pair_list])
    db_threshold = np.median(db_diff_array)

    duration_argmax = 0
    duration_max = 0

    score_array = np.array([pair.score for pair in pair_list])
    score_threshold = np.percentile(score_array, 90)

    # Must be a negligible difference from the top score
    score_threshold = max(score_threshold, pair_list[0].score - 1e-4)

    # Since pair_list is already sorted
    # Break the loop if the condition is not met
    for idx, pair in enumerate(pair_list):
        if pair.score < score_threshold:
            break
        duration = pair.loop_end - pair.loop_start
        if duration > duration_max and pair.loudness_difference <= db_threshold:
            duration_max, duration_argmax = duration, idx

    if duration_argmax:
        pair_list.insert(0, pair_list.pop(duration_argmax))


def _calculate_loop_score(
    b1: int,
    b2: int,
    chroma: np.ndarray,
    test_duration: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculates the similarity of two sequences given the starting indices `b1` and `b2` for the period of the `test_duration` specified.
        Returns the best score based on the cosine similarity of subsequent (or preceding) notes.

    Args:
        b1 (int): Frame index of the first beat to compare
        b2 (int): Frame index of the second beat to compare
        chroma (np.ndarray): The chroma spectrogram of the audio
        test_duration (int): How many frames along the chroma spectrogram to test.
        weights (np.ndarray, optional): If specified, will provide a weighted average of the note scores according to the weight array provided. Defaults to None.

    Returns:
        float: the weighted average of the cosine similarity of the notes along the tested region
    """
    lookahead_score = _calculate_subseq_beat_similarity(
        b1, b2, chroma, test_duration, weights=weights
    )
    lookbehind_score = _calculate_subseq_beat_similarity(
        b1, b2, chroma, -test_duration, weights=weights[::-1]
    )

    return max(lookahead_score, lookbehind_score)


def _calculate_subseq_beat_similarity(
    b1_start: int,
    b2_start: int,
    chroma: np.ndarray,
    test_end_offset: int,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Calculates the similarity of subsequent notes of the two specified indices (b1_start, b2_start) using cosine similarity

    Args:
        b1_start (int): Starting frame index of the first beat to compare
        b2_start (int): Starting frame index of the second beat to compare
        chroma (np.ndarray): The chroma spectrogram of the audio
        test_end_offset (int): The number of frames to offset from the starting index. If negative, will be testing the preceding frames instead of the subsequent frames.
        weights (np.ndarray, optional): If specified, will provide a weighted average of the note scores according to the weight array provided. Defaults to None.

    Returns:
        float: the weighted average of the cosine similarity of the notes along the tested region
    """
    chroma_len = chroma.shape[-1]
    test_length = abs(test_end_offset)

    # Compute slice bounds directly
    if test_end_offset < 0:
        max_offset = min(test_length, b1_start, b2_start)
        b1_slice = chroma[..., b1_start - max_offset : b1_start]
        b2_slice = chroma[..., b2_start - max_offset : b2_start]
    else:
        max_offset = min(test_length, chroma_len - b1_start, chroma_len - b2_start)
        b1_slice = chroma[..., b1_start : b1_start + max_offset]
        b2_slice = chroma[..., b2_start : b2_start + max_offset]

    # Cosine similarity per frame (vectorized)
    dot_prod = np.einsum("ij,ij->j", b1_slice, b2_slice)
    norm_prod = np.linalg.norm(b1_slice, axis=0) * np.linalg.norm(b2_slice, axis=0)
    cosine_sim = dot_prod / np.maximum(norm_prod, 1e-10)

    # Pad if needed and return weighted average
    if max_offset < test_length:
        cosine_sim = np.pad(
            cosine_sim, (0, test_length - max_offset), constant_values=0
        )

    return np.average(cosine_sim, weights=weights)


def _weights(length: int, start: int = 100, stop: int = 1):
    return np.geomspace(start, stop, num=length)


@njit(cache=True, fastmath=True)
def nearest_zero_crossing(audio: np.ndarray, rate: int, sample_idx: int) -> int:
    """Returns the best closest sample point at a rising zero crossing point.

    Implementation of Audacity's 'At Zero Crossings' feature.
    """
    n_channels = audio.shape[1]
    window_size = max(1, rate // 100)  # 1/100th of a second
    offset = window_size // 2

    # Sample window centered around sample_idx
    start = max(0, sample_idx - offset)
    end = min(audio.shape[0], sample_idx + offset)
    sample_window = audio[start:end]
    length = sample_window.shape[0]

    # Offset correction for left-side clipping
    offset_correction = max(0, offset - sample_idx)
    pos_scale = 0.2 / window_size  # Simplified: 0.1 / (window_size / 2)

    dist = np.zeros(length)

    for channel in range(n_channels):
        samples = sample_window[:, channel]
        prev = 2.0

        for i in range(length):
            fdist = abs(samples[i])
            if prev * samples[i] > 0:  # Same sign - no good
                fdist += 0.4
            elif prev > 0.0:  # Downward crossing - medium penalty
                fdist += 0.1
            prev = samples[i]
            dist[i] += fdist + pos_scale * abs(i - offset + offset_correction)

    argmin = np.argmin(dist)
    threshold = 0.2 if n_channels == 1 else 0.6 * n_channels

    if dist[argmin] > threshold:
        return sample_idx

    return sample_idx + argmin - offset + offset_correction
