# pymusiclooper

[![downloads](https://static.pepy.tech/badge/pymusiclooper)](https://pepy.tech/project/pymusiclooper)
[![downloads](https://static.pepy.tech/badge/pymusiclooper/month)](https://pepy.tech/project/pymusiclooper)
[![pypi pyversions](https://img.shields.io/pypi/v/pymusiclooper.svg)](https://pypi.python.org/pypi/pymusiclooper/)
[![pypi pyversions](https://img.shields.io/pypi/pyversions/pymusiclooper.svg)](https://pypi.python.org/pypi/pymusiclooper/)

a python-based program for repeating music seamlessly and endlessly, by automatically finding the best loop points.

features:

- find loop points within any audio file (if they exist).
- supports loading the most common audio formats (mp3, ogg, flac, wav), with additional codec support available through ffmpeg.
- play the audio file endlessly and seamlessly with the best automatically discovered loop points, or using the loop metadata tags present in the audio file.
- export to intro/loop/outro sections for editing or seamless playback within any music player that supports [gapless playback](https://en.wikipedia.org/wiki/gapless_playback).
- export loop points in samples directly to the terminal or to a text file (e.g. for use in creating custom themes with seamlessly looping audio).
- export the loop points as metadata tags to a copy of the input audio file(s), for use with game engines, etc.
- export a longer, extended version of an audio track by looping it seamlessly to the desired length

## pre-requisites

the following software must be installed for `pymusiclooper` to function correctly.

- [python (64-bit)](https://www.python.org/downloads/) >=3.10
- [ffmpeg](https://ffmpeg.org/download.html): required for loading audio from youtube (or any stream supported by [yt-dlp](https://github.com/yt-dlp/yt-dlp)) and adds support for loading additional audio formats and codecs such as m4a/aac, apple lossless (alac), wma, atrac (.at9), etc. a full list can be found at [ffmpeg's documentation](https://www.ffmpeg.org/general.html#audio-codecs). if the aforementioned features are not required, can be skipped.

supported audio formats _without_ ffmpeg include: wav, flac, ogg/vorbis, ogg/opus, mp3.
a full list can be found at [libsndfile's supported formats page](https://libsndfile.github.io/libsndfile/formats.html)

additionally, to use the `play` command on linux systems, you may need to
install the portaudio library. on ubuntu, run `sudo apt install libportaudio2`.

## installation

### option 1: installing using uv [recommended]

this method of installation is strongly recommended, as it isolates pymusiclooper's dependencies from the rest of your environment,
and as a result, avoids dependency conflicts and breakage due to other packages.

required tool: [`uv`](https://github.com/astral-sh/uv).

note: python is not required, as `uv` automatically installs this package's required python version automatically if not present.

```sh
# normal install
# (follows the official releases on https://pypi.org/project/pymusiclooper/)
uv tool install pymusiclooper

# alternative install
# (follows the git repository; equivalent to a nightly release channel)
uv tool install git+https://github.com/arkrow/pymusiclooper.git

# updating to new releases in either case can be done simply using:
uv tool upgrade pymusiclooper
```

installation note: you may need to specify a python version if the latest python release is not supported and fails to install, e.g.

```sh
uv tool install pymusiclooper --python "3.12"
```

### option 2: installing using pipx

like `uv`, isolates pymusiclooper's dependencies from the rest of your environment,
and as a result, avoids dependency conflicts and breakage due to other packages.
however, unlike `uv`, requires python to already be installed along with `pipx`.

required python packages: [`pipx`](https://pypa.github.io/pipx/) (can be installed using `pip install pipx` ).

```sh
# normal install
# (follows the official releases on https://pypi.org/project/pymusiclooper/)
pipx install pymusiclooper

# alternative install
# (follows the git repository; equivalent to a nightly release channel)
pipx install git+https://github.com/arkrow/pymusiclooper.git

# updating to new releases in either case can be done simply using:
pipx upgrade pymusiclooper
```

### option 3: installing using pip

traditional package installation method.

_note: fragile compared to an installation using `uv` or `pipx`. pymusiclooper may suddenly stop working if its dependencies were overwritten by another package (e.g. [issue #12](https://github.com/arkrow/pymusiclooper/issues/12))._

```sh
pip install pymusiclooper
```

## available commands

![pymusiclooper --help](https://github.com/arkrow/pymusiclooper/raw/master/img/pymusiclooper.svg)

note: further help and options can be found in each subcommand's help message (e.g. `pymusiclooper export-points --help`);
all commands and their `--help` message can be seen in [cli_readme.md](https://github.com/arkrow/pymusiclooper/blob/master/cli_readme.md)

**note**: using the interactive `-i` option is highly recommended, since the automatically chosen "best" loop point may not necessarily be the best one perceptually. as such, it is shown in all the examples. can be disabled if the `-i` flag is omitted. interactive mode is also available when batch processing.

## example usage

### play

```sh
# play the song on repeat with the best discovered loop point.
pymusiclooper -i play --path "track_name.mp3"


# audio can also be loaded from any stream supported by yt-dlp, e.g. youtube
# (also available for the `tag` and `split-audio` subcommands)
pymusiclooper -i play --url "https://www.youtube.com/watch?v=dqw4w9wgxcq"


# reads the loop metadata tags from an audio file and play it with the loop active
# using the loop start and end specified in the file (must be stored as samples)
pymusiclooper play-tagged --path "track_name.mp3" --tag-names loop_start loop_end
```

### export

_note: batch processing is available for all export subcommands. simply specify a directory instead of a file as the path to be used._

```sh
# split the audio track into intro, loop and outro files.
pymusiclooper -i split-audio --path "track_name.ogg"

# extend a track to an hour long (--extended-length accepts a number in seconds)
pymusiclooper -i extend --path "track_name.ogg" --extended-length 3600

# extend a track to an hour long, with its outro and in ogg format
pymusiclooper -i extend --path "track_name.ogg" --extended-length 3600 --disable-fade-out --format "ogg"

# export the best/chosen loop points directly to the terminal as sample points
pymusiclooper -i export-points --path "/path/to/track.wav"

# export all the discovered loop points directly to the terminal as sample points
# same output as interactive mode with loop values in samples, but without the formatting and pagination
# format: loop_start loop_end note_difference loudness_difference score
pymusiclooper export-points --path "/path/to/track.wav" --alt-export-top -1

# add metadata tags of the best discovered loop points to a copy of the input audio file
# (or all audio files in a directory, if a directory path is used instead)
pymusiclooper -i tag --path "track_name.mp3" --tag-names loop_start loop_end


# export the loop points (in samples) of all tracks in a particular directory to a loops.txt file
# (compatible with https://github.com/libertyernie/loopingaudioconverter/)
# note: each line in loop.txt follows the following format: {loop-start} {loop-end} {filename}
pymusiclooper -i export-points --path "/path/to/dir/" --export-to txt
```

### miscellaneous

#### finding more potential loops

```sh
# if the detected loop points are unsatisfactory, the brute force option `--brute-force`
# may yield better results.
## note: brute force mode checks the entire audio track instead of the detected beats.
## this leads to much longer runtime (may take several minutes).
## the program may appear frozen during this time while it is processing in the background.
pymusiclooper -i export-points --path "track_name.wav" --brute-force


# by default, the program further filters the initial discovered loop points
# according to internal criteria when there are >=100 possible pairs.
# if that is undesirable, it can be disabled using the `--disable-pruning` flag, e.g.
pymusiclooper -i export-points --path "track_name.wav" --disable-pruning
# note: can be used with --brute-force if desired
```

#### adjusting the loop length constraints

_by default, the minimum loop duration is 35% of the track length (excluding trailing silence), and the maximum is unbounded.
alternative constraints can be specified using the options below._

```sh
# if the loop is very long (or very short), a different minimum loop duration can be specified.
## --min-duration-multiplier 0.85 implies that the loop is at least 85% of the track,
## excluding trailing silence.
pymusiclooper -i split-audio --path "track_name.flac" --min-duration-multiplier 0.85

# alternatively, the loop length constraints can be specified in seconds
pymusiclooper -i split-audio --path "track_name.flac" --min-loop-duration 120 --max-loop-duration 150
```

#### searching near a desired start/end loop point

```sh
# if a desired loop point is already known, and you would like to extract the best loop
# positions in samples, the `--approx-loop-position` option can be used,
# which searches with +/- 2 seconds of the point specified.
# best used interactively. example using the `export-points` subcommand:
pymusiclooper -i export-points --path "/path/to/track.mp3" --approx-loop-position 20 210
## `--approx-loop-position 20 210` means the desired loop point starts around 20 seconds
## and loops back around the 210 seconds mark.
```

## acknowledgement

this project started out as a fork of [nolan nicholson](https://github.com/nolannicholson)'s project [looper](https://github.com/nolannicholson/looper/). although at this point only a few lines of code remain from that project due to adopting a completely different approach and implementation; this project would not have been possible without their initial contribution.

## version history

available at [changelog.md](changelog.md)

## star history

[![star history chart](https://api.star-history.com/svg?repos=arkrow/pymusiclooper&type=date)](https://www.star-history.com/#arkrow/pymusiclooper&date)
