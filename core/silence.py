"""Silence detection via ffmpeg's silencedetect filter.

No machine learning needed here -- this is pure audio-energy analysis, which is
fast and reliable for the "auto cut silences" feature. Returns silent intervals
in absolute seconds relative to the start of the media file.
"""

import os
import re
import shutil
import subprocess

_SILENCE_RE = re.compile(r"silence_(start|end):\s*([\-0-9.]+)")

# Resolve launches scripts with a minimal PATH that omits Homebrew, so we look
# in the usual install locations too.
_FFMPEG_CANDIDATES = [
    "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew
    "/usr/local/bin/ffmpeg",     # Intel Homebrew
    "/opt/local/bin/ffmpeg",     # MacPorts
]


def find_ffmpeg():
    """Return a usable ffmpeg path, or None. Checks PATH then common locations."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in _FFMPEG_CANDIDATES:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def ffmpeg_available():
    return find_ffmpeg() is not None


def detect_silences(media_path, noise_db=-30, min_silence_dur=0.5):
    """Run ffmpeg silencedetect over the whole file.

    Args:
        media_path: path to the source media file on disk.
        noise_db: anything quieter than this (dBFS) counts as silence.
        min_silence_dur: minimum length (s) for a gap to be treated as silence.

    Returns:
        List of (start_s, end_s) tuples in absolute seconds.
    """
    ffmpeg = find_ffmpeg() or "ffmpeg"
    cmd = [
        ffmpeg, "-hide_banner", "-nostats",
        "-vn",                    # ignore video: we only analyse audio (much faster)
        "-i", media_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_dur}",
        "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    # silencedetect writes to stderr regardless of success.
    text = proc.stderr or ""

    silences = []
    pending_start = None
    for kind, value in _SILENCE_RE.findall(text):
        value = float(value)
        if kind == "start":
            pending_start = value
        elif pending_start is not None:  # "end"
            silences.append((pending_start, value))
            pending_start = None
    # An unterminated silence_start means silence runs to EOF; ignore (handled by
    # clip end clamping in the caller).
    return silences


# Loudness envelope
# silencedetect needs a fixed dB threshold, but recordings differ wildly: a quiet
# mic puts speech at -45..-30 dBFS, so the old -30 dB default called most of the
# speech "silence". The envelope lets us place the threshold per recording and
# snap cuts to real pauses instead of whisper's approximate word times.

FRAME_S = 0.01  # envelope resolution (10 ms)


def envelope(media_path, start_s=0.0, dur_s=None):
    """Per-10 ms loudness (dBFS) of the media's audio, starting at `start_s`."""
    import array
    import math
    from operator import mul
    ffmpeg = find_ffmpeg() or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-ss", str(start_s)]
    if dur_s is not None:
        cmd += ["-t", str(dur_s)]
    cmd += ["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    pcm = array.array("h")
    pcm.frombytes(raw[: len(raw) // 2 * 2])
    n = int(16000 * FRAME_S)
    dot = getattr(math, "sumprod", None) or (lambda a, b: sum(map(mul, a, b)))
    out = []
    for i in range(0, len(pcm) - n + 1, n):
        f = pcm[i:i + n]
        out.append(10 * math.log10(dot(f, f) / n / (32768.0 ** 2) + 1e-12))
    return out


def auto_threshold(env, ceiling_db=-30, below_speech_db=15):
    """Silence threshold `below_speech_db` under the speech level (90th pct), so
    breaths and room tone between phrases count as pause; kept 6 dB above the
    noise floor (10th pct) and never above `ceiling_db` (the panel's Práh)."""
    if not env:
        return ceiling_db
    s = sorted(env)
    floor, speech = s[len(s) // 10], s[len(s) * 9 // 10]
    return min(ceiling_db, max(floor + 6, speech - below_speech_db))


def pauses(env, thr_db, min_dur_s, t0=0.0):
    """(start_s, end_s) runs of the envelope below `thr_db` lasting >= min_dur_s."""
    out, run = [], None
    for i, v in enumerate(env + [0.0]):  # sentinel closes a trailing run
        if v < thr_db:
            run = i if run is None else run
        elif run is not None:
            if (i - run) * FRAME_S >= min_dur_s:
                out.append((t0 + run * FRAME_S, t0 + i * FRAME_S))
            run = None
    return out
