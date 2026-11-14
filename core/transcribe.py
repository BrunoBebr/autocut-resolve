"""Transcription via whisper.cpp (the `whisper-cli` binary).

Reuses the ggml model already on disk (e.g. Vowen's large-v3-turbo) so nothing
needs downloading. Produces segments with start/end times in seconds, plus
word-level timing when requested (needed later for filler-word removal and
repeat detection).
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

from silence import find_ffmpeg


class Cancelled(Exception):
    """Raised when the user stops a transcription in progress."""


def _run(cmd, cancel=None):
    """Run a command, killable mid-flight via a `cancel` threading.Event."""
    if cancel is None:
        subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", check=True)
        return
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None:
        if cancel.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise Cancelled("zrušeno uživatelem")
        time.sleep(0.15)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

# whisper-cli ships with a few apps; check PATH then known locations.
_WHISPER_CANDIDATES = [
    "/Applications/Vowen.app/Contents/Resources/bin/whisper-cli",
    "/opt/homebrew/bin/whisper-cli",
    "/usr/local/bin/whisper-cli",
]

# ggml models already on disk (install.sh drops one in ~/.autocut/models).
_MODEL_CANDIDATES = [
    os.path.expanduser("~/.autocut/models/ggml-large-v3-turbo.bin"),
    os.path.expanduser("~/Library/Application Support/vowen/models/ggml-large-v3-turbo.bin"),
]


def find_whisper_cli():
    found = shutil.which("whisper-cli")
    if found:
        return found
    for c in _WHISPER_CANDIDATES:
        if os.path.exists(c) and os.access(c, os.X_OK):
            return c
    return None


def find_model():
    for c in _MODEL_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def extract_audio(media_path, out_wav, start_s=None, dur_s=None, cancel=None):
    """Extract mono 16 kHz WAV (what whisper.cpp expects) via ffmpeg."""
    ffmpeg = find_ffmpeg() or "ffmpeg"
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-y"]
    if start_s is not None:
        cmd += ["-ss", str(start_s)]
    if dur_s is not None:
        cmd += ["-t", str(dur_s)]
    cmd += ["-i", media_path, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_wav]
    _run(cmd, cancel=cancel)
    return out_wav


def transcribe_wav(wav_path, language="cs", model=None, whisper=None, max_len=0,
                   log=print, cancel=None, dtw=None):
    """Run whisper-cli on a WAV and return a list of segments.

    Each segment: {"start": float_s, "end": float_s, "text": str}.
    """
    whisper = whisper or find_whisper_cli()
    if not whisper:
        raise RuntimeError("whisper-cli not found (Vowen app, PATH, or Homebrew).")
    model = model or find_model()
    if not model:
        raise RuntimeError("No ggml whisper model found on disk.")

    out_base = os.path.splitext(wav_path)[0] + ".out"
    cmd = [
        whisper, "-m", model, "-l", language,
        "-oj", "-of", out_base,
        "-pp",                      # progress
    ]
    if max_len:
        cmd += ["-ml", str(max_len), "-sow"]
    if dtw:
        # DTW token times need flash attention off (`dtw` is the model preset).
        cmd += ["-ojf", "-dtw", dtw, "-nfa"]
    cmd += [wav_path]

    log(f"Transcribing ({os.path.basename(model)}, lang={language})...")
    # We read the result from the JSON file, so stdout isn't needed; _run lets the
    # user cancel a long transcription mid-flight.
    _run(cmd, cancel=cancel)

    json_path = out_base + ".json"
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    segments = []
    for seg in data.get("transcription", []):
        offs = seg.get("offsets", {})
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        item = {
            "start": offs.get("from", 0) / 1000.0,  # ms -> s
            "end": offs.get("to", 0) / 1000.0,
            "text": text,
        }
        # DTW anchors (centiseconds): first/last real token, a point inside the word.
        dtw = [t["t_dtw"] / 100.0 for t in seg.get("tokens", [])
               if t.get("t_dtw", -1) >= 0 and not t.get("text", "").startswith("[_")]
        if dtw:
            item["a0"], item["a1"] = dtw[0], dtw[-1]
        segments.append(item)
    return segments


_DTW_PRESETS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium",
                "medium.en", "large.v1", "large.v2", "large.v3", "large.v3.turbo"}


def _dtw_preset(model_path):
    """ggml-large-v3-turbo.bin -> 'large.v3.turbo'; None if whisper.cpp has no preset."""
    name = os.path.basename(model_path)
    name = re.sub(r"^ggml-|\.bin$", "", name)
    name = re.sub(r"-q\d.*$", "", name).replace("-", ".")
    return name if name in _DTW_PRESETS else None


def transcribe_media(media_path, language="cs", start_s=None, dur_s=None,
                     max_len=0, log=print, cancel=None):
    """Convenience: extract audio from any media file, then transcribe."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        extract_audio(media_path, wav, start_s=start_s, dur_s=dur_s, cancel=cancel)
        words = transcribe_wav(wav, language=language, max_len=max_len, log=log, cancel=cancel)
        preset = _dtw_preset(find_model() or "") if max_len else None
        if preset:
            # Word offsets from whisper.cpp touch each other and can be off by
            # seconds. A second DTW pass gives anchors that track the audio; it
            # decodes slightly differently (drops some words), so it only lends
            # its timing to the first pass via text alignment.
            log("Aligning word timing (DTW pass)...")
            try:
                ref = transcribe_wav(wav, language=language, max_len=max_len, log=log,
                                     cancel=cancel, dtw=preset)
                _attach_anchors(words, ref)
            except Cancelled:
                raise
            except Exception as exc:  # timing falls back to whisper offsets
                log(f"  DTW pass failed, using plain word times: {exc}")
        return words


def _norm(text):
    return re.sub(r"[^\w%]", "", text.lower())


def _attach_anchors(words, ref):
    """Copy a0/a1 from DTW words `ref` onto matching `words`; words with no match
    get anchors interpolated between their matched neighbours."""
    import difflib
    sm = difflib.SequenceMatcher(None, [_norm(w["text"]) for w in words],
                                 [_norm(w["text"]) for w in ref], autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            if "a0" in ref[j + k]:
                words[i + k]["a0"], words[i + k]["a1"] = ref[j + k]["a0"], ref[j + k]["a1"]
    i = 0
    while i < len(words):
        if "a0" in words[i]:
            i += 1
            continue
        j = i
        while j < len(words) and "a0" not in words[j]:
            j += 1
        lo = words[i - 1]["a1"] if i else words[i]["start"]
        hi = words[j]["a0"] if j < len(words) else words[j - 1]["end"]
        step = (hi - lo) / (j - i + 1)
        for k in range(i, j):
            words[k]["a0"] = words[k]["a1"] = lo + step * (k - i + 1)
        i = j


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else None
    segs = transcribe_media(path, dur_s=dur)
    for s in segs:
        print(f"[{s['start']:6.2f} - {s['end']:6.2f}] {s['text']}")
