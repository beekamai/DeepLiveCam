"""Beeps and spoken prompts for the calibration flow.

The user looks away from the screen while turning their head, so the
countdown, the capture and the next pose are announced by sound.  Tones use
the OS beep; speech uses the system synthesiser (SAPI on Windows, ``say`` on
macOS, ``spd-say``/``espeak`` on Linux) and is always English — the voices
shipped with the OS are reliable only there.  Everything runs on one
background thread so cues never block the UI, and a new utterance replaces
one still playing.
"""

from __future__ import annotations

import platform
import queue
import shutil
import subprocess
import sys
import threading
from typing import Optional, Sequence, Tuple

import modules.globals

# (frequency Hz, duration ms) sequences per cue
TONES = {
    "tick": ((880, 60),),
    "arm": ((660, 90), (880, 90)),
    "captured": ((880, 80), (1175, 80), (1568, 160)),
    "done": ((1047, 120), (1319, 120), (1568, 120), (2093, 240)),
}

_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
_thread: Optional[threading.Thread] = None
_speaking: Optional[subprocess.Popen] = None
_lock = threading.Lock()


def _play_tones(tones: Sequence[Tuple[int, int]]) -> None:
    if sys.platform == "win32":
        import winsound

        for freq, ms in tones:
            winsound.Beep(freq, ms)
    else:
        sys.stdout.write("\a")
        sys.stdout.flush()


def _speech_command(text: str) -> Optional[list]:
    system = platform.system()
    if system == "Windows":
        safe = text.replace("'", "''")
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                f"$s.Rate = 1; $s.Speak('{safe}')"]
    if system == "Darwin":
        return ["say", text]
    for tool in ("spd-say", "espeak"):
        if shutil.which(tool):
            return [tool, text]
    return None


def _speak(text: str) -> None:
    global _speaking
    command = _speech_command(text)
    if command is None:
        return
    with _lock:
        if _speaking is not None and _speaking.poll() is None:
            _speaking.kill()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        _speaking = subprocess.Popen(command, creationflags=flags,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _worker() -> None:
    while True:
        kind, payload = _queue.get()
        try:
            if kind == "tone":
                _play_tones(payload)
            elif kind == "say":
                _speak(payload)
        except Exception as error:
            print(f"[audio] {kind} failed: {error}")


def _ensure_thread() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_worker, name="audio-cues", daemon=True)
        _thread.start()


def tone(name: str) -> None:
    """Play a named tone sequence when sound cues are on."""
    if not getattr(modules.globals, "calibration_sounds", True):
        return
    tones = TONES.get(name)
    if tones:
        _ensure_thread()
        _queue.put(("tone", tones))


def say(text: str) -> None:
    """Speak ``text`` (English) when voice prompts are on."""
    if not getattr(modules.globals, "calibration_voice", False) or not text:
        return
    _ensure_thread()
    _queue.put(("say", text))


def stop() -> None:
    """Silence a prompt still playing (dialog closed)."""
    with _lock:
        if _speaking is not None and _speaking.poll() is None:
            _speaking.kill()
