import os
import re
import sys
import time
import shutil
import atexit
import signal
import tempfile
import subprocess
import threading
import queue
import urllib.request
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import pyttsx3
import yfinance as yf
from openai import OpenAI


# ---------------------------------------------------------------------------
# SILENCE macOS PaMacCore C-level audio errors
# Redirects the raw C-level stderr (fd 2) to /dev/null so OS-level audio
# glitch messages never reach the terminal. Python's sys.stderr stays
# intact, so tracebacks and Python-level errors still print normally.
# ---------------------------------------------------------------------------
if sys.platform == "darwin":
    _real_stderr = os.fdopen(os.dup(2), "w")
    _devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull_fd, 2)
    os.close(_devnull_fd)
    # Point Python's sys.stderr at the saved copy so Python errors still show
    sys.stderr = _real_stderr

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
MY_NAME = "Viraat"
SPEECH_RATE = 190
VOICE = "Daniel"  # macOS built-in voice. Run `say -v '?'` in terminal to see all options.
AI_MODEL = "llama3.1:8b"  # smarter 8B model — still runs fine on a MacBook Air
LISTEN_AMPLITUDE_THRESHOLD = 0.04  # how loud you must be to "wake up" JARVIS
BARGE_IN_AMPLITUDE_THRESHOLD = 0.06  # unused — kept for future interrupt support
SAMPLE_RATE = 16000
RECORD_SECONDS = 10  # max recording time (will stop early on silence)
SILENCE_TIMEOUT = 1.5  # seconds of silence before stopping recording
SPEECH_AMPLITUDE_THRESHOLD = 0.02  # minimum volume to count as "talking"
STOCK_SYMBOLS = {
    "tesla": "TSLA", "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
    "google": "GOOGL", "amazon": "AMZN", "meta": "META", "amd": "AMD",
}
SHUTDOWN_WORDS = [
    "shutdown the terminal", "turn off the software", "power down mainframe",
    "shut down", "shutdown", "power down", "turn off jarvis", "exit program",
]

# ---------------------------------------------------------------------------
# SHARED STATE
# ---------------------------------------------------------------------------
is_speaking = threading.Event()      # set while JARVIS is talking
interrupt_now = threading.Event()    # set to instantly stop JARVIS talking
keep_running = threading.Event()
keep_running.set()
chat_history = []
history_lock = threading.Lock()
OLLAMA_URL = "http://localhost:11434/v1"
client = OpenAI(
    base_url=OLLAMA_URL,
    api_key="ollama",  # Ollama doesn't check this, but the client requires it
)

# ---------------------------------------------------------------------------
# OLLAMA AUTO-START
# ---------------------------------------------------------------------------
_ollama_process = None


def _is_ollama_running():
    """Check if Ollama is already responding."""
    try:
        urllib.request.urlopen("http://localhost:11434/", timeout=2)
        return True
    except Exception:
        return False


def _model_exists(model_name):
    """Check if a model is already pulled in Ollama."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return any(model_name in m for m in models)
    except Exception:
        return False


def ensure_ollama():
    """Start Ollama automatically and pull the model if needed. No separate terminal required."""
    global _ollama_process

    # Start Ollama if it's not already running
    if not _is_ollama_running():
        print("🔄 Starting Ollama...")
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for it to be ready (up to 15 seconds)
        for _ in range(30):
            if _is_ollama_running():
                break
            time.sleep(0.5)
        else:
            print(
                "\n⚠️  Ollama failed to start.\n"
                "Make sure it's installed: https://ollama.com/download\n"
            )
            sys.exit(1)
        print("✅ Ollama started!")
    else:
        print("✅ Ollama is already running.")

    # Pull the model if it's not downloaded yet
    if not _model_exists(AI_MODEL):
        print(f"📦 Downloading {AI_MODEL} (one-time, may take a few minutes)...")
        subprocess.run(["ollama", "pull", AI_MODEL])
        print(f"✅ {AI_MODEL} ready!")
    else:
        print(f"✅ {AI_MODEL} is already downloaded.")

    # Clean up Ollama when JARVIS exits
    if _ollama_process:
        def _stop_ollama():
            if _ollama_process and _ollama_process.poll() is None:
                _ollama_process.terminate()
        atexit.register(_stop_ollama)


# ---------------------------------------------------------------------------
# TEMP FILE
# ---------------------------------------------------------------------------
# Temp file for voice recording — auto-cleaned on exit
_temp_wav_fd, TEMP_WAV_PATH = tempfile.mkstemp(suffix=".wav", prefix="jarvis_voice_")
os.close(_temp_wav_fd)


def _cleanup_temp_files():
    """Remove temp wav file on exit, even on crash."""
    try:
        if os.path.exists(TEMP_WAV_PATH):
            os.remove(TEMP_WAV_PATH)
    except OSError:
        pass


atexit.register(_cleanup_temp_files)

# ---------------------------------------------------------------------------
# TEXT-TO-SPEECH BACKEND
# ---------------------------------------------------------------------------
SAY_BIN = shutil.which("say")
_say_lock = threading.Lock()       # ensures only one `say` process runs at a time
_active_say_proc = None            # tracks the currently running `say` process


def _kill_active_say():
    """Forcefully kill any running `say` process to prevent overlapping voices."""
    global _active_say_proc
    if _active_say_proc and _active_say_proc.poll() is None:
        _active_say_proc.terminate()
        try:
            _active_say_proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _active_say_proc.kill()
    _active_say_proc = None


# ---------------------------------------------------------------------------
# SPEAKING (with real-time interrupt support)
# ---------------------------------------------------------------------------
def clean_for_speech(text):
    """Strip markdown, parentheticals, and extra whitespace for clean TTS."""
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\*.*?\*", "", text)
    text = text.replace("'", "").replace('"', "")
    return " ".join(text.split()).strip()


def speak_one_sentence(sentence):
    """Speaks a single sentence, checking every moment if we should stop."""
    global _active_say_proc
    clean = clean_for_speech(sentence)
    if not clean:
        return
    print(f"[JARVIS]: {clean}")
    if SAY_BIN:
        for attempt in range(2):  # retry once if say crashes (audio device conflict)
            with _say_lock:
                _kill_active_say()
                _active_say_proc = subprocess.Popen(
                    [SAY_BIN, "-v", VOICE, "-r", str(SPEECH_RATE), clean],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                proc = _active_say_proc
            # If say exits in under 0.2s, it crashed — wait and retry
            time.sleep(0.2)
            if proc.poll() is not None and proc.returncode != 0 and attempt == 0:
                time.sleep(0.5)  # let audio device settle
                continue
            break
        while proc.poll() is None:
            if interrupt_now.is_set():
                with _say_lock:
                    _kill_active_say()
                print("\n🛑 Interrupted!")
                return
            time.sleep(0.05)
    else:
        # Fallback for Windows/Linux where `say` doesn't exist.
        done = threading.Event()
        local_engine = pyttsx3.init()
        local_engine.setProperty("rate", SPEECH_RATE)

        def _talk():
            local_engine.say(clean)
            local_engine.runAndWait()
            done.set()

        talker = threading.Thread(target=_talk, daemon=True)
        talker.start()
        while not done.is_set():
            if interrupt_now.is_set():
                try:
                    local_engine.stop()
                except Exception:
                    pass
                print("\n🛑 Interrupted!")
                return
            time.sleep(0.05)


def speak_streaming_reply(sentence_queue):
    """
    Pulls finished sentences off the queue and speaks them one at a time,
    for as long as JARVIS is generating a reply (or until interrupted).
    """
    is_speaking.set()
    interrupt_now.clear()
    try:
        while True:
            if interrupt_now.is_set():
                break
            try:
                sentence = sentence_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if sentence is None:  # our signal that generation is finished
                break
            speak_one_sentence(sentence)
            if interrupt_now.is_set():
                break
    finally:
        # Drain any remaining sentences from the queue so nothing leaks
        while not sentence_queue.empty():
            try:
                sentence_queue.get_nowait()
            except queue.Empty:
                break
        is_speaking.clear()
        interrupt_now.clear()


# ---------------------------------------------------------------------------
# THE AI BRAIN (streamed for speed)
# ---------------------------------------------------------------------------
SENTENCE_END_PATTERN = re.compile(r"(?<=[.!?])\s+")


def ask_gpt_and_speak_as_it_thinks(user_message):
    """
    Sends the message to GPT and, as the reply streams in word by word,
    breaks it into sentences and hands each finished sentence to the
    speaking thread immediately.
    """
    with history_lock:
        history_copy = list(chat_history)
    messages = [
        {
            "role": "system",
            "content": (
                f"You are JARVIS, a practical and honest AI assistant for {MY_NAME}. "
                "CRITICAL RULES: 1) NEVER make up information, events, schedules, or facts you don't actually know. "
                "2) You are a text-based AI - you cannot monitor calendars, send refreshments, control rooms, or take physical actions. "
                "3) Do NOT roleplay or pretend to do things you cannot actually do. "
                "4) If you don't know something, just say so honestly. "
                "5) Be genuinely helpful - answer questions accurately, assist with tasks, explain concepts, brainstorm ideas, write content, solve problems. "
                f"6) Keep answers short and conversational, 2-4 sentences. "
                "7) No parentheses, asterisks, or stage directions - pure spoken words. "
                f"8) Address the user as 'sir' when appropriate, but don't overdo it."
            ),
        }
    ]
    messages.extend(history_copy)
    messages.append({"role": "user", "content": user_message})
    sentence_queue = queue.Queue()
    speaker_thread = threading.Thread(
        target=speak_streaming_reply, args=(sentence_queue,), daemon=True
    )
    speaker_thread.start()
    full_reply = ""
    buffer = ""
    try:
        stream = client.chat.completions.create(
            model=AI_MODEL,
            messages=messages,
            stream=True,
            temperature=0.3,  # lower = more factual, less hallucination
        )
        for chunk in stream:
            if interrupt_now.is_set():
                break
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            buffer += delta
            full_reply += delta
            # whenever we've got a complete sentence, send it off to be spoken
            match = SENTENCE_END_PATTERN.search(buffer)
            while match:
                finished_sentence = buffer[: match.end()].strip()
                if finished_sentence:
                    sentence_queue.put(finished_sentence)
                buffer = buffer[match.end():]
                match = SENTENCE_END_PATTERN.search(buffer)
    except Exception as error:
        print(f"(AI request failed: {error})")
        sentence_queue.put(f"Sorry sir {MY_NAME}, I'm having trouble thinking right now.")
    if buffer.strip() and not interrupt_now.is_set():
        sentence_queue.put(buffer.strip())
    sentence_queue.put(None)  # tells the speaker thread "no more sentences coming"
    speaker_thread.join()
    return full_reply.strip()


# ---------------------------------------------------------------------------
# STOCK PRICES
# ---------------------------------------------------------------------------
def get_stock_price(symbol):
    """Fetch the latest closing price for a stock symbol."""
    try:
        data = yf.Ticker(symbol).history(period="1d")
        if not data.empty:
            return round(float(data["Close"].iloc[-1]), 2)
    except Exception as error:
        print(f"(Stock lookup failed: {error})")
    return None


# ---------------------------------------------------------------------------
# DECIDING WHAT TO DO WITH WHAT YOU SAID
# ---------------------------------------------------------------------------
def handle_user_speech(text):
    """Route user speech to the appropriate handler."""
    if not text or not text.strip():
        return
    lowered = text.lower()

    # Check for shutdown commands
    if any(phrase in lowered for phrase in SHUTDOWN_WORDS):
        speak_one_sentence(f"Powering down. Goodbye, sir {MY_NAME}.")
        keep_running.clear()
        return

    # Check for stock price queries
    finance_words = ["stock", "price", "market", "ticker", "cost"]
    if any(word in lowered for word in finance_words):
        for name, symbol in STOCK_SYMBOLS.items():
            if name in lowered:
                price = get_stock_price(symbol)
                if price is not None:
                    speak_one_sentence(f"{name.title()} is at {price} dollars, sir.")
                else:
                    speak_one_sentence(f"Couldn't fetch the price for {name.title()}, sir.")
                # Always save to history for stock queries too
                with history_lock:
                    chat_history.append({"role": "user", "content": text})
                    chat_history.append({"role": "assistant", "content": f"{name.title()} is at {price} dollars." if price else "Price unavailable."})
                    _trim_history()
                return

    # Default: send to AI
    reply = ask_gpt_and_speak_as_it_thinks(text)
    with history_lock:
        chat_history.append({"role": "user", "content": text})
        chat_history.append({"role": "assistant", "content": reply})
        _trim_history()


def _trim_history():
    """Keep only the last 8 messages so the conversation doesn't get huge. Must hold history_lock."""
    while len(chat_history) > 8:
        chat_history.pop(0)


# ---------------------------------------------------------------------------
# LISTENING
# ---------------------------------------------------------------------------
def record_and_transcribe():
    """Record from the mic with smart silence detection, then transcribe."""
    print("\n🎙️  Listening...")

    chunk_duration = 0.5  # seconds per chunk
    chunk_samples = int(chunk_duration * SAMPLE_RATE)
    max_chunks = int(RECORD_SECONDS / chunk_duration)
    silent_chunks = 0
    speech_started = False
    max_silent_chunks = int(SILENCE_TIMEOUT / chunk_duration)
    audio_chunks = []

    for _ in range(max_chunks):
        if not keep_running.is_set():
            break
        try:
            chunk = sd.rec(chunk_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
        except Exception as error:
            print(f"(Microphone hiccup, skipping: {error})")
            return ""

        volume = float(np.linalg.norm(chunk) / np.sqrt(chunk.size))

        if volume > SPEECH_AMPLITUDE_THRESHOLD:
            speech_started = True
            silent_chunks = 0
            audio_chunks.append((chunk * 32767).astype("int16"))
        elif speech_started:
            silent_chunks += 1
            audio_chunks.append((chunk * 32767).astype("int16"))
            if silent_chunks >= max_silent_chunks:
                break  # user stopped talking

    if not audio_chunks:
        return ""  # never heard any speech

    # Concatenate all audio chunks and transcribe
    full_recording = np.concatenate(audio_chunks, axis=0)
    wav.write(TEMP_WAV_PATH, SAMPLE_RATE, full_recording)

    recognizer = sr.Recognizer()
    text = ""
    try:
        with sr.AudioFile(TEMP_WAV_PATH) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(f'Heard: "{text}"')
    except sr.UnknownValueError:
        pass  # silence or unintelligible
    except sr.RequestError as error:
        print(f"(Speech recognition error: {error})")
    return text



def wait_until_someone_talks():
    """Block until the mic detects speech above the amplitude threshold."""
    chunk_samples = int(0.4 * SAMPLE_RATE)
    while keep_running.is_set():
        try:
            chunk = sd.rec(
                chunk_samples, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
            )
            sd.wait()
        except Exception:
            time.sleep(0.1)
            continue
        volume = np.linalg.norm(chunk) / np.sqrt(chunk.size)
        if volume > LISTEN_AMPLITUDE_THRESHOLD:
            return True
    return False


# ---------------------------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------------------------
def clear_screen():
    """Clear the terminal screen in a cross-platform way."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")


def main():
    clear_screen()
    print("📡 Starting JARVIS (Ollama Edition - Local)...\n")

    # Auto-start Ollama and ensure model is available
    ensure_ollama()

    # Check that the chosen voice is installed on macOS
    if SAY_BIN:
        try:
            result = subprocess.run(
                [SAY_BIN, "-v", "?"],
                capture_output=True, text=True, timeout=5,
            )
            installed_voices = result.stdout
            if VOICE not in installed_voices:
                print(
                    f"\n\u26a0\ufe0f  Voice '{VOICE}' is not installed.\n"
                    f"Install it: System Settings > Accessibility > Spoken Content > System Voice > Manage Voices\n"
                    f"Search for '{VOICE}' and download it.\n"
                    f"Falling back to the default voice for now.\n"
                )
        except Exception:
            pass
    print()

    speak_one_sentence(f"Systems online, sir {MY_NAME}. I'm listening.")

    try:
        while keep_running.is_set():
            if is_speaking.is_set():
                time.sleep(0.1)
                continue
            if wait_until_someone_talks():
                text = record_and_transcribe()
                handle_user_speech(text)
                time.sleep(0.3)  # brief pause so mic and speaker don't fight
    finally:
        keep_running.clear()
        _kill_active_say()  # stop any lingering speech on exit


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        keep_running.clear()
        print("\nShutting down. Bye!")

