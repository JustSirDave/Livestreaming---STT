# All tunable constants for the STT pipeline.
# No logic here — every other component imports these by name.

SAMPLE_RATE = 16000          # Hz — audio sample rate; must match AudioContext rate in client and VAD/ASR input contract
FRAME_SIZE = 512             # samples — PCM frames per chunk (32ms at 16kHz); minimum for Silero VAD at 16kHz (sr/31.25=512)
VAD_THRESHOLD = 0.4          # probability [0.0–1.0] — Silero confidence cutoff; lower = more sensitive, higher = more conservative
VAD_FLUSH_FRAMES = 8         # frames — consecutive silent frames before flushing segment to ASR (~256ms at 32ms/frame)
ASR_TIMEOUT = 10.0           # seconds — max inference time per segment; exceeded → empty ASRResult with is_timeout=True
ASR_BEAM_SIZE = 1            # beams — 1 = greedy decode (fastest); increase for accuracy at cost of speed
INTERIM_INTERVAL_SEC = 1.5   # seconds — how often the interim worker re-transcribes during speech (Groq API ~300-600ms RTT)
INTERIM_TAIL_FRAMES = 62     # frames — last ~2s fed to tiny.en interim (~150ms inference, keeps up with 0.25s interval)
WS_QUEUE_MAX = 50            # frames — max depth of audio_queue (~1s buffer); frames dropped oldest-first when full
MIN_SEGMENT_DURATION = 0.1   # seconds — segments shorter than this are discarded before ASR
INTERIM_INTERVAL_FRAMES = 25 # frames — fire interim ASR every N frames during speech (~500ms at 32ms/frame)
MAX_SPEECH_FRAMES = 150      # frames — force-flush mid-speech after this many frames (~4.8s)
MAX_SEGMENT_DURATION = 30.0  # seconds — segments longer than this are rejected by AudioProcessor._validate()
VAD_MAX_SPEECH_SEC = 12.0    # seconds — force-flush mid-speech after this long to avoid hitting MAX_SEGMENT_DURATION
