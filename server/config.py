# All tunable constants for the STT pipeline.
# No logic here — every other component imports these by name.

SAMPLE_RATE = 16000          # Hz — audio sample rate; must match AudioContext rate in client and VAD/ASR input contract
FRAME_SIZE = 512             # samples — PCM frames per chunk (32ms at 16kHz); minimum for Silero VAD at 16kHz (sr/31.25=512)
VAD_THRESHOLD = 0.5          # probability [0.0–1.0] — Silero confidence cutoff; lower = more sensitive, higher = more conservative
VAD_FLUSH_FRAMES = 10        # frames — consecutive silent frames before flushing segment to ASR (~320ms at 32ms/frame)
ASR_TIMEOUT = 10.0           # seconds — max inference time per segment; exceeded → empty ASRResult with is_timeout=True
ASR_BEAM_SIZE = 1            # beams — 1 = greedy decode (fastest); increase for accuracy at cost of speed
INTERIM_INTERVAL_SEC = 0.8   # seconds — how often the interim worker re-transcribes during speech
INTERIM_TAIL_FRAMES = 62     # frames — only feed the last ~2s of audio to interim (faster inference)
WS_QUEUE_MAX = 50            # frames — max depth of audio_queue (~1s buffer); frames dropped oldest-first when full
MIN_SEGMENT_DURATION = 0.3   # seconds — segments shorter than this are discarded before ASR
MAX_SEGMENT_DURATION = 30.0  # seconds — segments longer than this are rejected by AudioProcessor._validate()
