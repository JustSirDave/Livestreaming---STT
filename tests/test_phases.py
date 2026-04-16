"""
Exit-condition tests for all backend phases (P1–P5).
Run from project root: env/Scripts/python.exe tests/test_phases.py
"""
import sys, asyncio
sys.path.insert(0, "server")

results = []

# ── Phase 1 — config ────────────────────────────────────────────────────────
def test_p1():
    import config
    assert config.SAMPLE_RATE == 16000
    assert config.FRAME_SIZE == 512        # min for Silero VAD at 16kHz
    assert config.VAD_FLUSH_FRAMES == 10
    assert config.ASR_BEAM_SIZE == 1
    results.append("P1  config.py                  PASS")

# ── Phase 2a — AudioProcessor ───────────────────────────────────────────────
def test_p2_audio_processor():
    from audio_processor import AudioProcessor
    import numpy as np
    ap = AudioProcessor()
    samples = (np.random.randn(16000) * 8000).astype(np.int16)
    frames = [samples[i:i+512].tobytes() for i in range(0, 16000, 512)]
    result = ap.process(frames, 16000)
    assert result.dtype == np.float32
    assert result.ndim == 1
    assert 0.9 <= result.max() <= 1.0
    results.append("P2a audio_processor.py         PASS")

# ── Phase 2b — VadEngine ────────────────────────────────────────────────────
def test_p2_vad():
    import numpy as np, torch
    from vad_engine import VadEngine

    class MockModel:
        def __call__(self, t, sr): return torch.tensor([[0.8]])
        def reset_states(self): pass

    class BrokenModel:
        def __call__(self, t, sr): raise RuntimeError("boom")
        def reset_states(self): pass

    frame = (np.random.randn(512) * 8000).astype(np.int16).tobytes()
    vad = VadEngine(model=MockModel(), threshold=0.5)
    assert vad.classify(frame) == True
    assert abs(vad.get_confidence(frame) - 0.8) < 1e-6
    assert VadEngine(model=BrokenModel()).classify(frame) == False
    vad.reset()  # must not raise
    results.append("P2b vad_engine.py              PASS")

# ── Phase 2c — PostProcessor ────────────────────────────────────────────────
def test_p2_post():
    from dataclasses import dataclass, field
    from post_processor import PostProcessor, TranscriptMessage

    @dataclass
    class ASRResult:
        text: str
        words: list = field(default_factory=list)
        confidence: float = 0.9
        duration: float = 1.0
        is_timeout: bool = False
        segment_id: str = "s1"

    pp = PostProcessor()
    assert pp.process(ASRResult(text="hello world")).text == "Hello world."
    assert pp.process(ASRResult(text="i have three meetings today")).text == "I have 3 meetings today."
    assert pp.process(ASRResult(text="fifty dollars")).text == "$50."
    r = pp.process(ASRResult(text="um i think uh three people"))
    assert "um" in r.text and "uh" in r.text and "3" in r.text

    class BrokenITN:
        def normalize(self, t, verbose=False): raise RuntimeError("x")

    assert isinstance(PostProcessor(itn_processor=BrokenITN()).process(ASRResult(text="fifty dollars")), TranscriptMessage)
    results.append("P2c post_processor.py          PASS")

# ── Phase 3 — ASRModel ──────────────────────────────────────────────────────
def test_p3():
    import numpy as np
    from unittest.mock import MagicMock, AsyncMock
    from concurrent.futures import ThreadPoolExecutor
    from asr_model import ASRModel, ASRResult

    MockWord = MagicMock()
    MockWord.word = "hello"; MockWord.start = 0.0; MockWord.end = 0.5; MockWord.probability = 0.95
    MockSeg = MagicMock()
    MockSeg.text = " hello world"; MockSeg.words = [MockWord]
    MockInfo = MagicMock()
    MockInfo.no_speech_prob = 0.05; MockInfo.duration = 1.5
    MockModel = MagicMock()
    MockModel.transcribe.return_value = ([MockSeg], MockInfo)

    asr = ASRModel(model=MockModel, executor=ThreadPoolExecutor(max_workers=2))
    result = asyncio.run(asr.transcribe(np.zeros(16000, dtype=np.float32), segment_id="s1"))
    assert isinstance(result, ASRResult)
    assert "hello" in result.text
    assert result.is_timeout == False
    assert result.confidence > 0.0
    results.append("P3  asr_model.py               PASS")

# ── Phase 4 — SessionManager ────────────────────────────────────────────────
def test_p4():
    import numpy as np
    from unittest.mock import MagicMock, AsyncMock
    from session_manager import SessionManager, VadState
    from asr_model import ASRResult
    from post_processor import TranscriptMessage

    mock_vad = MagicMock()
    mock_audio_proc = MagicMock()
    mock_post = MagicMock()
    mock_asr = MagicMock()

    # 512-sample frames to match FRAME_SIZE
    speech_frame  = (np.random.randn(512) * 8000).astype(np.int16).tobytes()
    silence_frame = (np.zeros(512, dtype=np.int16)).tobytes()
    call_count = [0]

    def vad_side(f):
        call_count[0] += 1
        return call_count[0] <= 20  # first 20 frames = speech

    mock_vad.classify.side_effect = vad_side
    mock_vad.reset = MagicMock()
    mock_audio_proc.process.return_value = np.zeros(16000, dtype=np.float32)
    mock_asr.transcribe = AsyncMock(return_value=ASRResult(
        text="hello world", words=[], confidence=0.9,
        duration=1.0, is_timeout=False, segment_id="s1"
    ))
    mock_post.process.return_value = TranscriptMessage(
        type="final", text="Hello world.", segment_id="s1",
        start=0.0, end=1.0, confidence=0.9
    )

    async def run():
        session = SessionManager()
        assert session.vad_state == VadState.SILENCE

        for _ in range(20):
            await session.audio_queue.put(speech_frame)
        for _ in range(20):
            await session.audio_queue.put(silence_frame)
        session.teardown()  # puts None on audio_queue

        await session.pipeline_loop(mock_vad, mock_asr, mock_post, mock_audio_proc)

        # Let pending tasks (flush, interim) run
        await asyncio.sleep(0.1)

        # Collect all messages from transcript_queue
        messages = []
        while not session.transcript_queue.empty():
            try:
                messages.append(session.transcript_queue.get_nowait())
            except Exception:
                break

        types = [m["type"] for m in messages if m is not None]
        assert "final" in types, f"Expected 'final' in messages, got: {types}"
        assert len(session.segments) == 1
        assert session.segments[0].text == "Hello world."
        assert mock_vad.reset.called  # VAD reset called on flush

    asyncio.run(run())
    results.append("P4  session_manager.py         PASS")

# ── Phase 5 — Health check (live server) ────────────────────────────────────
def test_p5():
    import urllib.request, json
    try:
        resp = urllib.request.urlopen("http://localhost:8000/health", timeout=3)
        data = json.loads(resp.read())
        assert data == {"status": "ok"}
        results.append("P5  main.py (server live)      PASS")
    except Exception as e:
        results.append(f"P5  main.py (server live)      FAIL  ({e})")

# ── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for fn in [test_p1, test_p2_audio_processor, test_p2_vad, test_p2_post, test_p3, test_p4, test_p5]:
        try:
            fn()
        except Exception as e:
            results.append(f"{fn.__name__:<30} FAIL  {e}")

    print("\n=== STT Backend — Phase Test Results ===")
    for r in results:
        print(" ", r)
    print()
