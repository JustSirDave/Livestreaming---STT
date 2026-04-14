// C1 — AudioCapture
// Captures mic at 16kHz, converts float32→int16 client-side,
// emits 320-sample (20ms) binary frames via onFrame callback.

const SAMPLE_RATE = 16000;
const FRAME_SIZE = 512; // samples per frame (32ms at 16kHz) — must match server/config.py

export class AudioCapture {
  constructor(onFrame) {
    this.onFrame = onFrame;   // callback(Int16Array)
    this.context = null;
    this.processor = null;
    this.stream = null;
    this._overflow = new Int16Array(0); // leftover samples between processor callbacks
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    this.context = new AudioContext({ sampleRate: SAMPLE_RATE });
    const source = this.context.createMediaStreamSource(this.stream);

    // ScriptProcessorNode — 4096 samples per callback, 1 channel in/out
    this.processor = this.context.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = (e) => this._onAudio(e);

    source.connect(this.processor);
    this.processor.connect(this.context.destination);
  }

  stop() {
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.context) {
      this.context.close();
      this.context = null;
    }
    this._overflow = new Int16Array(0);
  }

  _onAudio(event) {
    const float32 = event.inputBuffer.getChannelData(0);

    // Convert float32 → int16
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    // Prepend any leftover samples from last callback
    const combined = new Int16Array(this._overflow.length + int16.length);
    combined.set(this._overflow, 0);
    combined.set(int16, this._overflow.length);

    // Emit complete FRAME_SIZE-sample frames
    let offset = 0;
    while (offset + FRAME_SIZE <= combined.length) {
      this.onFrame(combined.slice(offset, offset + FRAME_SIZE));
      offset += FRAME_SIZE;
    }

    // Keep remainder for next callback
    this._overflow = combined.slice(offset);
  }
}
