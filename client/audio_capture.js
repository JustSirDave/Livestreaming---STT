// C1 — AudioCapture
// Captures mic at 16kHz via AudioWorklet (non-throttled, non-deprecated).
// Emits 512-sample int16 frames via onFrame callback.
// Emits smoothed RMS amplitude (0.0–1.0) via onAmplitude callback.

const SAMPLE_RATE = 16000;
const AMP_SMOOTH  = 0.60;
const AMP_SCALE   = 14.0;

export class AudioCapture {
  constructor(onFrame, onAmplitude) {
    this.onFrame     = onFrame;
    this.onAmplitude = onAmplitude;
    this.context     = null;
    this._node       = null;
    this.stream      = null;
    this._smoothAmp  = 0.0;
    this._muted      = false;
  }

  get muted() { return this._muted; }

  mute() {
    this._muted = true;
    if (this.onAmplitude) this.onAmplitude(0.0);
  }

  unmute() {
    this._muted = false;
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
    await this.context.audioWorklet.addModule("./audio_worklet_processor.js");

    const source = this.context.createMediaStreamSource(this.stream);
    this._node   = new AudioWorkletNode(this.context, "frame-processor");

    this._node.port.onmessage = (e) => this._onFrame(e.data);

    source.connect(this._node);
    // Route through a muted gain node to keep the audio graph alive without
    // playing mic audio back to speakers, which would trigger AEC cancellation.
    const silentGain = this.context.createGain();
    silentGain.gain.value = 0;
    this._node.connect(silentGain);
    silentGain.connect(this.context.destination);

    // OS-level mic mute (keyboard hardware key)
    this.stream.getTracks().forEach((track) => {
      track.onmute   = () => console.warn("[AudioCapture] OS mic muted");
      track.onunmute = () => {
        console.log("[AudioCapture] OS mic unmuted — resuming context");
        this._resumeIfSuspended();
      };
    });

    // Resume if AudioContext suspended for any reason
    this.context.onstatechange = () => this._resumeIfSuspended();
  }

  // Called by index.html visibilitychange handler when tab regains focus
  resumeIfSuspended() {
    this._resumeIfSuspended();
  }

  stop() {
    if (this._node) {
      this._node.disconnect();
      this._node = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.context) {
      this.context.close();
      this.context = null;
    }
    this._smoothAmp = 0.0;
    if (this.onAmplitude) this.onAmplitude(0.0);
  }

  _resumeIfSuspended() {
    if (this.context && this.context.state === "suspended") {
      this.context.resume();
    }
  }

  _onFrame(buffer) {
    const int16 = new Int16Array(buffer);

    if (this.onAmplitude) {
      let sum = 0;
      for (let i = 0; i < int16.length; i++) {
        const s = int16[i] / 32768;
        sum += s * s;
      }
      const rms = Math.sqrt(sum / int16.length);
      this._smoothAmp = AMP_SMOOTH * this._smoothAmp + (1 - AMP_SMOOTH) * rms;
      this.onAmplitude(Math.min(1.0, this._smoothAmp * AMP_SCALE));
    }

    if (!this._muted) this.onFrame(int16);
  }
}
