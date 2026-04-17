// Runs in the dedicated audio thread — not throttled by browser tab visibility.
// Buffers incoming float32 samples into 512-sample frames, converts to int16,
// and posts each frame to the main thread via this.port.

const FRAME_SIZE = 512;

class FrameProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(FRAME_SIZE);
    this._pos = 0;
  }

  process(inputs) {
    const channel = inputs[0][0];
    if (!channel) return true;

    let offset = 0;
    while (offset < channel.length) {
      const toCopy = Math.min(FRAME_SIZE - this._pos, channel.length - offset);
      this._buf.set(channel.subarray(offset, offset + toCopy), this._pos);
      this._pos += toCopy;
      offset += toCopy;

      if (this._pos === FRAME_SIZE) {
        const int16 = new Int16Array(FRAME_SIZE);
        for (let i = 0; i < FRAME_SIZE; i++) {
          const s = Math.max(-1, Math.min(1, this._buf[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        this.port.postMessage(int16.buffer, [int16.buffer]);
        this._pos = 0;
      }
    }
    return true;
  }
}

registerProcessor("frame-processor", FrameProcessor);
