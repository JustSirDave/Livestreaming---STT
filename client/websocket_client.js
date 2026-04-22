// C2 — WebSocketClient
// Sends binary audio frames upstream, receives JSON transcript downstream.
// Reconnects automatically with exponential backoff.

const WS_URL = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
const BACKOFF_BASE_MS = 500;
const BACKOFF_MAX_MS = 16000;
const BACKOFF_MULTIPLIER = 2;

export class WebSocketClient {
  constructor(onMessage) {
    this.onMessage = onMessage; // callback(parsedObject)
    this.ws = null;
    this._backoff = BACKOFF_BASE_MS;
    this._stopped = false;
    this._connectTimer = null;
  }

  connect() {
    this._stopped = false;
    this.ws = new WebSocket(WS_URL);
    this.ws.binaryType = "arraybuffer";

    this.ws.onopen = () => {
      console.log("[WS] connected");
      this._backoff = BACKOFF_BASE_MS; // reset on success
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        this.onMessage(msg);
      } catch (e) {
        console.warn("[WS] unparseable message", e);
      }
    };

    this.ws.onclose = (event) => {
      console.log(`[WS] closed (code=${event.code}) — reconnecting in ${this._backoff}ms`);
      if (!this._stopped) this._scheduleReconnect();
    };

    this.ws.onerror = (err) => {
      console.error("[WS] error", err);
      // onclose fires after onerror, so reconnect is handled there
    };
  }

  sendFrame(int16Frame) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(int16Frame.buffer);
    }
  }

  disconnect() {
    this._stopped = true;
    clearTimeout(this._connectTimer);
    if (this.ws) {
      this.ws.onclose = null; // suppress reconnect
      this.ws.close();
      this.ws = null;
    }
  }

  _scheduleReconnect() {
    this._connectTimer = setTimeout(() => {
      this._backoff = Math.min(this._backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX_MS);
      this.connect();
    }, this._backoff);
  }
}
