// C8 — TranscriptUI
// Renders interim and final transcript segments, mic amplitude indicator, auto-scroll.
//
// Mic colour is purely amplitude-driven — no server state involved:
//   0.00–0.00  →  grey   (silent)
//   0.00–0.33  →  grey → yellow
//   0.33–0.66  →  yellow → amber
//   0.66–1.00  →  amber → green

export class TranscriptUI {
  constructor() {
    this.container  = document.getElementById("transcript");
    this.ampRect    = document.getElementById("mic-amp-rect");
    this.micRing    = document.getElementById("mic-ring");
    this._interimEl = null;
    this._MIC_HEIGHT = 64; // must match SVG viewBox height
  }

  // Handle a message object from the server (transcript only — no colour logic)
  handle(msg) {
    switch (msg.type) {
      case "session":
        this._log(`Session: ${msg.session_id}`);
        break;
      case "interim":
        this._renderInterim(msg.text);
        break;
      case "final":
        this._commitFinal(msg.text);
        break;
      case "warning":
        console.warn("[server warning]", msg.message);
        break;
      case "error":
        console.error("[server error]", msg.message);
        break;
    }
  }

  // Drive mic fill and ring colour from audio amplitude (0.0–1.0)
  setAmplitude(level) {
    if (!this.ampRect || !this.micRing) return;

    // Fill rect grows from bottom
    const fillH = this._MIC_HEIGHT * level;
    const fillY = this._MIC_HEIGHT - fillH;
    this.ampRect.setAttribute("y", fillY);
    this.ampRect.setAttribute("height", fillH);

    // Colour interpolation: grey → yellow → amber → green
    const colour = this._amplitudeColour(level);
    this.ampRect.setAttribute("fill", colour);
    this.micRing.setAttribute("stroke", colour);
  }

  // Interpolate colour across three stops based on amplitude level
  _amplitudeColour(level) {
    if (level < 0.01) return "#3a3a3a"; // silent — grey

    // Colour stops: [threshold, r, g, b]
    const stops = [
      [0.00,  58,  58,  58],  // grey
      [0.15, 234, 179,   8],  // yellow  #eab308
      [0.35, 217, 119,   6],  // amber   #d97706
      [0.60,  34, 197,  94],  // green   #22c55e
    ];

    // Find the two stops to interpolate between
    for (let i = 1; i < stops.length; i++) {
      const [t0, r0, g0, b0] = stops[i - 1];
      const [t1, r1, g1, b1] = stops[i];
      if (level <= t1) {
        const t = (level - t0) / (t1 - t0);
        const r = Math.round(r0 + t * (r1 - r0));
        const g = Math.round(g0 + t * (g1 - g0));
        const b = Math.round(b0 + t * (b1 - b0));
        return `rgb(${r},${g},${b})`;
      }
    }
    return "#22c55e"; // clamp to green
  }

  // Interim: update or create a muted span for in-progress speech
  _renderInterim(text) {
    if (!this._interimEl) {
      this._interimEl = document.createElement("span");
      this._interimEl.className = "interim";
      this.container.appendChild(this._interimEl);
    }
    this._interimEl.textContent = text;
    this._scrollToBottom();
  }

  // Final: replace interim span with a committed paragraph
  _commitFinal(text) {
    if (this._interimEl) {
      this._interimEl.remove();
      this._interimEl = null;
    }
    const p = document.createElement("p");
    p.className = "final";
    p.textContent = text;
    this.container.appendChild(p);
    this._scrollToBottom();
  }

  _scrollToBottom() {
    this.container.scrollTop = this.container.scrollHeight;
  }

  _log(text) {
    const el = document.createElement("p");
    el.className = "log";
    el.textContent = text;
    this.container.appendChild(el);
  }

  clear() {
    this.container.innerHTML = "";
    this._interimEl = null;
    this.setAmplitude(0);
  }
}
