// TranscriptUI
// Renders interim and final transcript segments with auto-scroll.

export class TranscriptUI {
  constructor() {
    this.container  = document.getElementById("transcript");
    this._interimEl = null;
  }

  // Handle a message object from the server
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
  }
}
