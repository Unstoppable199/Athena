  const messages = document.getElementById("messages");

  /* The welcome panel, kept as written so it can be put back when a
     conversation is cleared. Read once, before anything has had a
     chance to replace it. */
  const EMPTY_STATE_HTML = messages.innerHTML;

  /* The most recent reply from Athena, which is the only one that can
     be redone - redoing an older one would mean discarding everything
     said since. Declared here because addRow() and clearConversation()
     both touch it and neither owns it. */
  let lastAthenaRow = null;

  const input = document.getElementById("input");
  const sendBtn = document.getElementById("send");
  const pendingImageEl = document.getElementById("pending-image");
  const pendingImageContainer = document.getElementById("pending-image-container");
  const removeImageBtn = document.getElementById("remove-image");
  const sessionTokensEl = document.getElementById("session-tokens");
  const soundToggle = document.getElementById("sound-toggle");

  let pendingImageBase64 = null;
  let sessionTokens = 0;
  let soundOn = localStorage.getItem("athena-sound") !== "off";

  syncSoundButton();

  function syncSoundButton() {
    soundToggle.classList.toggle("off", !soundOn);
    soundToggle.title = soundOn
      ? "Completion sound on"
      : "Completion sound off";
  }

  soundToggle.addEventListener("click", () => {
    soundOn = !soundOn;
    localStorage.setItem("athena-sound", soundOn ? "on" : "off");
    syncSoundButton();
    if (soundOn) beep();
  });

  /* ---------------------------------------------------------------
     Mode picker.

     Switching modes swaps the loaded model, which takes time - only
     one model fits in the graphics card at once, so the outgoing one
     is unloaded before the new one loads. The picker is disabled and
     the status pill says what is happening, otherwise the first
     message afterwards just looks slow for no visible reason.
     --------------------------------------------------------------- */

  const modeSelect = document.getElementById("mode-select");
  let currentMode = localStorage.getItem("athena-mode") || "";

  /* True only while a model is being swapped. Checked in send() as
     well as disabling the button, because Enter reaches send()
     directly and would otherwise sail past a disabled button. */
  let switchingMode = false;

  /* True from the moment a message is sent until its reply lands.
     While it is set the send button becomes a stop button and the mode
     picker is locked - switching mode unloads the model that is
     halfway through answering.

     Declared up here, not down with send(), because applyMode() reads
     it and runs first: loadModes() calls it during startup, which is
     before a declaration further down the file has been reached. */
  let answering = false;

  /* Lets the browser give up waiting. The server is told separately,
     through /stop: this only ends the wait on this side, and without
     it the page would sit on a request whose answer is being thrown
     away anyway. */
  let inFlight = null;

  async function loadModes() {
    try {
      const res = await fetch("/modes");
      const data = await res.json();

      modeSelect.innerHTML = "";

      for (const mode of data.modes) {
        const option = document.createElement("option");
        option.value = mode.name;
        option.textContent = mode.label;
        option.title = mode.blurb + "  (" + mode.model + ")";
        modeSelect.appendChild(option);
      }

      /* The server is the authority on what is actually loaded. A
         stored choice is only applied if it differs, so a stale value
         in the browser cannot silently disagree with the model that
         is really running.

         A stored name that is no longer in the list is still sent:
         the server follows renames, so a browser holding "study" from
         before it became Max gets Max, rather than quietly falling
         back to the default with no sign the choice was dropped.
         applyMode() overwrites currentMode from the reply either
         way. */
      if (currentMode && currentMode !== data.current) {
        await applyMode(currentMode);
      } else {
        currentMode = data.current;
      }

      modeSelect.value = currentMode;
      fitModeWidth();

    } catch (err) {
      modeSelect.style.display = "none";
    }
  }

  /* A dropdown is normally as wide as its LONGEST option, so picking
     "Fast" still left a gap the width of "Balanced" before the arrow.
     The selected label is measured in the picker's own font and the
     width set to match, so the arrow sits right after the word. */
  function fitModeWidth() {
    const ruler = document.createElement("span");
    const style = getComputedStyle(modeSelect);

    ruler.textContent =
      modeSelect.options[modeSelect.selectedIndex]?.textContent || "";
    ruler.style.cssText =
      "position:absolute;visibility:hidden;white-space:pre;" +
      "font-family:" + style.fontFamily + ";" +
      "font-size:" + style.fontSize + ";" +
      "font-weight:" + style.fontWeight + ";" +
      "letter-spacing:" + style.letterSpacing + ";";

    document.body.appendChild(ruler);

    /* The text, plus the padding already set in the CSS - that is what
       leaves room for the arrow drawn on the right. */
    const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    modeSelect.style.width = Math.ceil(ruler.offsetWidth + padding) + "px";

    ruler.remove();
  }

  async function applyMode(name) {
    /* Refused outright while a reply is running. Switching unloads the
       model that is halfway through answering, so the running turn
       would fail on its next call. The picker is disabled during a
       reply, but this is reached from loadModes() on startup too. */
    if (answering) return;

    /* Nothing may be sent while a model is loading. The outgoing model
       is unloaded before the new one loads, so for those seconds there
       is nothing to answer with - a message sent now would either fail
       or be answered by whichever model happened to win the race. */
    switchingMode = true;
    modeSelect.disabled = true;
    sendBtn.disabled = true;
    input.disabled = true;
    /* The placeholder in the box says "Loading model..." on its own,
       so there is no second label for it in the header. */
    input.placeholder = "Loading model...";

    try {
      const res = await fetch("/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: name }),
      });

      const data = await res.json();

      if (data.ok) {
        currentMode = data.mode;
        localStorage.setItem("athena-mode", currentMode);
      } else {
        currentMode = data.mode;
        alert(data.error || "That mode couldn't be loaded.");
      }

    } catch (err) {
      alert("Couldn't reach Athena to change mode.");

    } finally {
      switchingMode = false;
      modeSelect.value = currentMode;
      fitModeWidth();
      /* Not unconditionally re-enabled: a reply could have started in
         the meantime, and the picker has to stay locked for it. */
      modeSelect.disabled = answering;
      sendBtn.disabled = false;
      input.disabled = false;
      input.placeholder = "Talk to Athena...";
    }
  }

  modeSelect.addEventListener("change", () => applyMode(modeSelect.value));

  loadModes();


  /* ---------------------------------------------------------------
     Saved conversations.

     Conversations used to live only in memory, so closing Athena threw
     them away. They are now written to disk after every turn, and this
     is the way back to them.
     --------------------------------------------------------------- */

  const historyPanel = document.getElementById("history-panel");
  const historyList = document.getElementById("history-list");
  const historyToggle = document.getElementById("history-toggle");
  const newChatBtn = document.getElementById("new-chat");

  function whenText(seconds) {
    if (!seconds) return "";

    const then = new Date(seconds * 1000);
    const mins = Math.floor((Date.now() - then.getTime()) / 60000);

    if (mins < 1) return "just now";
    if (mins < 60) return mins + (mins === 1 ? " minute ago" : " minutes ago");

    const hours = Math.floor(mins / 60);
    if (hours < 24) return hours + (hours === 1 ? " hour ago" : " hours ago");

    const days = Math.floor(hours / 24);
    if (days < 7) return days + (days === 1 ? " day ago" : " days ago");

    return then.toLocaleDateString();
  }

  async function loadHistory() {
    try {
      const res = await fetch("/conversations");
      const data = await res.json();

      historyList.innerHTML = "";

      if (!data.conversations.length) {
        const empty = document.createElement("div");
        empty.className = "history-empty";
        empty.textContent =
          "No saved conversations yet. They appear here as you talk.";
        historyList.appendChild(empty);
        return;
      }

      for (const convo of data.conversations) {

        /* One waiting on its undo timer is still on the server, so it
           comes back in this list. Leaving it out keeps the list
           agreeing with what the undo bar says has happened. */
        if (pendingDelete && pendingDelete.id === convo.id) continue;

        const item = document.createElement("div");
        item.className = "history-item"
          + (convo.id === data.current ? " current" : "");

        const who = document.createElement("div");
        who.className = "who";

        const title = document.createElement("span");
        title.className = "title";
        /* textContent, not innerHTML: the title is the user's own
           first message and could contain anything. */
        title.textContent = convo.title;

        /* Exchanges, not messages. Every turn stores two - yours and
           the reply - so a three-turn conversation reporting "6
           messages" reads as twice as much as it was. */
        const turns = Math.max(1, Math.round(convo.messages / 2));

        const when = document.createElement("span");
        when.className = "when";
        when.textContent =
          whenText(convo.updated_at)
          + "  ·  " + turns + (turns === 1 ? " exchange" : " exchanges");

        who.appendChild(title);
        who.appendChild(when);
        item.appendChild(who);

        const remove = document.createElement("button");
        remove.className = "remove";
        remove.title = "Delete this conversation";
        remove.textContent = "×";
        remove.addEventListener("click", (e) => {
          /* Or clicking delete would also open the conversation on
             its way past. */
          e.stopPropagation();
          /* The row is passed so undo can simply show it again -
             nothing has been sent, so there is nothing to restore. */
          deleteConversation(convo.id, convo.title, item);
        });

        item.appendChild(remove);
        item.addEventListener("click", () => openConversation(convo.id));
        historyList.appendChild(item);
      }

    } catch (err) {
      historyList.innerHTML =
        '<div class="history-empty">Couldn\'t load saved conversations.</div>';
    }
  }

  /* Wipes the window back to an empty conversation. Used both when
     starting a new one and before drawing a loaded one, so the two
     cannot drift apart. */
  function clearConversation() {
    messages.innerHTML = "";
    minimapBlocks.innerHTML = "";
    minimapPairs.length = 0;
    rail.classList.remove("on");
    lastAthenaRow = null;
    sessionTokens = 0;
    sessionTokensEl.textContent = "0";
    resetFlow();
  }

  /* The suggestion buttons need no rewiring - they are handled by a
     click listener on the document, so a fresh copy works the moment
     it is in the page. */
  function showEmptyState() {
    messages.innerHTML = EMPTY_STATE_HTML;
  }

  async function openConversation(id) {
    if (answering || switchingMode) return;

    try {
      const res = await fetch("/conversations/" + encodeURIComponent(id), {
        method: "POST",
      });
      const data = await res.json();

      if (!data.ok) {
        alert(data.error || "That conversation couldn't be opened.");
        loadHistory();
        return;
      }

      clearConversation();

      for (const msg of data.messages) {
        addRow(msg.content, msg.role === "user" ? "user" : "athena");
      }

      if (!data.messages.length) showEmptyState();

      showHistory(false);
      scrollToBottom();

    } catch (err) {
      alert("Couldn't reach Athena to open that conversation.");
    }
  }

  async function newChat() {
    if (answering || switchingMode) return;

    try {
      const res = await fetch("/conversations/new", { method: "POST" });
      const data = await res.json();

      if (!data.ok) {
        alert(data.error || "Couldn't start a new conversation.");
        return;
      }

      clearConversation();
      showEmptyState();
      showHistory(false);
      input.focus();

    } catch (err) {
      alert("Couldn't reach Athena to start a new conversation.");
    }
  }

  /* Deleting a conversation.

     Nothing is asked and nothing is sent for a few seconds. The row
     disappears immediately, a bar drains at the bottom of the window,
     and the request only goes out when it empties.

     Holding the deletion rather than undoing it afterwards is what
     makes undo honest: there is nothing to restore, because nothing
     was destroyed. It also means the server needs no way to bring a
     conversation back, which would be a new thing to get wrong.

     A confirmation box did the same job by making every delete cost
     two clicks, including the ones you meant. This costs one, and
     forgives the one you didn't. */

  const UNDO_SECONDS = 6;

  const toast = document.getElementById("toast");
  const toastText = document.getElementById("toast-text");
  const toastUndo = document.getElementById("toast-undo");
  const toastTimer = document.getElementById("toast-timer");

  function showToast(text, onUndo) {
    toastText.textContent = text;
    toast.hidden = false;

    /* Restarted from the beginning each time. Without clearing it the
       animation carries on from wherever the last one had got to, so a
       second delete would show a bar that was already half gone. */
    toastTimer.style.animation = "none";
    void toastTimer.offsetWidth;   // forces the style to be applied
    toastTimer.style.animation = `drain ${UNDO_SECONDS}s linear forwards`;

    toastUndo.onclick = onUndo;
  }

  function hideToast() {
    toast.hidden = true;
    toastTimer.style.animation = "none";
    toastUndo.onclick = null;
  }

  /* At most one at a time. Starting a second delete commits the
     first - simpler than a queue, and by then the first has visibly
     been on screen with its own timer. */
  let pendingDelete = null;

  function commitPendingDelete() {
    if (!pendingDelete) return;

    const { id, timer } = pendingDelete;
    clearTimeout(timer);
    pendingDelete = null;
    hideToast();

    fetch("/conversations/" + encodeURIComponent(id), { method: "DELETE" })
      .then(() => fetch("/conversations"))
      .then((r) => r.json())
      .then((data) => {
        /* Deleting the open conversation leaves the server on a fresh
           one, so the window has to follow it. */
        if (!data.current) {
          clearConversation();
          showEmptyState();
        }
        if (!historyPanel.hidden) loadHistory();
      })
      .catch(() => {
        /* The conversation is still there, so put the row back rather
           than leaving the list claiming otherwise. */
        if (!historyPanel.hidden) loadHistory();
      });
  }

  function undoPendingDelete() {
    if (!pendingDelete) return;

    clearTimeout(pendingDelete.timer);

    const row = pendingDelete.row;

    pendingDelete = null;
    hideToast();

    /* Nothing was sent, so this is only a matter of showing the row
       again - unless the list was rebuilt in the meantime, in which
       case that row is no longer part of the page and the list has to
       be redrawn to include the conversation again. */
    if (row && row.isConnected) {
      row.hidden = false;
    } else if (!historyPanel.hidden) {
      loadHistory();
    }
  }

  function deleteConversation(id, title, row) {
    commitPendingDelete();

    if (row) row.hidden = true;

    pendingDelete = {
      id: id,
      row: row,
      timer: setTimeout(commitPendingDelete, UNDO_SECONDS * 1000),
    };

    showToast('Deleted "' + title + '"', undoPendingDelete);
  }

  /* Closing the page commits whatever was waiting. The alternative is
     a conversation that looks deleted, is not, and comes back on the
     next load.

     keepalive lets the request outlive the page; sendBeacon cannot be
     used here because it only ever sends a POST. */
  window.addEventListener("beforeunload", () => {
    if (!pendingDelete) return;

    fetch("/conversations/" + encodeURIComponent(pendingDelete.id), {
      method: "DELETE",
      keepalive: true,
    }).catch(() => {});
  });

  /* One place that opens and closes the panel, so the button's lit
     state cannot drift out of step with whether it is showing. */
  function showHistory(open) {
    historyPanel.hidden = !open;
    historyToggle.classList.toggle("open", open);
    if (open) loadHistory();
  }

  historyToggle.addEventListener("click", () => showHistory(historyPanel.hidden));

  document.getElementById("history-close")
    .addEventListener("click", () => showHistory(false));

  newChatBtn.addEventListener("click", newChat);

  /* Clicking away closes the panel, which is what people expect of
     anything that drops down over the page. */
  document.addEventListener("click", (e) => {
    if (historyPanel.hidden) return;
    if (historyPanel.contains(e.target) || historyToggle.contains(e.target)) return;
    showHistory(false);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !historyPanel.hidden) showHistory(false);
  });

  /* Called at the very end of this script, not here: it reaches the
     minimap and the flow diagram, which are set up further down. */
  async function restoreLastConversation() {
    try {
      const res = await fetch("/conversations");
      const data = await res.json();

      if (data.current) {
        await openConversation(data.current);
      }

    } catch (err) {
      /* A missing history is not worth interrupting startup for. */
    }
  }

  /* ---------------------------------------------------------------
     Completion sound. Synthesised with the Web Audio API rather than
     an audio file, so it needs no asset and makes no network request
     - the app is meant to run fully offline.
     --------------------------------------------------------------- */

  let audioCtx = null;

  function beep() {
    if (!soundOn) return;

    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") audioCtx.resume();

      const now = audioCtx.currentTime;

      // Two short notes a fourth apart - a soft "ding-dong" that
      // reads as finished rather than as an alert.
      [[784, 0], [1046, 0.11]].forEach(([freq, offset]) => {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();

        osc.type = "sine";
        osc.frequency.value = freq;

        gain.gain.setValueAtTime(0.0001, now + offset);
        gain.gain.exponentialRampToValueAtTime(0.12, now + offset + 0.015);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.28);

        osc.connect(gain);
        gain.connect(audioCtx.destination);

        osc.start(now + offset);
        osc.stop(now + offset + 0.3);
      });
    } catch (e) {
      /* audio is a nicety - never let it break the reply */
    }
  }

  function scrollToBottom() {
    requestAnimationFrame(() => {
      const last = messages.lastElementChild;
      if (last) last.scrollIntoView({ behavior: "smooth", block: "end" });
    });
  }

  function fmtDuration(seconds) {
    if (seconds < 60) return seconds.toFixed(1) + "s";
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return m + "m " + String(s).padStart(2, "0") + "s";
  }

  function addRow(text, who, opts) {
    opts = opts || {};

    const empty = document.getElementById("empty-state");
    if (empty) empty.remove();

    const row = document.createElement("div");
    row.className = "row " + who;

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = who === "user" ? "you" : "athena";
    row.appendChild(label);

    /* "stopped" is styled apart from "error": nothing went wrong, the
       user asked for it. Shown dimmed rather than red. */
    const bubble = document.createElement("div");
    bubble.className = "bubble"
      + (opts.error ? " error" : "")
      + (opts.stopped ? " stopped" : "");

    if (opts.imageDataUrl) {
      const img = document.createElement("img");
      img.src = opts.imageDataUrl;
      img.className = "thumb";
      img.onload = scrollToBottom;
      bubble.appendChild(img);
    }

    const textNode = document.createElement("div");
    textNode.textContent = text;
    bubble.appendChild(textNode);
    row.appendChild(bubble);

    if (who === "athena" && Array.isArray(opts.sources) && opts.sources.length) {
      const sources = document.createElement("details");
      sources.className = "sources";

      const summary = document.createElement("summary");
      summary.textContent = opts.sources.length === 1
        ? "1 verified source"
        : opts.sources.length + " verified sources";
      sources.appendChild(summary);

      const links = document.createElement("div");
      links.className = "source-links";

      opts.sources.forEach((source) => {
        const item = document.createElement(source.url ? "a" : "span");
        item.textContent = source.label || source.url || source.path || "Source";
        if (source.url) {
          item.href = source.url;
          item.target = "_blank";
          item.rel = "noopener noreferrer";
        } else if (source.path) {
          item.title = source.path;
        }
        links.appendChild(item);
      });

      sources.appendChild(links);
      row.appendChild(sources);
    }

    /* Athena's replies always get a meta line, even with no stats to
       put in it, because the redo button lives there. */
    if ((opts.meta && opts.meta.length) || who === "athena") {
      const meta = document.createElement("div");
      meta.className = "meta";

      (opts.meta || []).forEach((item) => {
        const s = document.createElement("span");
        s.textContent = item;
        meta.appendChild(s);
      });

      if (who === "athena") {
        const again = document.createElement("button");
        again.className = "redo";
        again.type = "button";
        again.title = "Answer again, forgetting this attempt";
        again.textContent = "redo";
        again.addEventListener("click", redo);
        meta.appendChild(again);
      }

      row.appendChild(meta);
    }

    messages.appendChild(row);
    addMinimapBlock(row, who, text, opts.error);

    /* Only the newest reply carries the button. Redoing an older one
       would mean discarding everything said since, which is a
       different and much bigger promise than "try that again". */
    if (who === "athena") {
      if (lastAthenaRow) {
        const old = lastAthenaRow.querySelector(".redo");
        if (old) old.remove();
      }
      lastAthenaRow = row;
    }

    scrollToBottom();
  }

  /* Takes a row off screen and out of the minimap together, so the
     two cannot end up disagreeing about what the conversation is. */
  function removeRow(row) {
    const index = minimapPairs.findIndex((p) => p.row === row);

    if (index !== -1) {
      minimapPairs[index].block.remove();
      minimapPairs.splice(index, 1);
    }

    row.remove();
    updateMinimapViewport();
  }

  /* ---------------------------------------------------------------
     Minimap. One block per message, sized from how long the message
     is, so the rail reads as the shape of the conversation. There is
     deliberately no text in it - the blocks carry position and length,
     and clicking one jumps to that message.
     --------------------------------------------------------------- */

  /* <main> is the element that actually scrolls, so it is the one the
     viewport marker has to read its position from. */
  const main = document.querySelector("main");

  const rail = document.getElementById("rail");
  const minimap = document.getElementById("minimap");
  const minimapBlocks = document.getElementById("minimap-blocks");
  const minimapViewport = document.getElementById("minimap-viewport");

  /* Turns a message length into a block height. Short replies get a
     small block, long ones a tall block, but the growth tapers so a
     single enormous answer cannot swallow the whole rail. */
  function blockHeight(text) {
    const length = (text || "").length;
    /* SIZE: the 14 is the smallest block, 2.8 how fast they grow with
       message length, and 84 the tallest any block may get. */
    return Math.round(Math.min(84, 14 + Math.sqrt(length) * 2.8));
  }

  /* Every block paired with the message it stands for. The viewport
     marker is worked out from these rather than from scroll position
     alone: block heights are deliberately not to scale (a huge answer
     is capped), so a straight scrollTop-to-rail ratio drifts out of
     step with what is actually on screen. */
  const minimapPairs = [];

  function addMinimapBlock(row, who, text, isError) {
    const block = document.createElement("button");
    block.type = "button";
    block.className = "mm-block " + who + (isError ? " error" : "");
    block.style.height = blockHeight(text) + "px";

    /* No label and no title: the user asked for shapes only. The
       button still needs a name for screen readers, which is what
       aria-label is for - it is never drawn on screen. */
    block.setAttribute("aria-label",
      who === "user" ? "Jump to your message" : "Jump to Athena's reply");

    block.addEventListener("click", () => {
      row.scrollIntoView({ behavior: "smooth", block: "center" });
    });

    minimapBlocks.appendChild(block);
    minimapPairs.push({ row: row, block: block });

    /* Shown as soon as there is something to map.

       This used to be switched on once when the page loaded, which
       stopped working the moment conversations could be reloaded:
       clearConversation() turns the rail off, and restoring the last
       conversation on startup calls it - so the minimap disappeared on
       every load and never came back. Tying it to the blocks means it
       cannot get out of step with whether any exist. */
    rail.classList.add("on");

    updateMinimapViewport();
  }

  /* The pale window marking which messages are on screen right now.
     It is drawn around the blocks whose messages are actually visible,
     so it always lines up with real blocks instead of landing between
     them. */
  function updateMinimapViewport() {
    if (!minimapPairs.length) return;

    const view = main.getBoundingClientRect();
    const railTop = minimapBlocks.getBoundingClientRect().top;

    let first = null;
    let last = null;

    for (const pair of minimapPairs) {
      const box = pair.row.getBoundingClientRect();

      /* Any overlap at all counts as visible - a message half off the
         bottom of the screen is still partly being read. */
      if (box.bottom > view.top && box.top < view.bottom) {
        if (first === null) first = pair.block;
        last = pair.block;
      }
    }

    /* Between two messages with neither overlapping - rare, but the
       marker should hold still rather than vanish. */
    if (!first) return;

    const top = first.getBoundingClientRect().top - railTop;
    const bottom = last.getBoundingClientRect().bottom - railTop;

    /* Padded slightly so the outline sits around the blocks rather
       than exactly on their edges. */
    minimapViewport.style.top = Math.round(top - 3) + "px";
    minimapViewport.style.height = Math.round(bottom - top + 6) + "px";
  }

  main.addEventListener("scroll", updateMinimapViewport, { passive: true });

  /* The rail is positioned against the window, so it needs the real
     header and footer heights to know where to stop. Measured rather
     than guessed, because both change with the font sizes above. */
  function measureChrome() {
    const header = document.querySelector("header");
    const footer = document.querySelector("footer");
    document.documentElement.style.setProperty(
      "--header-h", (header ? header.offsetHeight : 90) + "px");
    document.documentElement.style.setProperty(
      "--footer-h", (footer ? footer.offsetHeight : 150) + "px");
    updateMinimapViewport();
  }

  /* ---------------------------------------------------------------
     The pipeline diagram.

     Steps light up from the "key" the server sends with each stage,
     never from the wording, so adding a capability with new wording
     cannot quietly stop the diagram working.

     Steps stay lit once passed, so at the end of a turn the diagram
     shows the route the request actually took - and a chat reply
     visibly leaves plan, run and verify dark.
     --------------------------------------------------------------- */

  const flowRail = document.getElementById("flow-rail");

  const node = (id) => document.getElementById(id);
  const pipe = (id) => document.getElementById(id);

  /* Which node each stage key lights, and which pipe leads into it.
     Keys come from the server; nothing here matches on wording. */
  const FLOW_NODES = {
    route:   "n-route",
    plan:    "n-plan",
    run:     "n-run",
    compose: "n-compose",
    verify:  "n-verify",
  };

  const ALL_NODES = ["n-message", "n-route", "n-chat", "n-plan", "n-run",
                     "n-compose", "n-verify", "n-answer"];

  const ALL_PIPES = ["p-msg-route", "p-route-chat", "p-route-plan",
                     "p-plan-run", "p-run-plan", "p-chat-compose",
                     "p-compose-answer", "p-run-compose",
                     "p-compose-verify", "p-verify-answer"];

  /* Whether this turn has gone down the lookup branch. It decides
     which way "compose" was reached: through plan and run, or straight
     across from chat. */
  let tookLookupBranch = false;

  /* ---- capabilities, listed inside the run node ----

     The run node grows a line for each capability the turn uses, and
     everything below it shifts down by the same amount so the diagram
     stays joined up. Geometry is kept here rather than in the markup
     because the node's height is not known until the request runs. */

  const SVG_NS = "http://www.w3.org/2000/svg";
  const runRect = document.getElementById("run-rect");
  const runTools = document.getElementById("run-tools");
  const lower = document.getElementById("lower");
  const lowerPipes = document.getElementById("lower-pipes");
  const flowSvg = document.getElementById("flow-svg");

  const RUN_TOP = 226;        /* where the execute box starts */
  const RUN_BASE_H = 32;      /* its height with no capabilities */
  const TOOL_LINE = 17;       /* SIZE: height of one capability line */
  const BASE_VIEW_H = 520;    /* diagram height with no capabilities */
  const COMPOSE_TOP = 316;    /* where compose sits before shifting */
  const IDLE_VIEW_H = 205;    /* compact trace height between requests */

  /* Technical capability IDs are useful in logs but visually noisy. The
     pipeline uses plain names and keeps the exact ID in an SVG tooltip. */
  const TOOL_LABELS = {
    "filesystem.search": "file search",
    "filesystem.read": "document read",
    "filesystem.list": "list folder",
    "filesystem.exists": "check path",
    "filesystem.info": "file details",
    "filesystem.semantic_search": "meaning search",
    "web.search": "web research",
    "weather.current": "weather",
    "finance.quote": "market price",
    "finance.exchange": "exchange rate",
    "system.datetime": "date & time",
    "system.timezone": "time zone",
    "code.generate": "write code",
    "code.run": "run code",
    "python.run": "run program",
    "image.analyze": "image analysis",
  };

  let currentToolExtra = 0;
  let flowCollapseTimer = null;
  let flowPinned = false;
  let hasCompletedFlow = false;
  let lastFlowOutcome = "complete";

  function updateFlowViewBox(detailed) {
    flowSvg.setAttribute(
      "viewBox",
      "0 0 280 " + (detailed ? BASE_VIEW_H + currentToolExtra : IDLE_VIEW_H)
    );
  }

  function setFlowMode(mode) {
    flowRail.classList.remove(
      "idle", "working", "complete", "failed", "stopped", "pinned"
    );
    flowRail.classList.add(mode);

    if (flowPinned && mode !== "working") {
      flowRail.classList.add("pinned");
    }

    const detailed = mode !== "idle" || flowPinned;
    updateFlowViewBox(detailed);

    if (mode === "working") {
      flowRail.setAttribute("aria-label", "Athena is processing this request.");
    } else if (mode === "failed") {
      flowRail.setAttribute(
        "aria-label", "The request failed. Click to collapse its pipeline."
      );
    } else if (mode === "stopped") {
      flowRail.setAttribute(
        "aria-label", "The request was stopped. Click to collapse its pipeline."
      );
    } else if (detailed) {
      flowRail.setAttribute(
        "aria-label", "Processing pipeline expanded. Click to collapse it."
      );
    } else {
      flowRail.setAttribute(
        "aria-label", "Processing pipeline. Click to expand the last route."
      );
    }
  }

  function scheduleFlowCollapse() {
    clearTimeout(flowCollapseTimer);
    flowCollapseTimer = setTimeout(() => {
      if (!flowPinned) setFlowMode("idle");
    }, 8000);
  }

  function layoutTools(tools, running) {
    tools = tools || [];

    /* Nothing to redraw if the picture would be identical. */
    const signature = tools.join("|") + (running ? "|R" : "");
    if (runTools.dataset.shown === signature) return;
    runTools.dataset.shown = signature;

    while (runTools.firstChild) runTools.removeChild(runTools.firstChild);

    const extra = tools.length * TOOL_LINE;
    currentToolExtra = extra;

    runRect.setAttribute("height", RUN_BASE_H + extra);

    tools.forEach((tool, i) => {
      const row = document.createElementNS(SVG_NS, "g");
      row.setAttribute("class", "run-tool-row");

      const title = document.createElementNS(SVG_NS, "title");
      title.textContent = tool;

      const text = document.createElementNS(SVG_NS, "text");
      text.setAttribute("x", 210);
      /* Below "use tools", one friendly line per capability. */
      text.setAttribute("y", RUN_TOP + RUN_BASE_H + i * TOOL_LINE + 12);
      text.setAttribute("class",
        "run-tool" + (running && i === tools.length - 1 ? " live" : ""));
      text.textContent = TOOL_LABELS[tool] || tool.replaceAll(".", " ");

      row.appendChild(title);
      row.appendChild(text);
      runTools.appendChild(row);
    });

    /* Everything downstream moves by the amount the node grew. */
    const shift = "translate(0," + extra + ")";
    lower.setAttribute("transform", shift);
    lowerPipes.setAttribute("transform", shift);

    /* The two pipes feeding compose start above the shift and end
       below it, so they are redrawn rather than moved. */
    const composeTop = COMPOSE_TOP + extra;

    pipe("p-run-compose").setAttribute("d",
      "M210," + (RUN_TOP + RUN_BASE_H + extra) +
      " C210," + (composeTop - 21) +
      " 140," + (composeTop - 28) +
      " 140," + composeTop);

    pipe("p-chat-compose").setAttribute("d",
      "M60,184 C60," + (composeTop - 26) +
      " 140," + (composeTop - 31) +
      " 140," + composeTop);

    updateFlowViewBox(!flowRail.classList.contains("idle") || flowPinned);
  }

  function resetFlow() {
    clearTimeout(flowCollapseTimer);
    flowPinned = false;
    ALL_NODES.forEach((id) =>
      node(id).classList.remove("active", "done", "halted"));
    ALL_PIPES.forEach((id) =>
      pipe(id).classList.remove("flowing", "done", "halted"));
    tookLookupBranch = false;
    layoutTools([], false);
    setFlowMode("working");
  }

  /* Whatever was flowing has now been travelled. */
  function settlePipes(outcomeClass = "done") {
    ALL_PIPES.forEach((id) => {
      const el = pipe(id);
      if (el.classList.contains("flowing")) {
        el.classList.remove("flowing");
        el.classList.add(outcomeClass);
      }
    });
  }

  function activate(nodeId, pipeIds) {
    /* Whatever was running has finished, so it becomes done rather
       than going dark - the pipeline is a loop, not a line, and "run"
       must not un-light when the planner comes back round. */
    ALL_NODES.forEach((id) => {
      const el = node(id);
      if (el.classList.contains("active")) {
        el.classList.remove("active");
        el.classList.add("done");
      }
    });

    settlePipes();

    (pipeIds || []).forEach((id) => {
      pipe(id).classList.remove("done");
      pipe(id).classList.add("flowing");
    });

    node(nodeId).classList.add("active");
  }

  function markFlow(key) {
    const nodeId = FLOW_NODES[key];
    if (!nodeId) return;

    if (key === "route") {
      node("n-message").classList.add("done");
      activate("n-route", ["p-msg-route"]);
      return;
    }

    if (key === "plan") {
      tookLookupBranch = true;
      node("n-message").classList.add("done");
      node("n-route").classList.add("done");
      /* Coming back to plan after a tool is the loop, not the branch. */
      const via = node("n-run").classList.contains("done")
        ? ["p-run-plan"]
        : ["p-route-plan"];
      activate("n-plan", via);
      return;
    }

    if (key === "run") {
      tookLookupBranch = true;
      activate("n-run", ["p-plan-run"]);
      return;
    }

    if (key === "compose") {
      node("n-message").classList.add("done");
      node("n-route").classList.add("done");

      if (tookLookupBranch) {
        activate("n-compose", ["p-run-compose"]);
      } else {
        /* The short path: straight from route through chat, with plan
           and run never touched. */
        node("n-chat").classList.add("done");
        pipe("p-route-chat").classList.add("done");
        activate("n-compose", ["p-chat-compose"]);
      }
      return;
    }

    if (key === "verify") {
      activate("n-verify", ["p-compose-verify"]);
    }
  }

  /* At the end of a turn the last stage becomes done, the answer
     endpoint lights, and nothing is left flowing. */
  function settleFlow(outcome = "complete") {
    ALL_NODES.forEach((id) => {
      const el = node(id);
      if (el.classList.contains("active")) {
        el.classList.remove("active");
        el.classList.add(outcome === "complete" ? "done" : "halted");
      }
    });

    settlePipes(outcome === "complete" ? "done" : "halted");

    if (outcome === "complete" && node("n-compose").classList.contains("done")) {
      if (node("n-verify").classList.contains("done")) {
        pipe("p-verify-answer").classList.add("done");
      } else {
        /* Ordinary chat never runs evidence verification. */
        pipe("p-compose-answer").classList.add("done");
      }
      node("n-answer").classList.add("done");
    }

    hasCompletedFlow = true;
    lastFlowOutcome = outcome;
    setFlowMode(outcome);
    scheduleFlowCollapse();
  }

  /* The pipeline is always available, but stays compact between turns.
     Click or press Enter/Space to inspect or collapse the last route. */
  flowRail.classList.add("on");
  setFlowMode("idle");

  function toggleFlowDetails() {
    if (flowRail.classList.contains("working") || !hasCompletedFlow) return;

    clearTimeout(flowCollapseTimer);

    if (flowRail.classList.contains("idle")) {
      flowPinned = true;
      setFlowMode(lastFlowOutcome);
    } else {
      flowPinned = false;
      setFlowMode("idle");
    }
  }

  flowRail.addEventListener("click", toggleFlowDetails);
  flowRail.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    toggleFlowDetails();
  });

  measureChrome();
  window.addEventListener("resize", measureChrome);

  /* The footer grows as the textarea does, which moves the bottom of
     the rail, so remeasure whenever it changes size. */
  if (window.ResizeObserver) {
    const footerEl = document.querySelector("footer");
    if (footerEl) new ResizeObserver(measureChrome).observe(footerEl);
  }

  /* ---------------------------------------------------------------
     Thinking indicator. The stage label comes from the server's
     /status endpoint, so it reports what the agent is genuinely
     doing (searching, reading a file, composing) rather than
     cycling through plausible-sounding words on a timer.
     --------------------------------------------------------------- */

  let thinkingRow = null;
  let tickTimer = null;
  let pollTimer = null;
  let startedAt = 0;

  function startThinking() {
    startedAt = performance.now();

    const empty = document.getElementById("empty-state");
    if (empty) empty.remove();

    thinkingRow = document.createElement("div");
    thinkingRow.className = "row athena";
    thinkingRow.innerHTML =
      '<div class="label">athena</div>' +
      '<div class="thinking">' +
        '<div class="dots"><i></i><i></i><i></i></div>' +
        '<div class="stage">Thinking</div>' +
        '<div class="elapsed">0.0s</div>' +
      '</div>';

    messages.appendChild(thinkingRow);
    scrollToBottom();

    /* No status pill for an ordinary reply - the thinking
       indicator in the conversation already reports the stage. */

    /* A fresh turn starts a fresh run through the pipeline. */
    resetFlow();

    tickTimer = setInterval(() => {
      const secs = (performance.now() - startedAt) / 1000;
      const el = thinkingRow && thinkingRow.querySelector(".elapsed");
      if (el) el.textContent = fmtDuration(secs);
    }, 100);

    pollStage();
    pollTimer = setInterval(pollStage, 500);
  }

  async function pollStage() {
    try {
      const res = await fetch("/status");
      if (!res.ok) return;

      const data = await res.json();
      const el = thinkingRow && thinkingRow.querySelector(".stage");

      if (el && data.stage && data.stage !== "idle") {
        el.textContent = data.stage;
      }

      /* Light the diagram from the key, not the sentence. */
      if (data.key && data.key !== "idle") markFlow(data.key);

      layoutTools(data.tools, true);
    } catch (e) {
      /* the reply itself is what matters - ignore polling blips */
    }
  }

  function stopThinking(outcome = "complete") {
    clearInterval(tickTimer);
    clearInterval(pollTimer);
    tickTimer = pollTimer = null;

    if (thinkingRow) thinkingRow.remove();
    thinkingRow = null;

    /* Leaves the route the request took on screen, with nothing still
       pulsing as though work were continuing. */
    settleFlow(outcome);

    /* One last read, so a capability that ran between polls still
       appears, and nothing is left marked as running. */
    fetch("/status")
      .then((r) => r.json())
      .then((d) => layoutTools(d.tools, false))
      .catch(() => {});


  }

  document.addEventListener("paste", (e) => {
    const items = e.clipboardData.items;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        const reader = new FileReader();
        reader.onload = () => {
          pendingImageBase64 = reader.result.split(",")[1];
          pendingImageEl.src = reader.result;
          pendingImageContainer.style.display = "block";
        };
        reader.readAsDataURL(file);
      }
    }
  });

  function setAnswering(on) {
    answering = on;
    sendBtn.classList.toggle("busy", on);
    sendBtn.title = on ? "Stop" : "Send";
    sendBtn.setAttribute("aria-label", on ? "Stop" : "Send");
    modeSelect.disabled = on || switchingMode;
  }

  async function stop() {
    if (!answering) return;

    /* Told to stop first, then abandoned. The other order works too,
       but leaves a moment where the page has stopped listening while
       the model is still going, and a mode switch in that gap would
       pull the model out from under a turn that is still running. */
    try {
      const response = await fetch("/stop", { method: "POST" });
      const result = await response.json();

      /* If the server already committed the answer, aborting here
         would hide a real reply while leaving it in conversation
         memory. Let that completed response arrive instead. */
      if (!result.ok) return;
    } catch (err) {
      /* Nothing useful to do: the local abort below still frees the
         page, which is the part the user is waiting to see. */
    }

    if (inFlight) inFlight.abort();
  }

  /* Puts a reply on screen with its stats. Shared by send() and
     redo(), which differ only in what they ask the server for. */
  function renderReply(data, wallSeconds) {
    const meta = [fmtDuration(data.seconds || wallSeconds)];

    if (data.tokens) {
      /* Three numbers that add up to the total, each meaning exactly
         one thing: prompt tokens the model read, prompt tokens Ollama
         served from its cache without reading, and tokens written.

         Two earlier attempts got this wrong in opposite directions.
         "9,137 tokens" counted cached tokens as if they were work,
         since Ollama re-reports a cached prefix in full. Replacing it
         with "19 tokens" next to "1,975 in / 19 out" read as a
         contradiction, and the 19 was the written tokens counted
         twice - once as "computed", once as "out". */
      const read = data.read_tokens || 0;
      const reused = data.cached_tokens || 0;
      const written = data.output_tokens || 0;

      meta.push(
        read.toLocaleString() + " read"
        + (reused > 0 ? " · " + reused.toLocaleString() + " reused" : "")
        + " · " + written.toLocaleString() + " written"
      );

      /* The header counts real work only - what was read plus what was
         written - or the session total drifts further from the truth
         with every message. Cached tokens are deliberately left out. */
      sessionTokens += read + written;
      sessionTokensEl.textContent = sessionTokens.toLocaleString();
    }

    if (data.model_calls) {
      meta.push(data.model_calls +
        (data.model_calls === 1 ? " model call" : " model calls"));
    }

    addRow(data.response, "athena", {
      meta,
      stopped: data.stopped,
      error: data.error,
      sources: data.sources || []
    });

    /* No chime for stopped or failed work - the sound means "your answer is
       ready", and neither outcome produced one. */
    if (!data.stopped && !data.error) beep();
  }

  /* Ask the last question again and forget the first answer.

     The visible half is only half the point. The server drops the
     earlier attempt from the conversation entirely, so it is not
     quoted back to the model on later turns as something already
     settled - which is what makes a wrong answer keep influencing
     everything after it. */
  async function redo() {
    if (answering || switchingMode) return;

    /* The reply being replaced goes now rather than when the new one
       arrives, so it is clear which answer is being reconsidered. */
    if (lastAthenaRow) {
      removeRow(lastAthenaRow);
      lastAthenaRow = null;
    }

    setAnswering(true);
    startThinking();
    inFlight = new AbortController();

    try {
      const res = await fetch("/retry", {
        method: "POST",
        signal: inFlight.signal
      });

      const wallSeconds = (performance.now() - startedAt) / 1000;
      const data = await res.json();
      stopThinking(!res.ok || data.error
        ? "failed"
        : (data.stopped ? "stopped" : "complete"));

      if (!res.ok) {
        addRow("Athena couldn't redo that (server error " + res.status + ").",
               "athena", { error: true });
        return;
      }

      renderReply(data, wallSeconds);

    } catch (err) {
      stopThinking(err && err.name === "AbortError" ? "stopped" : "failed");

      if (err && err.name === "AbortError") {
        addRow("Stopped.", "athena", { stopped: true });
      } else {
        addRow("Error: " + err, "athena", { error: true });
      }

    } finally {
      inFlight = null;
      setAnswering(false);
      input.focus();
    }
  }

  async function send() {
    /* Blocked during a model swap. Enter calls this directly, so the
       disabled send button alone would not stop it. */
    if (switchingMode) return;

    const text = input.value.trim();
    if (!text && !pendingImageBase64) return;
    if (sendBtn.disabled) return;

    const imageDataUrl = pendingImageContainer.style.display === "block"
      ? pendingImageEl.src
      : null;

    addRow(text || "(image)", "user", { imageDataUrl });

    input.value = "";
    input.style.height = "auto";
    setAnswering(true);

    const payload = { message: text || "Describe this image." };
    if (pendingImageBase64) payload.image = pendingImageBase64;

    pendingImageBase64 = null;
    pendingImageEl.src = "";
    pendingImageContainer.style.display = "none";

    startThinking();

    inFlight = new AbortController();

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: inFlight.signal
      });

      const wallSeconds = (performance.now() - startedAt) / 1000;
      const data = await res.json();
      stopThinking(!res.ok || data.error
        ? "failed"
        : (data.stopped ? "stopped" : "complete"));

      if (!res.ok) {
        addRow(
          "Athena couldn't complete that request (server error " + res.status + ").",
          "athena",
          { error: true, meta: [fmtDuration(wallSeconds)] }
        );
        return;
      }

      renderReply(data, wallSeconds);

    } catch (err) {
      stopThinking(err && err.name === "AbortError" ? "stopped" : "failed");

      /* An abort is the stop button working, not a failure. The
         server was already told and answers the request itself, but
         that reply is discarded once the page has stopped listening -
         so the transcript entry is written here instead. */
      if (err && err.name === "AbortError") {
        addRow("Stopped.", "athena", {
          stopped: true,
          meta: [fmtDuration((performance.now() - startedAt) / 1000)]
        });
      } else {
        addRow("Error: " + err, "athena", { error: true });
      }

    } finally {
      inFlight = null;
      setAnswering(false);
      input.focus();
    }
  }

  /* One handler for both jobs, matching the one button. */
  sendBtn.addEventListener("click", () => (answering ? stop() : send()));

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      /* Enter does not stop a running reply, only sends. Stopping is
         destructive and the key is pressed by reflex - it would be
         far too easy to throw away an answer that was nearly there.
         Escape does that instead, where a mistake costs nothing. */
      if (!answering) send();
    }

    if (e.key === "Escape" && answering) {
      e.preventDefault();
      stop();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = input.scrollHeight + "px";
  });

  removeImageBtn.addEventListener("click", () => {
    pendingImageBase64 = null;
    pendingImageEl.src = "";
    pendingImageContainer.style.display = "none";
  });

  document.addEventListener("click", (e) => {
    if (e.target.classList.contains("suggestion")) {
      input.value = e.target.textContent;
      input.focus();
      send();
    }
  });

  input.focus();

  /* Last, so everything it touches - the minimap, the flow diagram -
     already exists.

     Reopening whatever was last being talked about matters more here
     than in most programs: Athena restarts on every code change, and
     coming back to a blank window each time would make the history
     feel like an archive rather than somewhere work continues. */
  restoreLastConversation();
