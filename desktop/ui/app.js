/* Folder RAG desktop — UI logic */

const $ = (id) => document.getElementById(id);

const state = {
  folder: "",
  settings: null,
  asking: false,
  sessions: [],          // [{id, title, folder, createdAt, updatedAt, qa: [{q, a, sources}]}]
  activeSessionId: null,
  indexing: false,
};

const SESSIONS_KEY = "folder-rag.sessions.v1";

// ---------------------- helpers ----------------------
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function toast(msg, kind = "info", spinner = false) {
  removeToast();
  const t = document.createElement("div");
  t.className = "toast " + (kind === "ok" ? "ok" : kind === "err" ? "err" : "");
  t.id = "live-toast";
  if (spinner) {
    const s = document.createElement("span"); s.className = "spin"; t.appendChild(s);
  }
  const sp = document.createElement("span"); sp.textContent = msg; t.appendChild(sp);
  document.querySelector(".main").appendChild(t);
  return t;
}
function removeToast() { $("live-toast")?.remove(); }

function setStatus(text) { $("status-line").textContent = text; }

// ---------------------- folder + index ----------------------
async function pickFolder() {
  if (!window.pywebview?.api?.pick_folder) return;
  const folder = await window.pywebview.api.pick_folder();
  if (!folder) return;
  state.folder = folder;
  $("folder-display").textContent = folder;
  $("folder-display").classList.add("highlighted");
  $("rebuild-index").disabled = false;
  $("question").disabled = false;
  $("ask").disabled = false;
  setStatus("ready · ask a question or rebuild index");
  // remember it
  await api("/api/settings", { method: "POST", body: { last_folder: folder } });
}

async function rebuildIndex() {
  if (!state.folder || state.indexing) return;
  state.indexing = true;
  $("rebuild-index").disabled = true;
  showProgress(true);
  setProgress(0, "walking folder…");
  setStatus("indexing…");

  // simulated client-side progression so the bar isn't a single jump.
  // real backend doesn't stream — we ramp to 90% then snap to 100% on completion.
  const phases = [
    [12, "walking folder…"],
    [28, "reading files…"],
    [48, "chunking…"],
    [68, "embedding…"],
    [86, "writing index…"],
  ];
  let phaseIdx = 0;
  const tick = setInterval(() => {
    if (phaseIdx >= phases.length) return;
    const [pct, label] = phases[phaseIdx++];
    setProgress(pct, label);
  }, 520);

  try {
    const r = await api("/api/index", { method: "POST", body: { folder: state.folder } });
    clearInterval(tick);
    setProgress(100, `indexed ${r.docs} docs · ${r.chunks} chunks`);
    $("stat-docs").textContent = r.docs;
    $("stat-chunks").textContent = r.chunks;
    $("folder-stats").hidden = false;
    setStatus(`indexed · ${r.docs} docs · ${r.chunks} chunks`);
    setTimeout(() => showProgress(false), 1200);
  } catch (e) {
    clearInterval(tick);
    setProgress(0, `failed: ${e.message}`);
    $("indexing-progress").classList.add("err");
    setStatus("index failed");
    setTimeout(() => {
      showProgress(false);
      $("indexing-progress").classList.remove("err");
    }, 3200);
  } finally {
    state.indexing = false;
    $("rebuild-index").disabled = false;
  }
}

function showProgress(show) {
  $("indexing-progress").classList.toggle("hidden", !show);
  if (show) setProgress(0, "walking folder…");
}
function setProgress(pct, label) {
  $("progress-fill").style.width = pct + "%";
  $("indexing-pct").textContent = Math.round(pct) + "%";
  if (label) $("indexing-detail").textContent = label;
}

// ---------------------- ask ----------------------
async function ask() {
  const q = $("question").value.trim();
  if (!q || !state.folder || state.asking) return;
  state.asking = true;
  $("ask").disabled = true;
  $("question").disabled = true;
  setStatus("retrieving · reranking · asking…");

  // show the question immediately
  showEmpty(false);
  const block = document.createElement("div");
  block.className = "qa";
  block.innerHTML = `
    <div class="q"></div>
    <div class="a"><span class="mono-small">thinking…</span></div>
  `;
  block.querySelector(".q").textContent = q;
  $("conversation").appendChild(block);
  block.scrollIntoView ? null : null;
  $("conversation").scrollTop = $("conversation").scrollHeight;

  // ensure we have an active session
  if (!state.activeSessionId) newSession();
  const session = activeSession();
  if (session && session.qa.length === 0 && session.title === "new chat") {
    session.title = q.length > 40 ? q.slice(0, 40) + "…" : q;
  }

  try {
    const s = state.settings || {};
    const r = await api("/api/ask", {
      method: "POST",
      body: {
        folder: state.folder,
        question: q,
        top_k: s.default_top_k || 5,
        backend: s.default_backend || "Ollama",
        retrieval_mode: s.default_retrieval_mode || "fallback",
      },
    });
    block.querySelector(".a").innerHTML = formatAnswer(r.answer);
    if (r.tokens) {
      const ti = document.createElement("div");
      ti.className = "token-info";
      const p = r.tokens.prompt_tokens !== undefined || r.tokens.completion_tokens !== undefined
        ? `${r.tokens.total_tokens || 0} tokens (prompt ${r.tokens.prompt_tokens || 0} / completion ${r.tokens.completion_tokens || 0})`
        : `${r.tokens.total_tokens || 0} tokens`;
      ti.textContent = p;
      block.querySelector(".a").appendChild(ti);
    }
    if (r.sources?.length) {
      const list = document.createElement("div");
      list.className = "sources";
      r.sources.forEach((s) => {
        const row = document.createElement("div");
        row.className = "src";
        const meta = [
          s.page ? `p.${s.page}` : "",
          s.slide ? `slide ${s.slide}` : "",
          s.sheet ? `sheet ${s.sheet}` : "",
          s.section ? `"${s.section}"` : "",
        ].filter(Boolean).join(" · ");
        row.innerHTML = `
          <span class="n">[${s.n}]</span>
          <span class="path">${escape(s.source)}</span>
          <span class="meta">${escape(meta)}</span>
          <span class="score">score ${s.score.toFixed(2)}</span>
        `;
        list.appendChild(row);
      });
      block.appendChild(list);
    }
    setStatus(`answered · ${r.sources?.length || 0} sources`);

    // persist into the active session
    if (session) {
      session.qa.push({ q, a: r.answer, sources: r.sources || [], tokens: r.tokens || {} });
      session.updatedAt = Date.now();
      saveSessions();
      renderSessions();
    }
  } catch (e) {
    block.querySelector(".a").innerHTML = `<span class="mono-small" style="color:var(--err)">error: ${escape(e.message)}</span>`;
    setStatus("error · check settings");
  } finally {
    state.asking = false;
    $("ask").disabled = false;
    $("question").disabled = false;
    $("question").value = "";
    $("question").focus();
  }
}

function formatAnswer(text) {
  // turn [1] [2] into clickable superscripts
  const safe = escape(text);
  return safe.replace(/\[(\d+)\]/g, '<sup class="cite">[$1]</sup>');
}
function escape(s) {
  return (s || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function showEmpty(show) {
  $("empty").classList.toggle("hidden", !show);
  $("conversation").classList.toggle("hidden", show);
}

// ---------------------- chat sessions ----------------------
function loadSessions() {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      state.sessions = data.sessions || [];
      state.activeSessionId = data.activeSessionId || null;
    }
  } catch {}
}
function saveSessions() {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify({
      sessions: state.sessions,
      activeSessionId: state.activeSessionId,
    }));
  } catch {}
}
function activeSession() {
  return state.sessions.find((s) => s.id === state.activeSessionId) || null;
}
function newSession() {
  const s = {
    id: "s_" + Math.random().toString(36).slice(2, 9),
    title: "new chat",
    folder: state.folder || "",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    qa: [],
  };
  state.sessions.unshift(s);
  state.activeSessionId = s.id;
  saveSessions();
  renderSessions();
  renderConversation();
  return s;
}
function switchSession(id) {
  state.activeSessionId = id;
  saveSessions();
  renderSessions();
  renderConversation();
}
function deleteSession(id, evt) {
  if (evt) { evt.stopPropagation(); evt.preventDefault(); }
  state.sessions = state.sessions.filter((s) => s.id !== id);
  if (state.activeSessionId === id) {
    state.activeSessionId = state.sessions[0]?.id || null;
  }
  saveSessions();
  renderSessions();
  renderConversation();
}
function renderSessions() {
  const el = $("sessions-list");
  el.innerHTML = "";
  if (!state.sessions.length) {
    const empty = document.createElement("div");
    empty.className = "sessions-empty";
    empty.textContent = "no chats yet — ask something to start one.";
    el.appendChild(empty);
    return;
  }
  state.sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = "session" + (s.id === state.activeSessionId ? " active" : "");
    item.onclick = () => switchSession(s.id);
    item.title = s.title;

    const title = document.createElement("div");
    title.className = "session-title";
    title.textContent = s.title;

    const meta = document.createElement("div");
    meta.className = "session-meta";
    const turns = document.createElement("span");
    turns.textContent = `${s.qa.length} ${s.qa.length === 1 ? "turn" : "turns"}`;
    const dot = document.createElement("span"); dot.className = "dot"; dot.textContent = "·";
    const when = document.createElement("span");
    when.textContent = relativeTime(s.updatedAt);
    meta.appendChild(turns); meta.appendChild(dot); meta.appendChild(when);

    const del = document.createElement("button");
    del.className = "session-del";
    del.title = "delete chat";
    del.innerHTML = '<svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 6h12M8 6V4.5a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1V6M6 6l1 9.2a1 1 0 0 0 1 .9h4a1 1 0 0 0 1-.9L14 6"/></svg>';
    del.onclick = (e) => deleteSession(s.id, e);

    item.appendChild(title);
    item.appendChild(meta);
    item.appendChild(del);
    el.appendChild(item);
  });
}
function relativeTime(ts) {
  const diff = Date.now() - ts;
  const m = 60 * 1000, h = 60 * m, d = 24 * h;
  if (diff < m) return "just now";
  if (diff < h) return Math.floor(diff / m) + "m";
  if (diff < d) return Math.floor(diff / h) + "h";
  if (diff < 7 * d) return Math.floor(diff / d) + "d";
  return new Date(ts).toLocaleDateString();
}
function renderConversation() {
  const conv = $("conversation");
  conv.innerHTML = "";
  const s = activeSession();
  if (!s || s.qa.length === 0) {
    showEmpty(true);
    return;
  }
  showEmpty(false);
    s.qa.forEach((turn) => {
    const block = document.createElement("div");
    block.className = "qa";
    const q = document.createElement("div"); q.className = "q"; q.textContent = turn.q;
    const a = document.createElement("div"); a.className = "a"; a.innerHTML = formatAnswer(turn.a);
      if (turn.tokens) {
        const ti = document.createElement("div");
        ti.className = "token-info";
        const p = turn.tokens.prompt_tokens !== undefined || turn.tokens.completion_tokens !== undefined
          ? `${turn.tokens.total_tokens || 0} tokens (prompt ${turn.tokens.prompt_tokens || 0} / completion ${turn.tokens.completion_tokens || 0})`
          : `${turn.tokens.total_tokens || 0} tokens`;
        ti.textContent = p;
        a.appendChild(ti);
      }
    block.appendChild(q); block.appendChild(a);
    if (turn.sources?.length) {
      const list = document.createElement("div");
      list.className = "sources";
      turn.sources.forEach((src) => {
        const row = document.createElement("div");
        row.className = "src";
        const meta = [
          src.page ? `p.${src.page}` : "",
          src.slide ? `slide ${src.slide}` : "",
          src.sheet ? `sheet ${src.sheet}` : "",
          src.section ? `"${src.section}"` : "",
        ].filter(Boolean).join(" · ");
        row.innerHTML = `
          <span class="n">[${src.n}]</span>
          <span class="path">${escape(src.source)}</span>
          <span class="meta">${escape(meta)}</span>
          <span class="score">score ${src.score.toFixed(2)}</span>
        `;
        list.appendChild(row);
      });
      block.appendChild(list);
    }
    conv.appendChild(block);
  });
  conv.scrollTop = conv.scrollHeight;
}

// ---------------------- voice typing ----------------------
let recognition = null;
let recognitionActive = false;
function setupVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return;  // not supported — keep mic hidden
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = navigator.language || "en-US";

  let baseText = "";
  recognition.onstart = () => {
    recognitionActive = true;
    $("mic").classList.add("recording");
    setStatus("listening…");
    baseText = $("question").value;
    if (baseText && !baseText.endsWith(" ")) baseText += " ";
  };
  recognition.onend = () => {
    recognitionActive = false;
    $("mic").classList.remove("recording");
    setStatus("ready");
  };
  recognition.onerror = (e) => {
    recognitionActive = false;
    $("mic").classList.remove("recording");
    setStatus(`voice: ${e.error}`);
  };
  recognition.onresult = (e) => {
    let interim = "", final = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) final += t;
      else interim += t;
    }
    $("question").value = (baseText + final + interim).trimStart();
  };

  $("mic").hidden = false;
  $("mic").addEventListener("click", () => {
    if (!state.folder) return;
    if (recognitionActive) recognition.stop();
    else recognition.start();
  });
}

// ---------------------- settings modal ----------------------
async function openSettings() {
  const s = await api("/api/settings");
  state.settings = s;
  $("s-openai-key").value = "";
  $("s-openai-key").placeholder = s.openai_api_key_set ? `current: ${s.openai_api_key}` : "sk-...";
  $("key-hint").textContent = s.openai_api_key_set ? `currently set: ${s.openai_api_key}` : "not set";
  $("key-hint").classList.toggle("set", !!s.openai_api_key_set);

  // openai model dropdown
  const sel = $("s-openai-model");
  const custom = $("s-openai-model-custom");
  const models = s.openai_models || ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o", "o4-mini"];
  sel.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m; opt.textContent = m;
    sel.appendChild(opt);
  });
  const otherOpt = document.createElement("option");
  otherOpt.value = "__other__"; otherOpt.textContent = "other…";
  sel.appendChild(otherOpt);

  const current = s.openai_model || "gpt-4.1-mini";
  if (models.includes(current)) {
    sel.value = current;
    custom.hidden = true;
    custom.value = "";
  } else {
    sel.value = "__other__";
    custom.hidden = false;
    custom.value = current;
  }
  sel.onchange = () => {
    if (sel.value === "__other__") {
      custom.hidden = false;
      custom.focus();
    } else {
      custom.hidden = true;
    }
  };

  $("s-ollama-url").value = s.ollama_base_url || "";
  $("s-ollama-model").value = s.ollama_model || "";
  $("s-backend").value = s.default_backend || "Ollama";
  $("s-mode").value = s.default_retrieval_mode || "fallback";
  $("s-theme").value = s.theme || "dark";
  $("s-topk").value = s.default_top_k || 5;
  $("settings-modal").classList.remove("hidden");
}
function closeSettings() { $("settings-modal").classList.add("hidden"); }

async function saveSettings() {
  const sel = $("s-openai-model");
  const custom = $("s-openai-model-custom");
  const openai_model = sel.value === "__other__"
    ? (custom.value.trim() || "gpt-4.1-mini")
    : sel.value;

  const body = {
    openai_model,
    ollama_base_url: $("s-ollama-url").value.trim(),
    ollama_model: $("s-ollama-model").value.trim(),
    default_backend: $("s-backend").value,
    default_retrieval_mode: $("s-mode").value,
    theme: $("s-theme").value,
    default_top_k: parseInt($("s-topk").value, 10) || 5,
  };
  const newKey = $("s-openai-key").value.trim();
  if (newKey) body.openai_api_key = newKey;

  try {
    const s = await api("/api/settings", { method: "POST", body });
    state.settings = s;
    document.documentElement.dataset.theme = s.theme;
    if ($("meta-mode")) $("meta-mode").textContent = s.default_retrieval_mode;
    if ($("meta-model")) $("meta-model").textContent = s.default_backend === "OpenAI" ? (s.openai_model || "gpt-4.1-mini") : (s.ollama_model || "ollama");
    closeSettings();
    toast("settings saved", "ok");
    setTimeout(removeToast, 1800);
  } catch (e) {
    toast(`save failed: ${e.message}`, "err");
    setTimeout(removeToast, 3000);
  }
}

// ---------------------- bootstrap ----------------------
async function init() {
  // wait for pywebview bridge
  await new Promise((r) => {
    if (window.pywebview) r();
    else window.addEventListener("pywebviewready", r, { once: true });
  });

  const s = await api("/api/settings");
  state.settings = s;
  document.documentElement.dataset.theme = s.theme || "dark";
  $("meta-mode").textContent = s.default_retrieval_mode;
  $("meta-model").textContent = s.default_backend === "OpenAI" ? (s.openai_model || "gpt-4.1-mini") : (s.ollama_model || "ollama");

  if (s.last_folder) {
    state.folder = s.last_folder;
    $("folder-display").textContent = s.last_folder;
	$("folder-display").classList.add("highlighted");
    $("rebuild-index").disabled = false;
    $("question").disabled = false;
    $("ask").disabled = false;
    setStatus("ready · ask a question");
  }

  // health
  try {
    const h = await api("/api/health");
    $("device-pill").textContent = h.device || "cpu";
    $("device-pill").className = "pill " + (h.device === "cuda" || h.device === "mps" ? "ok" : "");
  } catch {}

  // sessions
  loadSessions();
  if (!state.sessions.length) newSession();
  renderSessions();
  renderConversation();

  // wires
  $("pick-folder").onclick = pickFolder;
  $("rebuild-index").onclick = rebuildIndex;
  $("ask").onclick = ask;
  $("new-session").onclick = () => newSession();
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
  });

  $("open-settings").onclick = openSettings;
  $("close-settings").onclick = closeSettings;
  $("settings-cancel").onclick = closeSettings;
  $("settings-save").onclick = saveSettings;
  $("reveal-key").onclick = () => {
    const inp = $("s-openai-key");
    inp.type = inp.type === "password" ? "text" : "password";
    $("reveal-key").textContent = inp.type === "password" ? "show" : "hide";
  };

  // shortcuts: ⌘, settings · space hold-to-dictate
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === ",") { e.preventDefault(); openSettings(); }
    if (e.key === "Escape") closeSettings();
  });

  setupVoice();
}

init();
