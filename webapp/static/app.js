/* Lutong Bahay — chat interface.
   Multi-turn conversation, streamed answers, recipe modal + side panel. */

const STORAGE_KEY = "lutong-bahay:conversations:v1";
const SIDEBAR_KEY = "lutong-bahay:sidebar:v1";
const MAX_CONVERSATIONS = 30;
const MAX_INPUT_HEIGHT = 180;

const RAT_IMG = `<img src="/static/rat-pizza.png" alt="">`;

const STARTERS = [
  { label: "Chicken adobo", text: "manok, toyo, suka, bawang, sibuyas, dahon ng laurel" },
  { label: "Sinigang", text: "baboy, sampalok, kangkong, kamatis, sibuyas" },
  { label: "Ginisang sitaw", text: "may sitaw, bawang, at manok ako" },
  { label: "Tilapia", text: "tilapia, kamatis, sibuyas, luya" },
];

const SUGGESTIONS = [
  "Pick the fastest one",
  "Suggest substitutions",
  "Show the cooking steps",
  "Show me more recipes",
];

const state = {
  conversations: [],
  activeId: null,
  health: null,
  streaming: false,
  composerEnabled: true,
  view: "closed",
  openRecipe: null,
};

const $ = (selector) => document.querySelector(selector);

/* ---------- storage ---------- */

function uid() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.conversations)) state.conversations = parsed.conversations;
    if (parsed.activeId) state.activeId = parsed.activeId;
  } catch {
    state.conversations = [];
    state.activeId = null;
  }

  if (!activeConversation() && state.conversations.length) {
    state.activeId = [...state.conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
  }

  // A reload during streaming leaves a pending assistant with no text.
  for (const conversation of state.conversations) {
    for (const message of conversation.messages || []) {
      if (message.pending) {
        message.pending = false;
        if (!message.content) message.error = "This answer was interrupted. Ask again.";
      }
    }
  }
}

function saveStore() {
  try {
    state.conversations = [...state.conversations]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, MAX_CONVERSATIONS);
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ conversations: state.conversations, activeId: state.activeId })
    );
  } catch {
    /* storage full or unavailable — the chat still works for this session */
  }
}

/* ---------- helpers ---------- */

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}

function inline(text) {
  return text
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}

/** Tiny, safe markdown renderer: headings, lists, bold, italics, code. */
function renderMarkdown(source) {
  const lines = esc(source).split("\n");
  const out = [];
  let list = null;
  let paragraph = [];

  const flushParagraph = () => {
    if (paragraph.length) {
      out.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (list) {
      out.push(`</${list}>`);
      list = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      flushParagraph();
      closeList();
      continue;
    }

    let match;
    if ((match = line.match(/^(#{1,4})\s+(.*)$/))) {
      flushParagraph();
      closeList();
      const level = Math.min(match[1].length + 1, 4);
      out.push(`<h${level}>${inline(match[2])}</h${level}>`);
      continue;
    }
    if ((match = line.match(/^[-*•]\s+(.*)$/))) {
      flushParagraph();
      if (list !== "ul") {
        closeList();
        out.push("<ul>");
        list = "ul";
      }
      out.push(`<li>${inline(match[1])}</li>`);
      continue;
    }
    if ((match = line.match(/^\d+[.)]\s+(.*)$/))) {
      flushParagraph();
      if (list !== "ol") {
        closeList();
        out.push("<ol>");
        list = "ol";
      }
      out.push(`<li>${inline(match[1])}</li>`);
      continue;
    }

    closeList();
    paragraph.push(line);
  }

  flushParagraph();
  closeList();
  return out.join("");
}

async function readError(response) {
  try {
    const body = await response.json();
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg || String(item)).join("; ");
    }
    return body.detail || body.message || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

function relativeTime(timestamp) {
  const diff = Date.now() - timestamp;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return new Date(timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function titleFrom(text) {
  const clean = String(text).replace(/\s+/g, " ").trim();
  return clean.length > 46 ? `${clean.slice(0, 46).trimEnd()}…` : clean;
}

function chipList(items, className, limit = 10) {
  if (!items || !items.length) return "";
  const shown = items
    .slice(0, limit)
    .map((item) => `<span class="chip ${className}">${esc(item)}</span>`)
    .join("");
  const rest = items.length - limit;
  return shown + (rest > 0 ? `<span class="chip ${className}">+${rest} more</span>` : "");
}

/* ---------- conversations ---------- */

function activeConversation() {
  return state.conversations.find((conversation) => conversation.id === state.activeId) || null;
}

function createConversation() {
  const conversation = {
    id: uid(),
    title: "New chat",
    pantry: "",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  state.conversations.unshift(conversation);
  state.activeId = conversation.id;
  return conversation;
}

function newChat() {
  const current = activeConversation();
  if (current && current.messages.length === 0) {
    $("#input").value = "";
    autoGrow();
    $("#input").focus();
    closeSidebar();
    return;
  }

  createConversation();
  state.openRecipe = null;
  setView("closed");
  $("#input").value = "";
  autoGrow();
  saveStore();
  renderConversations();
  renderChat();
  scrollToBottom(true);
  closeSidebar();
  $("#input").focus();
}

function selectConversation(id) {
  state.activeId = id;
  state.openRecipe = null;
  setView("closed");
  saveStore();
  renderConversations();
  renderChat();
  scrollToBottom(true);
  closeSidebar();
}

function deleteConversation(id) {
  state.conversations = state.conversations.filter((conversation) => conversation.id !== id);
  if (state.activeId === id) {
    state.activeId = state.conversations.length ? state.conversations[0].id : null;
  }
  saveStore();
  renderConversations();
  renderChat();
  scrollToBottom(true);
}

function renderConversations() {
  const list = $("#conv-list");
  const empty = $("#conv-empty");
  const conversations = [...state.conversations].sort((a, b) => b.updatedAt - a.updatedAt);

  empty.hidden = conversations.length > 0;
  list.innerHTML = conversations
    .map(
      (conversation) => `<div class="conv-item ${conversation.id === state.activeId ? "active" : ""}">
        <button class="conv" type="button" data-conv="${conversation.id}">
          <span class="conv-title">${esc(conversation.title)}</span>
          <span class="conv-time">${relativeTime(conversation.updatedAt)}</span>
        </button>
        <button class="conv-del" type="button" data-del="${conversation.id}" aria-label="Delete conversation">×</button>
      </div>`
    )
    .join("");
}

/* ---------- chat rendering ---------- */

function recipeMini(recipe, variant) {
  const missing = recipe.missing && recipe.missing.length
    ? ` · ${recipe.missing.length} missing`
    : " · nothing missing";
  const cls = variant ? `recipe-mini ${variant}` : "recipe-mini";
  const rank = variant === "version"
    ? `<span class="mini-rank">v${recipe.rank}</span>`
    : `<span class="mini-rank">${recipe.rank}</span>`;
  return `<button class="${cls}" type="button" data-row="${recipe.row}">
    <span class="mini-top">
      ${rank}
      <span class="mini-name">${esc(recipe.name)}</span>
    </span>
    <span class="mini-meta">${esc(recipe.time)}${missing}</span>
    <span class="mini-open">View recipe →</span>
  </button>`;
}

function constraintChips(constraints) {
  if (!constraints) return "";
  const chips = [];
  if (constraints.diet) chips.push(`<span class="chip c-diet">${esc(constraints.diet)}</span>`);
  if (constraints.max_minutes != null) {
    chips.push(`<span class="chip c-time">≤ ${constraints.max_minutes} min</span>`);
  }
  if (constraints.diet_fallback) {
    chips.push(`<span class="chip c-warn">no ${esc(constraints.diet)} match — closest shown</span>`);
  }
  if (constraints.time_fallback) {
    chips.push(`<span class="chip c-warn">over your time budget</span>`);
  }
  return chips.length ? `<div class="constraint-row">${chips.join("")}</div>` : "";
}

function recipeBlock(message) {
  const recipes = message.recipes || [];
  if (!recipes.length) return "";

  const rest = recipes.slice(1);
  const revealed = Boolean(message.moreRevealed);
  const parts = ['<div class="recipe-stack">'];

  parts.push(`<div class="recipe-strip primary">${recipeMini(recipes[0])}</div>`);

  if (message.versions && message.versions.length) {
    parts.push(
      `<div class="version-wrap">` +
      `<p class="version-label">Other versions of this dish</p>` +
      `<div class="recipe-strip versions">${message.versions
        .map((recipe) => recipeMini(recipe, "version"))
        .join("")}</div>` +
      `</div>`
    );
  }

  if (rest.length) {
    parts.push(
      `<div class="recipe-strip more" ${revealed ? "" : "hidden"}>` +
      rest.map((recipe) => recipeMini(recipe)).join("") +
      `</div>`
    );
    if (!revealed) {
      const word = rest.length === 1 ? "recipe" : "recipes";
      parts.push(`<button class="more-btn" type="button" data-more>Show ${rest.length} more ${word}</button>`);
    }
  }

  parts.push("</div>");
  return parts.join("");
}

function renderMessage(message) {
  if (message.role === "user") {
    return `<div class="msg user" data-msg-id="${message.id}">
      <div class="bubble">${esc(message.content)}</div>
    </div>`;
  }

  const answer = message.pending
    ? esc(message.content)
    : renderMarkdown(message.content || "");

  const recipes = recipeBlock(message);

  const suggestions = !message.pending && message.suggestions && message.suggestions.length
    ? `<div class="suggestions">${message.suggestions
        .map((item) => `<button class="suggestion" type="button" data-suggest="${esc(item)}">${esc(item)}</button>`)
        .join("")}</div>`
    : "";

  const error = message.error
    ? `<div class="answer-error">${esc(message.error)}</div>`
    : "";

  const foot = !message.pending && message.mode
    ? `<div class="msg-foot">${message.mode === "llm" ? "llama3.2:3b · retrieved context" : "local demo · no LLM"}</div>`
    : "";

  return `<div class="msg assistant" data-msg-id="${message.id}">
    <div class="avatar">${RAT_IMG}</div>
    <div class="assistant-body">
      <div class="answer${message.pending ? " streaming" : ""}">${answer}</div>
      ${error}
      ${constraintChips(message.constraints)}
      ${recipes}
      ${suggestions}
      ${foot}
    </div>
  </div>`;
}

function renderChat() {
  const conversation = activeConversation();
  const hasMessages = Boolean(conversation && conversation.messages.length);

  $("#welcome").hidden = hasMessages;
  $("#messages").innerHTML = hasMessages
    ? conversation.messages.map(renderMessage).join("")
    : "";

  updateHeader();
}

function updateHeader() {
  const conversation = activeConversation();
  $("#chat-title").textContent = conversation ? conversation.title : "New chat";
  $("#chat-sub").textContent = conversation && conversation.pantry
    ? `Pantry · ${conversation.pantry}`
    : "Filipino recipe assistant";
}

function renderStarters() {
  $("#starters").innerHTML = STARTERS.map(
    (starter, index) => `<button class="starter" type="button" data-starter="${index}">
      <b>${esc(starter.label)}</b>
      <span>${esc(starter.text)}</span>
    </button>`
  ).join("");
}

/* ---------- scrolling ---------- */

function isNearBottom() {
  const scroller = $("#chat-scroll");
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 160;
}

function scrollToBottom(force) {
  const scroller = $("#chat-scroll");
  if (force || isNearBottom()) scroller.scrollTop = scroller.scrollHeight;
}

/* ---------- streaming ---------- */

async function consumeSSE(response, handlers) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data && handlers[event]) handlers[event](JSON.parse(data));
    }
  }
}

function setStreaming(value) {
  state.streaming = value;
  $("#send").disabled = value || !state.composerEnabled;
  $("#input").setAttribute("aria-busy", String(value));
}

function updateStreaming(message) {
  const answer = document.querySelector(`[data-msg-id="${message.id}"] .answer`);
  if (!answer) return;
  answer.textContent = message.content;
  scrollToBottom(false);
}

function updateMeta(message) {
  const body = document.querySelector(`[data-msg-id="${message.id}"] .assistant-body`);
  if (!body) return;

  // Constraint chips appear right after the answer text.
  let chips = body.querySelector(".constraint-row");
  const chipsHtml = constraintChips(message.constraints);
  if (chipsHtml) {
    if (chips) chips.outerHTML = chipsHtml;
    else {
      const answer = body.querySelector(".answer");
      if (answer) answer.insertAdjacentHTML("afterend", chipsHtml);
    }
  } else if (chips) {
    chips.remove();
  }

  const existing = body.querySelector(".recipe-stack, .recipe-strip, .recipe-ref");
  const html = recipeBlock(message);

  if (!html) {
    if (existing) existing.remove();
    return;
  }

  if (existing) {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = html;
    existing.replaceWith(wrapper.firstElementChild);
    return;
  }

  const anchor = body.querySelector(".constraint-row") || body.querySelector(".answer");
  if (anchor) anchor.insertAdjacentHTML("afterend", html);
}

async function send(raw) {
  const content = String(raw ?? "").trim();
  if (!content || state.streaming) return;

  if (!state.health || !state.health.ok) {
    showSetup(state.health?.error || "Recipe data is not loaded.");
    return;
  }

  let conversation = activeConversation();
  if (!conversation) conversation = createConversation();

  const firstUser = !conversation.messages.some((message) => message.role === "user");
  if (firstUser) {
    conversation.pantry = content;
    conversation.title = titleFrom(content);
  }

  conversation.messages.push({ id: uid(), role: "user", content });

  const assistant = {
    id: uid(),
    role: "assistant",
    content: "",
    pending: true,
    recipes: [],
    versions: [],
    constraints: null,
    more_count: 0,
    moreRevealed: false,
    mode: null,
    suggestions: [],
  };
  conversation.messages.push(assistant);
  conversation.updatedAt = Date.now();

  $("#input").value = "";
  autoGrow();
  saveStore();
  renderConversations();
  renderChat();
  setStreaming(true);
  scrollToBottom(true);

  const history = conversation.messages
    .filter(
      (message) =>
        message.role === "user" ||
        (message.role === "assistant" && !message.pending && message.content)
    )
    .map((message) => ({ role: message.role, content: message.content }));

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pantry: conversation.pantry,
        messages: history,
        mode: "auto",
        k: 3,
        focus_row: state.openRecipe ? state.openRecipe.row : null,
      }),
    });
    if (!response.ok) throw new Error(await readError(response));

    await consumeSSE(response, {
      meta(data) {
        assistant.recipes = data.recipes || [];
        assistant.versions = data.versions || [];
        assistant.constraints = data.constraints || null;
        assistant.more_count = data.more_count || 0;
        assistant.mode = data.mode;
        updateMeta(assistant);
      },
      token(data) {
        assistant.content += data.text;
        updateStreaming(assistant);
      },
      error(data) {
        assistant.error = data.message;
      },
      done(data) {
        assistant.mode = data.mode;
      },
    });
  } catch (error) {
    assistant.error = error.message || String(error);
  } finally {
    assistant.pending = false;
    if (!assistant.error && assistant.content) assistant.suggestions = SUGGESTIONS;
    conversation.updatedAt = Date.now();
    saveStore();
    renderConversations();
    renderChat();
    setStreaming(false);
    scrollToBottom(false);
  }
}

/* ---------- recipe modal / side panel ---------- */

function findMessageById(id) {
  const conversation = activeConversation();
  if (!conversation) return null;
  return conversation.messages.find((message) => message.id === id) || null;
}

function findRecipe(row) {
  const conversation = activeConversation();
  if (!conversation) return null;
  for (const message of conversation.messages) {
    const match = (message.recipes || []).find((recipe) => recipe.row === row);
    if (match) return match;
    const version = (message.versions || []).find((recipe) => recipe.row === row);
    if (version) return version;
  }
  return null;
}

function recipeDetail(recipe) {
  const ingredientLines = recipe.ingredient_lines && recipe.ingredient_lines.length
    ? recipe.ingredient_lines
    : recipe.ingredients || [];
  const steps = recipe.instructions && recipe.instructions.length
    ? recipe.instructions
    : recipe.steps || [];

  const missingCount = recipe.missing ? recipe.missing.length : 0;
  const facts = [
    ["Total time", recipe.time || "unknown"],
    ["Servings", recipe.servings || "—"],
    ["Missing", `${missingCount} ingredient${missingCount === 1 ? "" : "s"}`],
  ];

  const ingredientItems = ingredientLines.length
    ? ingredientLines
        .map(
          (line) => `<li><label><input type="checkbox"><span>${esc(line)}</span></label></li>`
        )
        .join("")
    : `<li><label><span>No ingredient list available.</span></label></li>`;

  const methodItems = steps.length
    ? steps
        .map(
          (step, index) =>
            `<li><span class="step-num">${index + 1}</span><p>${esc(step)}</p></li>`
        )
        .join("")
    : `<li><span class="step-num">1</span><p>No instructions available.</p></li>`;

  const chips = [];
  if (recipe.matched && recipe.matched.length) {
    chips.push(`<div class="chip-row"><b>You have</b>${chipList(recipe.matched, "matched", 8)}</div>`);
  }
  if (recipe.missing && recipe.missing.length) {
    chips.push(`<div class="chip-row"><b>Missing</b>${chipList(recipe.missing, "missing", 8)}</div>`);
  }

  return `<article class="recipe-detail">
    <header class="recipe-hero">
      <p class="recipe-kicker">Retrieved recipe · rank ${recipe.rank}</p>
      <h2>${esc(recipe.name)}</h2>
      <div class="recipe-facts">
        ${facts
          .map(
            ([label, value]) =>
              `<div class="recipe-fact"><b>${label}</b><span>${esc(value)}</span></div>`
          )
          .join("")}
      </div>
      <div class="chips">${chips.join("")}</div>
    </header>
    <div class="recipe-columns">
      <section>
        <h3>Ingredients</h3>
        <ul class="ingredient-list">${ingredientItems}</ul>
      </section>
      <section>
        <h3>Method</h3>
        <ol class="method-list">${methodItems}</ol>
      </section>
    </div>
  </article>`;
}

function renderRecipeViews() {
  if (!state.openRecipe) return;
  const html = recipeDetail(state.openRecipe);
  $("#modal-body").innerHTML = html;
  $("#panel-body").innerHTML = html;
}

function openRecipe(row) {
  const recipe = findRecipe(row);
  if (!recipe) return;

  const nextView = state.view === "panel" ? "panel" : "modal";
  state.openRecipe = recipe;
  renderRecipeViews();
  setView(nextView);
}

function setView(view) {
  state.view = view;

  $("#modal").hidden = view !== "modal";
  $("#recipe-panel").hidden = view !== "panel";
  $("#app").classList.toggle("panel-open", view === "panel");
  document.body.classList.toggle("modal-open", view === "modal");

  if (view === "closed") {
    state.openRecipe = null;
    $("#input").focus();
  }
  if (view === "modal") $("#close-modal").focus();
  if (view === "panel") $("#close-panel").focus();
}

/* ---------- sidebar (mobile overlay + desktop collapse) ---------- */

function isMobileLayout() {
  return window.matchMedia("(max-width: 900px)").matches;
}

function openSidebar() {
  $("#sidebar").classList.add("open");
  $("#scrim").hidden = false;
}

function closeSidebar() {
  $("#sidebar").classList.remove("open");
  $("#scrim").hidden = true;
}

function applySidebarPreference() {
  let collapsed = false;
  try {
    collapsed = localStorage.getItem(SIDEBAR_KEY) === "collapsed";
  } catch {
    collapsed = false;
  }
  $("#app").classList.toggle("sidebar-collapsed", collapsed && !isMobileLayout());
}

function toggleSidebar() {
  if (isMobileLayout()) {
    if ($("#sidebar").classList.contains("open")) closeSidebar();
    else openSidebar();
    return;
  }
  const collapsed = $("#app").classList.toggle("sidebar-collapsed");
  try {
    localStorage.setItem(SIDEBAR_KEY, collapsed ? "collapsed" : "open");
  } catch {
    /* preference is cosmetic; ignore storage failures */
  }
}

/* ---------- health ---------- */

function showSetup(message) {
  const banner = $("#setup-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function setComposerEnabled(enabled) {
  state.composerEnabled = enabled;
  $("#input").disabled = !enabled;
  $("#send").disabled = !enabled || state.streaming;
}

function renderHealth(health) {
  const chip = $("#health-chip");
  const text = $("#health-text");
  const pill = $("#model-pill");
  chip.classList.remove("ok", "warn", "bad");

  if (!health.ok) {
    chip.classList.add("bad");
    text.textContent = "setup needed";
    showSetup(health.error || "Recipe data is not loaded.");
    setComposerEnabled(false);
    return;
  }

  $("#setup-banner").hidden = true;
  setComposerEnabled(true);

  const ollama = health.ollama || {};
  if (ollama.available && ollama.has_model) {
    chip.classList.add("ok");
    text.textContent = "Ollama ready";
    pill.innerHTML = `<span class="dot"></span> ${esc(health.llm_model || "llama3.2:3b")}`;
    pill.title = "Answers are written by your local Ollama model";
  } else if (ollama.available) {
    chip.classList.add("warn");
    text.textContent = "model missing";
    pill.innerHTML = `<span class="dot"></span> demo answers`;
    pill.title = `Ollama is running, but ${health.llm_model || "the model"} is not pulled`;
  } else {
    chip.classList.add("warn");
    text.textContent = "demo mode";
    pill.innerHTML = `<span class="dot"></span> demo answers`;
    pill.title = "Ollama is offline, so answers come from the local demo";
  }

  chip.title = ollama.available
    ? `Ollama at ${ollama.host}`
    : `Ollama not reachable at ${ollama.host}. Click to re-check.`;
}

async function loadHealth(force = false) {
  try {
    const response = await fetch(`/api/health${force ? "?fresh=1" : ""}`);
    const health = await response.json();
    state.health = health;
    renderHealth(health);
  } catch {
    state.health = null;
    renderHealth({
      ok: false,
      error: "Cannot reach the web server. Is it still running?",
    });
  }
}

/* ---------- composer ---------- */

function autoGrow() {
  const input = $("#input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, MAX_INPUT_HEIGHT)}px`;
}

/* ---------- wiring ---------- */

function init() {
  loadStore();
  applySidebarPreference();
  renderStarters();
  renderConversations();
  renderChat();
  scrollToBottom(true);
  loadHealth();

  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    send($("#input").value);
  });

  $("#input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send($("#input").value);
    }
  });

  $("#input").addEventListener("input", autoGrow);

  $("#new-chat").addEventListener("click", newChat);
  $("#health-chip").addEventListener("click", () => loadHealth(true));
  $("#menu-btn").addEventListener("click", toggleSidebar);
  $("#scrim").addEventListener("click", closeSidebar);
  window.addEventListener("resize", applySidebarPreference);

  $("#conv-list").addEventListener("click", (event) => {
    const remove = event.target.closest("[data-del]");
    if (remove) {
      event.stopPropagation();
      deleteConversation(remove.dataset.del);
      return;
    }
    const item = event.target.closest("[data-conv]");
    if (item) selectConversation(item.dataset.conv);
  });

  $("#messages").addEventListener("click", (event) => {
    const moreButton = event.target.closest("[data-more]");
    if (moreButton) {
      const container = moreButton.closest(".msg");
      const message = findMessageById(container && container.dataset.msgId);
      if (message) {
        message.moreRevealed = true;
        saveStore();
        renderChat();
        scrollToBottom(false);
      }
      return;
    }
    const card = event.target.closest("[data-row]");
    if (card) {
      openRecipe(Number(card.dataset.row));
      return;
    }
    const suggestion = event.target.closest("[data-suggest]");
    if (suggestion) send(suggestion.dataset.suggest);
  });

  $("#starters").addEventListener("click", (event) => {
    const button = event.target.closest("[data-starter]");
    if (!button) return;
    const starter = STARTERS[Number(button.dataset.starter)];
    if (starter) send(starter.text);
  });

  $("#dock-recipe").addEventListener("click", () => setView("panel"));
  $("#expand-recipe").addEventListener("click", () => setView("modal"));
  $("#close-modal").addEventListener("click", () => setView("closed"));
  $("#close-panel").addEventListener("click", () => setView("closed"));
  $("#modal-backdrop").addEventListener("click", () => setView("closed"));

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (state.view !== "closed") setView("closed");
    else if ($("#sidebar").classList.contains("open")) closeSidebar();
  });

  autoGrow();
  $("#input").focus();
}

init();
