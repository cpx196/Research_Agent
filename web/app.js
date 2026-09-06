const $ = (selector) => document.querySelector(selector);
const conversation = $("#conversation");
const composer = $("#composer");
const queryInput = $("#query");
const sendButton = $("#send");
const sessionKey = "research-agent-session-id";
let sessionId = sessionStorage.getItem(sessionKey);
if (!sessionId) {
  sessionId = globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  sessionStorage.setItem(sessionKey, sessionId);
}

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function renderInlineMarkdown(value) {
  const codeTokens = [];
  let rendered = escapeText(value).replace(/`([^`\n]+)`/g, (_, code) => {
    const token = `@@INLINE_CODE_${codeTokens.length}@@`;
    codeTokens.push(`<code>${code}</code>`);
    return token;
  });
  rendered = rendered
    .replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*\n]+)\*(?!\*)/g, "<em>$1</em>")
    .replace(/(?<!_)_([^_\n]+)_(?!_)/g, "<em>$1</em>");
  codeTokens.forEach((token, index) => {
    rendered = rendered.replace(`@@INLINE_CODE_${index}@@`, token);
  });
  return rendered;
}

function renderMarkdown(value) {
  const lines = String(value ?? "").replace(/\r\n?/g, "\n").split("\n");
  const output = [];
  let paragraph = [];
  let listType = "";
  let inCode = false;
  let codeLines = [];

  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = "";
  };
  const flushParagraph = () => {
    if (paragraph.length) {
      output.push(`<p>${paragraph.map(renderInlineMarkdown).join("<br />")}</p>`);
      paragraph = [];
    }
  };
  const flushCode = () => {
    output.push(`<pre><code>${escapeText(codeLines.join("\n"))}</code></pre>`);
    codeLines = [];
  };

  const tableCells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
  const tableSeparator = (line) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (/^\s*```/.test(line)) {
      if (inCode) flushCode();
      else {
        flushParagraph();
        closeList();
      }
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      codeLines.push(line);
      continue;
    }
    if (!line.trim()) {
      flushParagraph();
      closeList();
      continue;
    }
    if (lines[index + 1] && line.includes("|") && tableSeparator(lines[index + 1])) {
      flushParagraph();
      closeList();
      const headers = tableCells(line);
      output.push(`<table><thead><tr>${headers.map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>`);
      index += 2;
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
        output.push(`<tr>${tableCells(lines[index]).map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`);
        index += 1;
      }
      output.push("</tbody></table>");
      index -= 1;
      continue;
    }
    if (/^\s*(?:---+|___+|\*\s*\*\s*\*)\s*$/.test(line)) {
      flushParagraph();
      closeList();
      output.push("<hr />");
      continue;
    }
    const quote = line.match(/^\s*>\s?(.*)$/);
    if (quote) {
      flushParagraph();
      closeList();
      output.push(`<blockquote>${renderInlineMarkdown(quote[1])}</blockquote>`);
      continue;
    }
    const heading = line.match(/^\s*(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      flushParagraph();
      closeList();
      const level = heading[1].length;
      output.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unordered || ordered) {
      flushParagraph();
      const nextType = unordered ? "ul" : "ol";
      if (listType !== nextType) {
        closeList();
        output.push(`<${nextType}>`);
        listType = nextType;
      }
      output.push(`<li>${renderInlineMarkdown((unordered || ordered)[1])}</li>`);
      continue;
    }
    closeList();
    paragraph.push(line);
  }
  if (inCode) flushCode();
  flushParagraph();
  closeList();
  return output.join("") || "<p></p>";
}

function setMessageBody(body, text, markdown = false) {
  if (markdown) body.innerHTML = renderMarkdown(text);
  else body.textContent = text;
}

function addMessage(kind, label, text, meta = "") {
  const article = document.createElement("article");
  article.className = `message ${kind}`;
  article.innerHTML = `<div class="message-head"><span class="tag">${escapeText(label)}</span><span>${escapeText(meta)}</span></div>`;
  const body = document.createElement("div");
  body.className = `message-body${kind === "assistant-message" ? " markdown" : ""}`;
  setMessageBody(body, text, kind === "assistant-message");
  article.appendChild(body);
  conversation.appendChild(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

let claimAuditSnapshot = { planned_claims: [], claims: [], relations: [], repairs: [] };

function renderClaimAudit(payload) {
  const list = $("#audit-list");
  const summary = $("#audit-summary");
  if (String(payload?.stage || "").toLowerCase() === "researching") {
    claimAuditSnapshot = { planned_claims: [], claims: [], relations: [], repairs: [] };
  }
  for (const key of ["planned_claims", "claims", "relations", "repairs"]) {
    if (Array.isArray(payload?.[key])) claimAuditSnapshot[key] = payload[key];
  }
  const planned = claimAuditSnapshot.planned_claims;
  const claims = claimAuditSnapshot.claims;
  const relations = claimAuditSnapshot.relations;
  const repairs = claimAuditSnapshot.repairs;
  const repairByClaim = new Map(repairs.map((item) => [item.claim_id, item]));
  const relationsByClaim = new Map();
  for (const relation of relations) {
    const claimId = String(relation.claim_id || "");
    if (!relationsByClaim.has(claimId)) relationsByClaim.set(claimId, []);
    relationsByClaim.get(claimId).push(relation);
  }
  const stage = String(payload?.stage || "").toUpperCase();
  summary.textContent = `${planned.length} PLANNED / ${claims.length} OUTPUT / ${repairs.filter((item) => item.changed).length} REPAIRS${stage ? ` / ${stage}` : ""}`;
  if (!claims.length && !planned.length) {
    list.innerHTML = stage === "RESEARCHING"
      ? '<div class="audit-empty">正在检索和构建初稿；完整句子生成后将实时显示 provisional Claim。</div>'
      : '<div class="audit-empty">本次路径没有生成需要审计的研究 Claim。</div>';
    return;
  }
  const plannedCards = planned.map((claim) => `<article class="audit-card" data-status="PLANNED">
    <div class="audit-card-head"><span>${escapeText(claim.claim_id || "PLAN")} / CLAIM CONTRACT</span><span class="audit-status">${escapeText(claim.allowed_strength || "conditional")}</span></div>
    <div class="audit-card-text">${escapeText(claim.text || "")}</div>
    <div class="audit-evidence">Evidence: ${escapeText((claim.evidence_ids || []).join(", ") || "insufficient")}${claim.conditions?.length ? `<br />Conditions: ${escapeText(claim.conditions.join("；"))}` : ""}</div>
  </article>`).join("");
  const outputCards = claims.map((claim) => {
    const repair = repairByClaim.get(claim.claim_id);
    const evidence = (relationsByClaim.get(String(claim.claim_id || "")) || []).slice(0, 4);
    const evidenceHtml = evidence.length ? `<div class="audit-evidence">${evidence.map((item) => {
      const location = [item.source || item.title || "source", item.page ? `p.${item.page}` : "", item.evidence_role || ""].filter(Boolean).join(" · ");
      const quote = String(item.quote || "").replace(/\s+/g, " ").slice(0, 220);
      return `<div><strong>${escapeText(location)}</strong>${quote ? `<br />“${escapeText(quote)}”` : ""}</div>`;
    }).join("")}</div>` : '<div class="audit-evidence">No direct supporting passage.</div>';
    const diff = repair?.changed
      ? `<div class="repair-diff">${escapeText(repair.repair_action || "REPAIR")}<br />− ${escapeText(repair.original_claim || "")}<br />+ ${escapeText(repair.final_claim || "")}</div>`
      : "";
    return `<article class="audit-card" data-status="${escapeText(claim.status || "UNSUPPORTED")}">
      <div class="audit-card-head"><span>${escapeText(claim.claim_id || "CLAIM")} / ${escapeText(claim.claim_type || "fact")}</span><span class="audit-status">${escapeText(claim.status || "UNSUPPORTED")}</span></div>
      <div class="audit-card-text">${escapeText(claim.repaired_text || claim.text || "")}</div>${evidenceHtml}${diff}
    </article>`;
  }).join("");
  list.innerHTML = plannedCards + outputCards;
}

function setBusy(busy) {
  sendButton.disabled = busy;
  composer.setAttribute("aria-busy", String(busy));
  sendButton.querySelector("span:first-child").textContent = busy ? "RUNNING..." : "SEND QUERY";
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    $("#connection-label").textContent = "CONNECTED";
    document.body.dataset.mode = String(data.mode || "unknown");
  } catch (error) {
    $("#connection-label").textContent = "OFFLINE";
    document.body.dataset.mode = "offline";
  }
}

async function submitQuery(query) {
  query = query.trim();
  if (!query || sendButton.disabled) return;
  addMessage("user-message", "USER", query, new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}));
  queryInput.value = "";
  setBusy(true);
  const trace = $("#trace");
  if (!trace.dataset.hasRuns) {
    trace.textContent = "";
    trace.dataset.hasRuns = "true";
  }
  const traceQuery = query.replace(/\s+/g, " ").slice(0, 120);
  trace.textContent += `${trace.textContent ? "\n" : ""}========================================================================\n[SESSION QUERY] ${traceQuery}\n`;
  trace.scrollTop = trace.scrollHeight;
  $("#trace-meta").textContent = "RUNNING / WAITING FOR GRAPH EVENTS";
  $("#latency").textContent = "LATENCY --";
  $("#evidence").textContent = "EVIDENCE --";
  $("#log-file").textContent = "LOG WRITING";
  const loading = addMessage("loading-message", "AGENT", "正在运行 LangGraph...", "STREAMING");
  let answerArticle = null;
  let answerBody = null;
  let answerText = "";
  let completed = false;
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({query, session_id: sessionId}),
    });
    if (!response.ok || !response.body) throw new Error("Streaming connection failed");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = "";
    const processBlock = (block) => {
      if (!block.trim()) return;
      const lines = block.split("\n");
      const eventName = (lines.find((line) => line.startsWith("event:")) || "event: message").slice(6).trim();
      const dataLine = lines.find((line) => line.startsWith("data:"));
      if (!dataLine) return;
      const data = JSON.parse(dataLine.slice(5).trim());
      if (eventName === "trace") {
        const trace = $("#trace");
        trace.textContent += `${data.line}\n`;
        trace.scrollTop = trace.scrollHeight;
      } else if (eventName === "claim_audit") {
        renderClaimAudit(data || {});
      } else if (eventName === "answer_start") {
        loading.remove();
        answerArticle = addMessage("assistant-message", "AGENT", "", "STREAMING");
        answerBody = answerArticle.querySelector(".message-body");
      } else if (eventName === "answer_chunk") {
        if (!answerBody) {
          loading.remove();
          answerArticle = addMessage("assistant-message", "AGENT", "", "STREAMING");
          answerBody = answerArticle.querySelector(".message-body");
        }
        answerText += data.text || "";
        setMessageBody(answerBody, answerText, true);
        conversation.scrollTop = conversation.scrollHeight;
      } else if (eventName === "answer_reset") {
        answerText = "";
        if (answerBody) setMessageBody(answerBody, "", true);
        if (answerArticle) answerArticle.querySelector(".message-head").lastElementChild.textContent = "FINALIZING / VERIFIED";
      } else if (eventName === "done") {
        completed = true;
        const stats = data.stats || {};
        if (answerBody && typeof data.answer === "string" && answerBody.textContent !== data.answer) {
          answerText = data.answer;
          setMessageBody(answerBody, data.answer, true);
        }
        if (answerArticle) answerArticle.querySelector(".message-head").lastElementChild.textContent = `${data.mode || "GRAPH"} / COMPLETE`;
        $("#trace-meta").textContent = `${stats.graph_steps ?? "--"} GRAPH NODES  /  ${stats.tool_calls ?? "--"} TOOL CALLS`;
        $("#latency").textContent = `LATENCY ${stats.request_latency_seconds ?? stats.latency_seconds ?? "--"}S`;
        $("#evidence").textContent = `EVIDENCE ${stats.evidence ?? "--"}`;
        $("#log-file").textContent = data.log_error ? "LOG ERROR" : `LOG ${data.log_file || "--"}`;
        renderClaimAudit(data.claim_audit || {});
      } else if (eventName === "error") {
        $("#log-file").textContent = data.log_error ? "LOG ERROR" : `LOG ${data.log_file || "--"}`;
        throw new Error(data.error || "Request failed");
      }
    };
    while (true) {
      const {value, done} = await reader.read();
      sseBuffer += decoder.decode(value || new Uint8Array(), {stream: !done});
      const blocks = sseBuffer.split("\n\n");
      sseBuffer = blocks.pop() || "";
      blocks.forEach(processBlock);
      if (completed) {
        await reader.cancel();
        break;
      }
      if (done) break;
    }
    if (!completed) throw new Error("Streaming connection closed before completion");
  } catch (error) {
    loading.remove();
    addMessage("error-message", "ERROR", error.message, "REQUEST FAILED");
  } finally {
    setBusy(false);
    queryInput.focus();
    refreshHealth();
  }
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuery(queryInput.value);
});

refreshHealth();
