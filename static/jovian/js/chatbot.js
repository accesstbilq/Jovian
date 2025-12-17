const sessionId =
  'sess-' +
  Math.random().toString(36).slice(2, 10) +
  '-' +
  Date.now().toString(36);

const STREAM_URL = '/api/chat';

const messagesEl = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const firstChatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const firstMessageInput = document.getElementById('firstMessageInput');
const emptyState = document.getElementById('emptyState');
const sendFirstMessage = document.getElementById('sendFirstMessage');
const fotterChat = document.getElementById('fotterChat');

let controller = null;

/* ---------- HELPERS ---------- */

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function escapeHtml(s) {
  return s
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
    .replaceAll('\n', '<br/>');
}

/* ---------- UI ---------- */

function appendUserMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'flex justify-end';

  const bubble = document.createElement('div');
  bubble.className = 'bg-[#F6E339] text-black rounded-xl p-4 max-w-2xl text-sm';

  bubble.innerHTML = escapeHtml(text);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollToBottom();
}

function appendAgentMessage() {
  const wrap = document.createElement('div');
  wrap.className = 'flex justify-start';

  const card = document.createElement('div');
  card.className =
    'bg-surface dark:bg-darkSurface border border-border dark:border-darkBorder rounded-xl p-5 shadow-sm';

  const title = document.createElement('div');
  title.className = 'font-semibold mb-1';
  title.textContent = '';

  const subtitle = document.createElement('div');
  subtitle.className = 'text-xs text-muted dark:text-darkTextSecondary mb-3';
  subtitle.textContent = '';

  const body = document.createElement('div');
  body.className = 'text-sm leading-relaxed flex items-center gap-2';

  // typing dot
  const dot = document.createElement('div');
  dot.className = 'typing-dot';

  body.appendChild(dot);

  card.append(title, subtitle, body);
  wrap.appendChild(card);
  messagesEl.appendChild(wrap);

  scrollToBottom();

  //   return { body, dot };
  return body;
}

/* ---------- STREAM ---------- */

async function startStream(e) {
    console.log("START STREAM")
  e.preventDefault();

  let text = messageInput.value.trim();

  if (emptyState) emptyState.remove();
  if (sendFirstMessage && firstMessageInput.value) {
    console.log("FIRST MESSAGE")
    text = firstMessageInput.value.trim();
    fotterChat.classList.remove('hidden');
    sendFirstMessage.remove();
  }

  console.log("TEXT OF MESSAGE ", text)
  if (!text) return;

  appendUserMessage(text);
  messageInput.value = '';
  firstMessageInput.value = '';

  if (controller) controller.abort();
  controller = new AbortController();

  const agentBody = appendAgentMessage();
  let streamedText = '';

  const formData = new FormData();
  formData.append('user_message', text);
  formData.append('session_id', sessionId);

  try {
    const resp = await fetch(STREAM_URL, {
      method: 'POST',
      body: formData,
      signal: controller.signal
    });

    resetTextarea();

    if (!resp.ok) {
      agentBody.innerHTML = 'Server error.';
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(/\n\n/);
      buffer = parts.pop();

      for (const part of parts) {
        const match = part.match(/data:\s*(.*)/);
        if (!match) continue;

        const event = JSON.parse(match[1]);
        if (event.node === 'general_message' && !event.complete) {
          streamedText += event.content || '';
          agentBody.innerHTML = escapeHtml(streamedText);
          scrollToBottom();
        }
      }
    }
  } catch (err) {
    agentBody.innerHTML =
      '<span class="text-red-500">Connection interrupted</span>';
  } finally {
    controller = null;
  }
}

/* ---------- EVENTS ---------- */

chatForm.addEventListener('submit', startStream);
firstChatForm.addEventListener('submit', startStream);

firstMessageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    startStream(e);
  }
});

messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    startStream(e);
  }
});

const MAX_TEXTAREA_HEIGHT = 160; // px (≈ 4–5 lines like ChatGPT)

function autoExpandTextarea(el) {
  // reset height so scrollHeight is accurate
  el.style.height = 'auto';

  if (el.scrollHeight <= MAX_TEXTAREA_HEIGHT) {
    el.style.height = el.scrollHeight + 'px';
    el.style.overflowY = 'hidden';
  } else {
    el.style.height = MAX_TEXTAREA_HEIGHT + 'px';
    el.style.overflowY = 'auto';
  }
}

const textarea = document.getElementById('messageInput');
const textAreaFirst = document.getElementById('firstMessageInput');


textAreaFirst.addEventListener('input', () => {
  autoExpandTextarea(textAreaFirst);
});

// paste (important for large text)
textAreaFirst.addEventListener('paste', () => {
  // allow paste to complete before measuring height
  requestAnimationFrame(() => {
    autoExpandTextarea(textAreaFirst);
  });
});
// typing
textarea.addEventListener('input', () => {
  autoExpandTextarea(textarea);
});

// paste (important for large text)
textarea.addEventListener('paste', () => {
  // allow paste to complete before measuring height
  requestAnimationFrame(() => {
    autoExpandTextarea(textarea);
  });
});

function resetTextarea() {
  textarea.value = '';
  textarea.style.height = 'auto';
  textarea.style.overflowY = 'hidden';
}
