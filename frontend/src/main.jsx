import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function formatDate(dateStr) {
  if (!dateStr) return "Unknown date";
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    }).format(d);
  } catch {
    return dateStr;
  }
}

function parseSender(senderStr) {
  if (!senderStr) return { name: "Unknown sender", address: "" };
  const match = senderStr.match(/^(.*?)\s*<([^>]+)>$/);
  if (match) {
    const name = match[1].replace(/^["']|["']$/g, "").trim();
    const address = match[2].trim();
    return { name: name || address, address };
  }
  return { name: senderStr.trim(), address: "" };
}

function getGmailUrl(messageId) {
  if (!messageId) return "";
  return `https://mail.google.com/mail/u/0/#inbox/${encodeURIComponent(messageId)}`;
}

const SAMPLE_QUESTIONS = [
  "What is the amount in my latest bank statement email?",
  "When was my last email from Instagram?",
  "Show me recent flight confirmations",
  "Did I receive any security alerts?",
];

// ---------------------------------------------------------------------------
// EmailBodyText — renders cleaned plain-text email bodies safely.
// Does NOT use dangerouslySetInnerHTML.
// Does NOT treat the content as Markdown (it's email text, not Markdown).
// Long URLs are CSS-wrapped to prevent horizontal overflow.
// ---------------------------------------------------------------------------
function EmailBodyText({ text }) {
  if (!text) return <p className="empty-body-text">No preview available.</p>;

  // Split on lines and render each, turning bare URLs into safe <a> tags.
  const urlPattern = /https?:\/\/[^\s"'<>()]+/g;

  function renderLine(line, lineIdx) {
    const parts = [];
    let lastIndex = 0;
    let match;
    urlPattern.lastIndex = 0;
    while ((match = urlPattern.exec(line)) !== null) {
      if (match.index > lastIndex) {
        parts.push(line.slice(lastIndex, match.index));
      }
      const url = match[0].replace(/[.,;:!?)]+$/, ""); // strip trailing punctuation
      parts.push(
        <a
          key={`url-${lineIdx}-${match.index}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="email-inline-link"
        >
          {url}
        </a>
      );
      lastIndex = match.index + url.length;
    }
    if (lastIndex < line.length) {
      parts.push(line.slice(lastIndex));
    }
    return parts;
  }

  const lines = text.split("\n");
  const nodes = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      // Blank line → paragraph break
      nodes.push(<br key={`br-${i}`} />);
    } else {
      nodes.push(
        <span key={`line-${i}`} className="email-body-line">
          {renderLine(line, i)}
        </span>
      );
      // Add a line break after non-blank lines (unless next is blank)
      if (i + 1 < lines.length && lines[i + 1].trim()) {
        nodes.push(<br key={`lbr-${i}`} />);
      }
    }
    i++;
  }

  return <div className="email-body-plain">{nodes}</div>;
}

function App() {
  const [connected, setConnected] = useState(false);
  const [emails, setEmails] = useState([]);
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [expandedEmails, setExpandedEmails] = useState({});
  const [expandedSources, setExpandedSources] = useState({});
  const chatLogRef = useRef(null);
  const textareaRef = useRef(null);

  async function refresh() {
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/auth/status`, { credentials: "include" });
      const status = await response.json();
      setConnected(status.connected);
      if (status.connected) {
        const emailResponse = await fetch(`${API_URL}/api/emails?limit=10`, { credentials: "include" });
        const payload = await emailResponse.json();
        if (!emailResponse.ok) throw new Error(payload.detail || "Unable to fetch emails.");
        setEmails(payload.emails || []);
      }
    } catch (reason) {
      setError(reason.message === "Failed to fetch" ? "Unable to connect to backend server." : reason.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    const chatLog = chatLogRef.current;
    if (chatLog) {
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  }, [messages, sending]);

  function connectGmail() {
    window.location.href = `${API_URL}/api/auth/login`;
  }

  async function logout() {
    setError("");
    try {
      const response = await fetch(`${API_URL}/api/auth/logout`, {
        method: "POST",
        credentials: "include",
      });
      if (!response.ok) throw new Error("Unable to disconnect Gmail.");
      setConnected(false);
      setEmails([]);
      setMessages([]);
      setQuery("");
    } catch (reason) {
      setError(reason.message === "Failed to fetch" ? "Unable to disconnect Gmail. Please try again." : reason.message);
    }
  }

  function handleQueryKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  }

  function toggleEmailExpand(id) {
    setExpandedEmails((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  }

  function toggleSourceExpand(id) {
    setExpandedSources((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  }

  async function submitQuery(queryString) {
    const cleanQuery = queryString.trim();
    if (!cleanQuery || sending) return;
    setError("");
    setMessages((current) => [...current, { role: "user", text: cleanQuery }]);
    setQuery("");
    setSending(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: cleanQuery }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "Something went wrong while searching your Gmail. Please try again.");
      }
      setMessages((current) => [...current, { role: "assistant", ...payload }]);
    } catch (reason) {
      setError(
        reason.message === "Failed to fetch"
          ? "Something went wrong while searching your Gmail. Please try again."
          : reason.message
      );
    } finally {
      setSending(false);
    }
  }

  function handleFormSubmit(event) {
    event.preventDefault();
    submitQuery(query);
  }

  return (
    <div className="app-viewport">
      <main className="shell">
        <header className="site-header">
          <div className="header-brand">
            <span className="eyebrow">AI GMAIL KNOWLEDGE ASSISTANT</span>
            <h1>Ask what your inbox knows.</h1>
            <p className="intro">
              Answers stay grounded strictly in your connected Gmail, with verified source emails shown beneath each response.
            </p>
          </div>
          <div className="auth-row">
            {!connected ? (
              <button className="btn-primary" onClick={connectGmail} aria-label="Sign in with Google">
                Sign in with Google
              </button>
            ) : (
              <div className="auth-status-box">
                <span className="status-badge connected-badge">
                  <span className="status-dot"></span> Gmail connected
                </span>
                <button className="btn-secondary logout" onClick={logout} aria-label="Log out of Gmail">
                  Log out
                </button>
              </div>
            )}
          </div>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </header>

        <section className="chat-section" aria-label="Gmail chat">
          <div className="chat-log" aria-live="polite" ref={chatLogRef}>
            {messages.length === 0 && (
              <div className="empty-chat-state">
                <p className="empty-title">Ask your Gmail anything</p>
                <p className="empty-subtitle">Try asking about recent emails, specific senders, invoices, or decisions:</p>
                <div className="suggestion-chips">
                  {SAMPLE_QUESTIONS.map((suggestion, i) => (
                    <button
                      key={i}
                      type="button"
                      className="suggestion-chip"
                      disabled={!connected || sending}
                      onClick={() => submitQuery(suggestion)}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message, index) => {
              const isUser = message.role === "user";
              return (
                <article
                  className={`chat-bubble-container ${isUser ? "user-bubble-container" : "assistant-bubble-container"}`}
                  key={`${message.role}-${index}`}
                >
                  <div className={`chat-bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
                    <div className="message-header">
                      <span className="message-author">{isUser ? "You" : "Gmail Assistant"}</span>
                    </div>

                    <div className="message-content">
                      {isUser ? (
                        <p>{message.text}</p>
                      ) : (
                        // AI responses use ReactMarkdown — they are generated, structured text.
                        <ReactMarkdown
                          components={{
                            a: ({ node, ...props }) => (
                              <a {...props} target="_blank" rel="noopener noreferrer" />
                            ),
                          }}
                        >
                          {message.answer || message.text || ""}
                        </ReactMarkdown>
                      )}
                    </div>

                    {!isUser && message.sources && message.sources.length > 0 && (
                      <div className="sources-container">
                        <div className="sources-header">
                          <span className="sources-title">Verified Sources ({message.sources.length})</span>
                        </div>
                        <div className="sources-list">
                          {message.sources.map((source, sIdx) => {
                            const sourceId = source.message_id || `source-${sIdx}`;
                            const parsedSender = parseSender(source.sender);
                            const isExpanded = !!expandedSources[sourceId];
                            const sourceGmailUrl = source.gmail_url || getGmailUrl(source.message_id);
                            // snippet is already display-cleaned by the backend
                            const snippet = source.snippet || "";

                            return (
                              <div className="source-card" key={sourceId}>
                                <div className="source-card-header">
                                  <h4 className="source-subject">{source.subject || "No subject"}</h4>
                                  <time className="source-date">{formatDate(source.date)}</time>
                                </div>
                                <div className="source-sender-line">
                                  <span className="sender-name">{parsedSender.name}</span>
                                  {parsedSender.address && (
                                    <span className="sender-address">&lt;{parsedSender.address}&gt;</span>
                                  )}
                                </div>
                                {snippet && (
                                  <div className="source-snippet-box">
                                    <p className="source-snippet-text">
                                      {isExpanded
                                        ? snippet
                                        : snippet.length > 220
                                        ? `${snippet.slice(0, 220)}…`
                                        : snippet}
                                    </p>
                                    {snippet.length > 220 && (
                                      <button
                                        type="button"
                                        className="btn-link"
                                        onClick={() => toggleSourceExpand(sourceId)}
                                      >
                                        {isExpanded ? "Show less ↑" : "Show more ↓"}
                                      </button>
                                    )}
                                  </div>
                                )}
                                {sourceGmailUrl && (
                                  <div className="source-footer">
                                    <a
                                      href={sourceGmailUrl}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      className="open-gmail-link"
                                      id={`open-gmail-${source.message_id}`}
                                    >
                                      Open in Gmail →
                                    </a>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                </article>
              );
            })}

            {sending && (
              <div className="chat-bubble-container assistant-bubble-container">
                <div className="chat-bubble assistant-bubble loading-bubble">
                  <span className="message-author">Gmail Assistant</span>
                  <div className="loading-indicator">
                    <span className="pulse-dot"></span>
                    <span className="pulse-dot"></span>
                    <span className="pulse-dot"></span>
                    <span className="loading-text">Searching your Gmail…</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <form className="composer-form" onSubmit={handleFormSubmit}>
            <label htmlFor="query" className="composer-label">
              Ask your inbox
            </label>
            <div className="composer-input-wrapper">
              <textarea
                ref={textareaRef}
                id="query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleQueryKeyDown}
                placeholder={connected ? "Ask a question about your emails, senders, deadlines, receipts..." : "Connect Gmail to start asking questions..."}
                disabled={!connected || sending}
                rows={2}
                className="composer-textarea"
                aria-label="Ask your inbox"
              />
              <button
                type="submit"
                disabled={!connected || sending || !query.trim()}
                className="btn-send"
                aria-label="Send query"
              >
                {sending ? "Searching…" : "Send"}
              </button>
            </div>
            <div className="composer-hint">Press Enter to send, Shift + Enter for new line</div>
          </form>
        </section>

        {connected && emails.length > 0 && (
          <section className="recent-emails-section" aria-label="Recent emails preview">
            <div className="section-title-row">
              <h2>Recent Inbox Emails</h2>
              <span className="section-subtitle">Showing latest {emails.length} normalized messages</span>
            </div>
            <div className="email-card-grid">
              {emails.map((email) => {
                const id = email.message_id;
                const parsedSender = parseSender(email.sender);
                const isExpanded = !!expandedEmails[id];
                // Prefer display_body (clean), fall back to body
                const rawBody = email.display_body || email.body || email.snippet || "";
                const isLong = rawBody.length > 400;
                const displayText = isExpanded ? rawBody : (isLong ? `${rawBody.slice(0, 400)}…` : rawBody);
                const emailGmailUrl = getGmailUrl(email.message_id);

                return (
                  <article className="email-card" key={id}>
                    <header className="email-card-header">
                      <div className="email-meta-bar">
                        <time className="email-date">{formatDate(email.date)}</time>
                        {email.labels && email.labels.length > 0 && (
                          <div className="email-labels">
                            {email.labels.slice(0, 3).map((lbl, idx) => (
                              <span key={idx} className="email-label-badge">{lbl}</span>
                            ))}
                          </div>
                        )}
                      </div>
                      <h3 className="email-subject-heading">{email.subject || "No subject"}</h3>
                      <div className="email-sender-info">
                        <span className="sender-display-name">{parsedSender.name}</span>
                        {parsedSender.address && (
                          <span className="sender-email-address">&lt;{parsedSender.address}&gt;</span>
                        )}
                      </div>
                    </header>

                    <div className="email-card-divider" />

                    <div className="email-card-body">
                      {/* Email body rendered as plain text — NOT as Markdown, NOT as raw HTML */}
                      <EmailBodyText text={displayText} />
                      {isLong && (
                        <button
                          type="button"
                          className="btn-toggle-expand"
                          onClick={() => toggleEmailExpand(id)}
                        >
                          {isExpanded ? "Show less ↑" : "Show more ↓"}
                        </button>
                      )}
                    </div>

                    <footer className="email-card-footer">
                      <span className="email-thread-id">Thread: {email.thread_id ? email.thread_id.slice(0, 16) : id.slice(0, 16)}</span>
                      {emailGmailUrl && (
                        <a
                          href={emailGmailUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="open-gmail-link"
                          id={`open-gmail-recent-${id}`}
                        >
                          Open in Gmail →
                        </a>
                      )}
                    </footer>
                  </article>
                );
              })}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);
