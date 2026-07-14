import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { useQuery } from "@tanstack/react-query";

import {
  createChatSession,
  sendChatMessage,
  getChatSession,
  type ChatMessage,
} from "../../lib/api/client";
import { readWorkspaceState } from "../../lib/workspace-state";
import { MessageList } from "./components/MessageList";
import { MessageInput } from "./components/MessageInput";
import { SessionSidebar } from "./components/SessionSidebar";
import { useChatStream } from "./hooks/useChatStream";
import { useSessionManager } from "./hooks/useSessionManager";

export function ChatWorkspaceRoute() {
  const location = useLocation();
  const navigate = useNavigate();
  const workspaceState = readWorkspaceState(location.pathname, location.search);
  const sessionId = workspaceState.sessionId;

  const { sessions, addSession, updateSessionTitle } = useSessionManager();
  const [localMessages, setLocalMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);

  // Load session messages from backend
  const sessionQuery = useQuery({
    queryKey: ["chat-session", sessionId],
    queryFn: () => getChatSession(sessionId!),
    enabled: Boolean(sessionId),
    retry: false,
  });

  // Sync local messages when query data arrives
  useEffect(() => {
    if (sessionQuery.data) {
      setLocalMessages(sessionQuery.data.messages);
    }
  }, [sessionQuery.data]);

  // Reset local messages when session changes
  useEffect(() => {
    setLocalMessages([]);
  }, [sessionId]);

  // Stable callback for SSE assistant messages
  const onAssistantMessage = useCallback((msg: ChatMessage) => {
    setLocalMessages((prev) => [...prev, msg]);
  }, []);

  const { streamingContent } = useChatStream({ sessionId, onAssistantMessage });

  async function onSend(content: string) {
    if (!sessionId) return;

    setIsSending(true);

    // Optimistically append user message
    const userMessage: ChatMessage = { role: "user", content };
    setLocalMessages((prev) => [...prev, userMessage]);

    // Update title if this is the first user message
    const isFirstMessage = localMessages.length === 0;
    if (isFirstMessage) {
      updateSessionTitle(sessionId, content.slice(0, 40));
    }

    try {
      // Fire and forget — the SSE "assistant.final" event will deliver the reply
      void sendChatMessage(sessionId, content);
    } catch {
      setLocalMessages((prev) => [
        ...prev,
        { role: "assistant", content: "[Error] Failed to send message." },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function onNewSession() {
    try {
      const result = await createChatSession();
      const newId = result.session.session_id;
      addSession(newId, "New Chat");
      void navigate(`/chat/${newId}`);
    } catch {
      // Silently ignore — UI stays stable
    }
  }

  function onSelectSession(id: string) {
    void navigate(`/chat/${id}`);
  }

  return (
    <div className="-m-7 rounded-3xl overflow-hidden flex min-h-[600px] h-[calc(100vh-320px)]">
      <SessionSidebar
        sessions={sessions}
        activeSessionId={sessionId}
        onSelectSession={onSelectSession}
        onNewSession={onNewSession}
      />
      <div className="flex flex-col flex-1 min-w-0">
        {sessionId ? (
          <>
            <MessageList messages={localMessages} streamingContent={streamingContent} />
            <MessageInput onSend={onSend} disabled={isSending} />
          </>
        ) : (
          <div className="flex flex-col flex-1 items-center justify-center gap-4 text-earth-muted">
            <p className="text-sm">Select a session or start a new chat</p>
            <button
              onClick={onNewSession}
              className="rounded-xl bg-earth-accent text-white px-6 py-2.5 text-sm hover:opacity-90 font-medium"
            >
              New Chat
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
