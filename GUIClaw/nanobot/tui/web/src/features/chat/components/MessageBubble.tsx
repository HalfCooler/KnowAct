import type { ChatMessage } from "../../../lib/api/client";

interface MessageBubbleProps {
  message: ChatMessage;
  isStreaming?: boolean;
}

export function MessageBubble({ message, isStreaming = false }: MessageBubbleProps) {
  const isUser = message.role === "user";

  const timestamp = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString()
    : null;

  return (
    <div className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}>
      <div
        className={
          isUser
            ? "bg-user-bubble text-user-bubble-text rounded-2xl px-4 py-3 max-w-[80%] ml-auto"
            : "bg-assistant-bubble border border-earth-border rounded-2xl px-4 py-3 max-w-[80%]"
        }
      >
        <p className="m-0 whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
        {isStreaming && (
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-current ml-1 animate-pulse" />
        )}
      </div>
      {timestamp && (
        <span className="text-xs text-earth-muted mt-1 px-1">{timestamp}</span>
      )}
    </div>
  );
}
