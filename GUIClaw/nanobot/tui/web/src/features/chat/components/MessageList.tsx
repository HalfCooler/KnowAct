import { useEffect, useRef } from "react";

import type { ChatMessage } from "../../../lib/api/client";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  messages: ChatMessage[];
  streamingContent?: string | null;
}

export function MessageList({ messages, streamingContent }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  if (messages.length === 0 && !streamingContent) {
    return (
      <div className="flex flex-col flex-1 items-center justify-center text-earth-muted text-sm">
        Start a conversation...
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 overflow-y-auto flex-1 p-4">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}
      {streamingContent && (
        <MessageBubble
          message={{ role: "assistant", content: streamingContent }}
          isStreaming
        />
      )}
      <div ref={bottomRef} />
    </div>
  );
}
