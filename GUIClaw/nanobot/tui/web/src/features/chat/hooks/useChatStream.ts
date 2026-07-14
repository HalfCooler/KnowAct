import { useEffect, useState } from "react";

import type { ChatMessage } from "../../../lib/api/client";
import { connectChatEvents } from "../../../lib/chat-events";

interface UseChatStreamOptions {
  sessionId: string | null;
  onAssistantMessage: (msg: ChatMessage) => void;
}

interface UseChatStreamResult {
  streamingContent: string | null;
}

export function useChatStream({
  sessionId,
  onAssistantMessage,
}: UseChatStreamOptions): UseChatStreamResult {
  const [streamingContent, setStreamingContent] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) {
      return () => undefined;
    }

    const disconnect = connectChatEvents(sessionId, (event) => {
      switch (event.type) {
        case "message.accepted":
          // User message acknowledged — no action needed
          break;

        case "progress":
          // Delta chunk — concatenate to build up partial response
          setStreamingContent((prev) => (prev ?? "") + (event.payload.content as string));
          break;

        case "assistant.final":
          onAssistantMessage({
            role: "assistant",
            content: event.payload.content as string,
          });
          setStreamingContent(null);
          break;

        case "error":
          onAssistantMessage({
            role: "assistant",
            content: "[Error] " + (event.payload.message as string),
          });
          setStreamingContent(null);
          break;

        case "complete":
          setStreamingContent(null);
          break;

        default:
          break;
      }
    });

    return disconnect;
  }, [sessionId, onAssistantMessage]);

  return { streamingContent };
}
