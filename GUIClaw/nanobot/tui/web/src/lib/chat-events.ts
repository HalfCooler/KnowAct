import { resolveApiUrl, type ApiEnv } from "./api/client";

export type ChatEvent = {
  id: string;
  type: string;
  session_id: string;
  run_id?: string | null;
  payload: Record<string, unknown>;
};

export function connectChatEvents(
  sessionId: string,
  onEvent: (event: ChatEvent) => void,
  env?: ApiEnv,
) {
  if (typeof window === "undefined" || typeof window.EventSource === "undefined") {
    return () => undefined;
  }

  const stream = new window.EventSource(resolveApiUrl(`/chat/sessions/${sessionId}/events`, env));
  const forward = (message: MessageEvent<string>) => {
    onEvent(JSON.parse(message.data) as ChatEvent);
  };

  for (const eventName of ["message.accepted", "progress", "assistant.final", "error", "complete"]) {
    stream.addEventListener(eventName, forward as EventListener);
  }

  return () => {
    stream.close();
  };
}
