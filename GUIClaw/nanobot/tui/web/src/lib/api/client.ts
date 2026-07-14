export type ApiEnv = {
  DEV?: boolean;
  VITE_NANOBOT_API_BASE?: string;
};

export type ChatMessage = {
  role: string;
  content: string;
  timestamp?: string | null;
};

export type ChatSessionSummary = {
  session_id: string;
  session_key: string;
  created_at?: string | null;
  updated_at?: string | null;
  metadata: Record<string, unknown>;
  message_count: number;
};

export type ChatSessionResponse = {
  session: ChatSessionSummary;
  messages: ChatMessage[];
};

export type RuntimeSessionStats = {
  total: number;
  active: number;
  most_recent_session_id?: string | null;
};

export type RuntimeRunSummary = {
  run_id: string;
  task_kind: string;
  status: string;
  summary?: string | null;
  steps_taken: number;
  started_at?: string | null;
  finished_at?: string | null;
};

export type RuntimeInspectionResponse = {
  status: string;
  channel_runtime_booted: boolean;
  agent_loop_booted: boolean;
  task_launch_available: boolean;
  session_stats: RuntimeSessionStats;
  active_runs: RuntimeRunSummary[];
  recent_failures: RuntimeRunSummary[];
};

function trimTrailingSlash(value: string) {
  return value.replace(/\/+$/, "");
}

export function resolveApiBase(env: ApiEnv = import.meta.env): string {
  if (env.DEV) {
    return "/api";
  }

  return trimTrailingSlash(env.VITE_NANOBOT_API_BASE ?? "");
}

export function resolveApiPath(path: string, env?: ApiEnv): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${resolveApiBase(env)}${normalizedPath}`;
}

export function resolveApiUrl(path: string, env?: ApiEnv): string {
  return new URL(resolveApiPath(path, env), window.location.origin).toString();
}

export async function fetchJson<T>(path: string, init?: RequestInit, env?: ApiEnv): Promise<T> {
  const response = await fetch(resolveApiUrl(path, env), init);

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function getChatSession(sessionId: string, env?: ApiEnv) {
  return fetchJson<ChatSessionResponse>(`/chat/sessions/${sessionId}`, undefined, env);
}

export function getRuntimeInspection(env?: ApiEnv) {
  return fetchJson<RuntimeInspectionResponse>("/runtime", undefined, env);
}

export type ChatCreateSessionResponse = { session: ChatSessionSummary; messages: ChatMessage[] };
export type ChatMessageResponse = { session: ChatSessionSummary; reply: ChatMessage };

export function createChatSession(env?: ApiEnv) {
  return fetchJson<ChatCreateSessionResponse>("/chat/sessions", { method: "POST" }, env);
}

export function sendChatMessage(sessionId: string, content: string, env?: ApiEnv) {
  return fetchJson<ChatMessageResponse>(`/chat/sessions/${sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content }),
  }, env);
}
