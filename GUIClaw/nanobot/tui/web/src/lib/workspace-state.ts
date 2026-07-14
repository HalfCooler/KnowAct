export type WorkspaceState = {
  sessionId: string | null;
  runId: string | null;
  panel: string | null;
};

function cleanValue(value: string | null | undefined) {
  if (!value) {
    return null;
  }
  return value.trim() || null;
}

export function readWorkspaceState(pathname: string, search: string): WorkspaceState {
  const searchParams = new URLSearchParams(search);
  const chatMatch = pathname.match(/^\/chat\/([^/?#]+)/);
  const pathSessionId = chatMatch?.[1] ? decodeURIComponent(chatMatch[1]) : null;

  return {
    sessionId: cleanValue(pathSessionId ?? searchParams.get("sessionId")),
    runId: cleanValue(searchParams.get("runId")),
    panel: cleanValue(searchParams.get("panel")),
  };
}

export function buildWorkspaceSearch(
  state: WorkspaceState,
  options: { includeSessionId?: boolean } = {},
): string {
  const { includeSessionId = true } = options;
  const searchParams = new URLSearchParams();

  if (includeSessionId && state.sessionId) {
    searchParams.set("sessionId", state.sessionId);
  }
  if (state.runId) {
    searchParams.set("runId", state.runId);
  }
  if (state.panel) {
    searchParams.set("panel", state.panel);
  }

  const next = searchParams.toString();
  return next ? `?${next}` : "";
}

export function buildChatHref(state: WorkspaceState): string {
  const path = state.sessionId ? `/chat/${encodeURIComponent(state.sessionId)}` : "/chat";
  return `${path}${buildWorkspaceSearch(state, { includeSessionId: false })}`;
}

export function buildOperationsHref(state: WorkspaceState): string {
  return `/operations${buildWorkspaceSearch(state)}`;
}
