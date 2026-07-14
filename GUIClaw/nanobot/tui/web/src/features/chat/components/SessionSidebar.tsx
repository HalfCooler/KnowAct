interface SessionItem {
  id: string;
  title: string;
  updatedAt?: string;
}

interface SessionSidebarProps {
  sessions: SessionItem[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
}

function relativeTime(isoString?: string): string {
  if (!isoString) return "";
  const diff = Date.now() - new Date(isoString).getTime();
  const minutes = Math.floor(diff / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function SessionSidebar({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
}: SessionSidebarProps) {
  return (
    <div className="w-64 border-r border-earth-border flex flex-col bg-earth-card">
      <div className="p-3">
        <button
          onClick={onNewSession}
          className="w-full rounded-xl bg-earth-accent text-white py-2.5 text-center text-sm hover:opacity-90 font-medium"
        >
          New Chat
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {sessions.map((session) => (
          <button
            key={session.id}
            onClick={() => onSelectSession(session.id)}
            className={`w-full text-left px-3 py-2.5 mx-2 my-0.5 rounded-lg cursor-pointer text-sm truncate hover:bg-earth-badge transition-colors ${
              session.id === activeSessionId
                ? "bg-earth-badge font-medium"
                : ""
            }`}
            style={{ width: "calc(100% - 1rem)" }}
          >
            <span className="block truncate">{session.title}</span>
            {session.updatedAt && (
              <span className="block text-xs text-earth-muted mt-0.5">
                {relativeTime(session.updatedAt)}
              </span>
            )}
          </button>
        ))}
        {sessions.length === 0 && (
          <p className="px-4 py-3 text-xs text-earth-muted">No sessions yet</p>
        )}
      </div>
    </div>
  );
}
