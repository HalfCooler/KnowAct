import { useState } from "react";

const STORAGE_KEY = "nanobot-chat-sessions";

interface StoredSession {
  id: string;
  title: string;
  updatedAt: string;
}

function readFromStorage(): StoredSession[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed as StoredSession[];
  } catch {
    return [];
  }
}

function writeToStorage(sessions: StoredSession[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

function sortByUpdatedAt(sessions: StoredSession[]): StoredSession[] {
  return [...sessions].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );
}

interface UseSessionManagerResult {
  sessions: StoredSession[];
  addSession: (id: string, title?: string) => void;
  updateSessionTitle: (id: string, title: string) => void;
  removeSession: (id: string) => void;
}

export function useSessionManager(): UseSessionManagerResult {
  const [sessions, setSessions] = useState<StoredSession[]>(() =>
    sortByUpdatedAt(readFromStorage()),
  );

  function addSession(id: string, title = "New Chat") {
    setSessions((prev) => {
      // Avoid duplicates
      if (prev.some((s) => s.id === id)) return prev;
      const next = sortByUpdatedAt([
        { id, title, updatedAt: new Date().toISOString() },
        ...prev,
      ]);
      writeToStorage(next);
      return next;
    });
  }

  function updateSessionTitle(id: string, title: string) {
    setSessions((prev) => {
      const next = sortByUpdatedAt(
        prev.map((s) =>
          s.id === id ? { ...s, title, updatedAt: new Date().toISOString() } : s,
        ),
      );
      writeToStorage(next);
      return next;
    });
  }

  function removeSession(id: string) {
    setSessions((prev) => {
      const next = prev.filter((s) => s.id !== id);
      writeToStorage(next);
      return next;
    });
  }

  return { sessions, addSession, updateSessionTitle, removeSession };
}
