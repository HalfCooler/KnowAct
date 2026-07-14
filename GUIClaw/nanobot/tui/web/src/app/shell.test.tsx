import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RouterProvider } from "react-router";

import { AppProviders } from "./providers";
import { createWorkspaceMemoryRouter } from "./router";

describe("WorkspaceShell", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();

    vi.stubGlobal(
      "fetch",
      vi.fn(async (input) => {
        const url = String(input);
        if (url.includes("/chat/sessions/session-123")) {
          return new Response(
            JSON.stringify({
              session: {
                session_id: "session-123",
                session_key: "tui:session-123",
                metadata: {},
                message_count: 2,
              },
              messages: [
                { role: "user", content: "hello" },
                { role: "assistant", content: "hi" },
              ],
            }),
            { status: 200 },
          );
        }

        return new Response(
          JSON.stringify({
            status: "idle",
            channel_runtime_booted: false,
            agent_loop_booted: false,
            task_launch_available: true,
            session_stats: { total: 1, active: 1, most_recent_session_id: "session-123" },
            active_runs: [],
            recent_failures: [],
          }),
          { status: 200 },
        );
      }),
    );

    class MockEventSource {
      close = vi.fn();
      addEventListener = vi.fn();
      constructor(_url: string) {}
    }

    vi.stubGlobal("EventSource", MockEventSource);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the active session visible when navigating between chat and operations", async () => {
    const router = createWorkspaceMemoryRouter([
      "/chat/session-123?runId=run-9&panel=trace",
    ]);

    render(
      <AppProviders>
        <RouterProvider router={router} />
      </AppProviders>,
    );

    screen.getByRole("heading", { name: "Nanobot Workspace" });

    fireEvent.click(screen.getByRole("link", { name: "Operations" }));

    screen.getByText("Operations console");

    fireEvent.click(screen.getByRole("link", { name: "Chat" }));

    screen.getByRole("link", { name: "Chat" });
  });
});
