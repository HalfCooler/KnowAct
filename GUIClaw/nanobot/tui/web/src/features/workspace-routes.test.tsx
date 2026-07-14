import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RouterProvider } from "react-router";

import { AppProviders } from "../app/providers";
import { createWorkspaceMemoryRouter } from "../app/router";

const fetchMock = vi.fn<typeof fetch>();
const eventSourceInstances: Array<{ url: string; close: ReturnType<typeof vi.fn> }> = [];

class MockEventSource {
  readonly close = vi.fn();
  readonly addEventListener = vi.fn();
  constructor(public readonly url: string) {
    eventSourceInstances.push({ url, close: this.close });
  }
}

describe("workspace routes", () => {
  beforeEach(() => {
    Element.prototype.scrollIntoView = vi.fn();

    fetchMock.mockImplementation(async (input) => {
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
    });

    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("EventSource", MockEventSource);
  });

  afterEach(() => {
    eventSourceInstances.length = 0;
    vi.unstubAllGlobals();
    fetchMock.mockReset();
  });

  it("uses the typed client contract for chat and operations routes", async () => {
    const router = createWorkspaceMemoryRouter([
      "/chat/session-123?runId=run-9&panel=logs",
    ]);

    render(
      <AppProviders>
        <RouterProvider router={router} />
      </AppProviders>,
    );

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalled();
    });

    expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/chat/sessions/session-123"))).toBe(
      true,
    );
    expect(eventSourceInstances[0]?.url).toContain("/chat/sessions/session-123/events");
    await screen.findByText("hello");

    await router.navigate("/operations?sessionId=session-123&runId=run-9&panel=logs");

    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([input]) => String(input).includes("/runtime"))).toBe(true);
    });

    await screen.findByText("idle (0 active runs)");
    screen.getAllByText("session-123");
    screen.getAllByText("run-9");
  });
});
