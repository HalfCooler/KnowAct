import { useQuery } from "@tanstack/react-query";
import { useLocation } from "react-router";

import { getRuntimeInspection } from "../../lib/api/client";
import { readWorkspaceState } from "../../lib/workspace-state";

export function OperationsWorkspaceRoute() {
  const location = useLocation();
  const workspaceState = readWorkspaceState(location.pathname, location.search);
  const runtimeQuery = useQuery({
    queryKey: ["runtime-overview"],
    queryFn: () => getRuntimeInspection(),
    retry: false,
  });

  return (
    <div style={{ display: "grid", gap: "16px" }}>
      <div>
        <h2 style={{ margin: "0 0 8px", fontSize: "1.55rem" }}>Operations console</h2>
      </div>
      <dl
        style={{
          display: "grid",
          gap: "12px",
          margin: 0,
          padding: 0,
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        }}
      >
        {[
          ["Linked session", workspaceState.sessionId ?? "No chat session linked yet."],
          ["Selected run", workspaceState.runId ?? "Choose a run once the runtime view is populated."],
          ["Panel", workspaceState.panel ?? "overview"],
          [
            "Runtime status",
            runtimeQuery.data
              ? `${runtimeQuery.data.status} (${runtimeQuery.data.active_runs.length} active runs)`
              : runtimeQuery.isError
                ? "Runtime fetch pending backend connectivity"
                : "Loading runtime overview",
          ],
        ].map(([label, value]) => (
          <div
            key={label}
            style={{
              padding: "16px",
              borderRadius: "18px",
              background: "rgba(246, 248, 245, 0.95)",
              border: "1px solid rgba(94, 109, 82, 0.14)",
            }}
          >
            <dt style={{ fontSize: "0.82rem", textTransform: "uppercase", letterSpacing: "0.08em" }}>
              {label}
            </dt>
            <dd style={{ margin: "8px 0 0", lineHeight: 1.5 }}>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
