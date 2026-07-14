import { describe, expect, it } from "vitest";

import { resolveApiBase, resolveApiPath } from "./client";

describe("resolveApiBase", () => {
  it("uses the /api proxy in development", () => {
    expect(resolveApiBase({ DEV: true })).toBe("/api");
    expect(resolveApiPath("/chat/sessions", { DEV: true })).toBe("/api/chat/sessions");
  });

  it("uses same-origin requests for the served frontend by default", () => {
    expect(resolveApiBase({ DEV: false })).toBe("");
    expect(resolveApiPath("/runtime", { DEV: false })).toBe("/runtime");
  });

  it("normalizes a configured packaged API base", () => {
    expect(resolveApiBase({ DEV: false, VITE_NANOBOT_API_BASE: "http://localhost:18791/" })).toBe(
      "http://localhost:18791",
    );
  });
});
