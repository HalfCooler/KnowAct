import {
  createBrowserRouter,
  createMemoryRouter,
} from "react-router";

import { ChatWorkspaceRoute } from "../features/chat/ChatWorkspaceRoute";
import { OperationsWorkspaceRoute } from "../features/operations/OperationsWorkspaceRoute";
import { WorkspaceShell } from "./shell";

export function createWorkspaceRouter() {
  return createBrowserRouter(buildWorkspaceRoutes());
}

export function createWorkspaceMemoryRouter(initialEntries: string[]) {
  return createMemoryRouter(buildWorkspaceRoutes(), { initialEntries });
}

function buildWorkspaceRoutes() {
  return [
    {
      path: "/",
      element: <WorkspaceShell />,
      children: [
        {
          index: true,
          element: <ChatWorkspaceRoute />,
        },
        {
          path: "chat",
          element: <ChatWorkspaceRoute />,
        },
        {
          path: "chat/:sessionId",
          element: <ChatWorkspaceRoute />,
        },
        {
          path: "operations",
          element: <OperationsWorkspaceRoute />,
        },
      ],
    },
  ];
}
