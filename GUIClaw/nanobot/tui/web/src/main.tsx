import "./index.css";
import { StrictMode } from "react";
import ReactDOM from "react-dom/client";

import { RouterProvider } from "react-router";

import { AppProviders } from "./app/providers";
import { createWorkspaceRouter } from "./app/router";

const root = document.getElementById("root");

if (!root) {
  throw new Error("Missing root element");
}

const router = createWorkspaceRouter();

ReactDOM.createRoot(root).render(
  <StrictMode>
    <AppProviders>
      <RouterProvider router={router} />
    </AppProviders>
  </StrictMode>,
);
