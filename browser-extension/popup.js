"use strict";

const fields = ["name", "url", "token"];
const status = document.getElementById("status");

async function restore() {
  const saved = await chrome.storage.local.get(["config", "connectionStatus"]);
  const defaults = {name: navigator.userAgent.includes("Edg/") ? "Edge" : "Chrome", url: "http://127.0.0.1:8766", token: ""};
  for (const name of fields) document.getElementById(name).value = saved.config?.[name] || defaults[name];
  status.textContent = saved.connectionStatus || "Start the CLI, then paste its bridge address and token.";
}

document.getElementById("connection").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.getElementById("connect");
  button.disabled = true;
  status.textContent = "Connecting…";
  try {
    const config = Object.fromEntries(fields.map(name => [name, document.getElementById(name).value.trim()]));
    const result = await chrome.runtime.sendMessage({action: "connect", config});
    status.textContent = result.ok ? "Connected. In the CLI, run browsers and select this browser." : result.error;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

document.getElementById("disconnect").addEventListener("click", async () => {
  try {
    await chrome.runtime.sendMessage({action: "disconnect"});
    status.textContent = "Disconnected. Your tabs remain open.";
  } catch (error) {
    status.textContent = error.message;
  }
});

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.connectionStatus) status.textContent = changes.connectionStatus.newValue;
});
restore();
