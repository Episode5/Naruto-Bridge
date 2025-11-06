#!/usr/bin/env node

// === Naruto Bridge CLI ===

const path = require("path");
const { spawn } = require("child_process");

// Get all command-line arguments (paths or URLs)
const args = process.argv.slice(2);

if (args.length === 0) {
  console.log("Usage:");
  console.log("  naruto-bridge <path-or-url> [more paths/urls]");
  process.exit(0);
}

// Execute your main framework file
const mainPath = path.join(__dirname, "main.js");

const child = spawn("node", [mainPath, ...args], { stdio: "inherit" });

child.on("exit", (code) => {
  process.exit(code);
});

