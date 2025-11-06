// === Naruto Bridge: Core Framework (Cross-System Flow) ===

const fs = require("fs");
const pathLib = require("path");
const { readScroll, writeScroll } = require("./readScroll");
const { forwardTransform, reverseTransform } = require("./transformScroll");
const { describe } = require("./analyzeScroll");
const { fetchScroll } = require("./FetchScrolls");
const { summarizeData } = require("./summarizeScroll");

const args = process.argv.slice(2);

if (args.length === 0) {
  console.log("Usage:");
  console.log("  naruto-bridge <path-or-url> [more paths/urls]");
  console.log("Examples:");
  console.log("  naruto-bridge schemas/ninja.json");
  console.log("  naruto-bridge https://jsonplaceholder.typicode.com/users/1");
  process.exit(0);
}

function isURL(str) {
  return /^https?:\/\//i.test(str);
}

async function processScroll(source, index, total) {
  console.log(`\n=== Processing Scroll ${index + 1} of ${total} ===`);

  let data;
  try {
    // --- Normalize paths for Windows/Unix compatibility ---
    if (!isURL(source)) {
      source = pathLib.normalize(source);
    }

    if (isURL(source)) {
      console.log("Fetching remote scroll:", source);
      data = await fetchScroll(source);
    } else {
      console.log("Reading local scroll:", source);
      data = readScroll(source);
    }
  } catch (err) {
    console.error("Error reading scroll:", err.message);
    return;
  }

  if (!data) {
    console.error("No data found for", source);
    return;
  }

  console.log("\n-- Scroll Structure --");
  describe(data);

  const summary = summarizeData(data);
  console.log("\n-- Data Summary --");
  console.log(
    `Keys: ${summary.keyCount}, Arrays: ${summary.arrayCount}, Objects: ${summary.objectCount}`
  );
  console.log(
    `Numeric fields: ${summary.numberCount}, Average numeric value: ${summary.avg}`
  );

  const model = forwardTransform(data);
  const newJson = reverseTransform(model);

  const outPath = isURL(source)
    ? `schemas/remote_output_${index + 1}.json`
    : source.replace(".json", `_output_${index + 1}.json`);

  writeScroll(outPath, newJson);
}

async function run() {
  console.log("=== Naruto Bridge Multi-Scroll Mode ===");
  for (let i = 0; i < args.length; i++) {
    await processScroll(args[i], i, args.length);
  }
  console.log("\nAll scrolls complete.\n");
}

run();

