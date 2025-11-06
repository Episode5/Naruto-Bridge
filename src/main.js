// === Naruto Bridge: Core Framework (Interface Jutsu) ===

const fs = require("fs");
const { readScroll, writeScroll } = require("./readScroll");
const { forwardTransform, reverseTransform } = require("./transformScroll");
const { describe } = require("./analyzeScroll");
const { fetchScroll } = require("./FetchScrolls");
const { summarizeData } = require("./summarizeScroll");

const arg = process.argv[2];
if (!arg) {
  console.log("Usage:");
  console.log("  node src/main.js <path-or-url>");
  console.log("Example:");
  console.log("  node src/main.js schemas/ninja.json");
  console.log("  node src/main.js https://jsonplaceholder.typicode.com/users/1");
  process.exit(0);
}

// --- Helper to detect if the argument is a URL ---
function isURL(str) {
  return /^https?:\/\//i.test(str);
}

// --- Unified logic ---
async function runBridge(source) {
  let data;

  try {
    if (isURL(source)) {
      console.log("\n=== Fetching remote scroll ===");
      data = await fetchScroll(source);
    } else {
      console.log("\n=== Reading local scroll ===");
      data = readScroll(source);
    }
  } catch (err) {
    console.error("Error reading scroll:", err.message);
    process.exit(1);
  }

  if (!data) {
    console.error("No data found.");
    process.exit(1);
  }

  // --- Describe structure ---
  console.log("\n-- Scroll Structure --");
  describe(data);
// --- Generate summary ---
const summary = summarizeData(data);
console.log("\n-- Data Summary --");
console.log(
  `Keys: ${summary.keyCount}, Arrays: ${summary.arrayCount}, Objects: ${summary.objectCount}`
);
console.log(
  `Numeric fields: ${summary.numberCount}, Average numeric value: ${summary.avg}`
);

  // --- Transform (forward & reverse) ---
  const model = forwardTransform(data);
  console.log("\n-- Internal Model --");
  console.log(model);

  const newJson = reverseTransform(model);
  const outPath = source.startsWith("http")
    ? "schemas/remote_output.json"
    : source.replace(".json", "_output.json");

  writeScroll(outPath, newJson);
  console.log("\nPipeline complete →", outPath);
}

// Run it
runBridge(arg);


