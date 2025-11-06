const path = __dirname + "/../schemas/ninja.json";
const { readScroll, writeScroll } = require("./readScroll");
const { forwardTransform, reverseTransform } = require("./transformScroll");
const { describe } = require("./analyzeScroll");

console.log("=== Naruto Bridge: Core Framework ===");

const raw = readScroll(path);
if (!raw) process.exit(1);

// Inspect
console.log("\n-- Scroll Structure --");
describe(raw);

// Transform forward & back
const model = forwardTransform(raw);
const newJson = reverseTransform(model);

writeScroll(__dirname + "/../schemas/ninja_framework_output.json", newJson);

console.log("\nPipeline complete.");

