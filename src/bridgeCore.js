const fs = require("fs");
const path = process.argv[2];

if (!path) {
  console.log("Usage: node src/bridgeCore.js <path-to-json>");
  process.exit(1);
}

function describe(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    if (value.length > 0) {
      console.log(pad + `array[${typeof value[0]}]`);
      if (typeof value[0] === "object") describe(value[0], indent + 2);
    } else {
      console.log(pad + "array[empty]");
    }
  } else if (value && typeof value === "object") {
    console.log(pad + "object {");
    for (const [k, v] of Object.entries(value)) {
      process.stdout.write(pad + "  " + k + ": ");
      if (typeof v === "object") {
        console.log();
        describe(v, indent + 4);
      } else {
        console.log(typeof v);
      }
    }
    console.log(pad + "}");
  } else {
    console.log(pad + typeof value);
  }
}

const raw = fs.readFileSync(path, "utf8");
const data = JSON.parse(raw);

console.log(`\n=== SHAPE OF ${path} ===`);
describe(data);

// Build a simple summary
let keyCount = 0, arrayCount = 0, objectCount = 0;

function countStructure(value) {
  if (Array.isArray(value)) {
    arrayCount++;
    if (value.length > 0) countStructure(value[0]);
  } else if (value && typeof value === "object") {
    objectCount++;
    for (const v of Object.values(value)) countStructure(v);
  } else {
    keyCount++;
  }
}
countStructure(data);

const summary = { keyCount, arrayCount, objectCount };
const outPath = path.replace(".json", "_summary.json");
fs.writeFileSync(outPath, JSON.stringify(summary, null, 2));

console.log(`\nSummary written → ${outPath}`);

