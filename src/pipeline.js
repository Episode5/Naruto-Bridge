const fs = require("fs");
const { createNinjaModel } = require("./model");

// 1) Read raw scroll
const raw = fs.readFileSync(__dirname + "/../schemas/ninja.json", "utf8");
const data = JSON.parse(raw);

// 2) Convert to internal model
const ninja = createNinjaModel(data);

// 3) Transform OR augment data
const upgraded = {
  ...ninja,
  chakra: ninja.chakra + 1,  // Example: buff chakra by +1
  status: "enhanced",
};

// 4) Save new scroll
fs.writeFileSync(
  __dirname + "/../schemas/ninja_pipeline_output.json",
  JSON.stringify(upgraded, null, 2)
);

console.log("Pipeline complete → schemas/ninja_pipeline_output.json");

