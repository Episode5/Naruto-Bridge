const fs = require("fs");
const { createNinjaModel } = require("./model");

// 1) Read the original scroll
const raw = fs.readFileSync(__dirname + "/../schemas/ninja.json", "utf8");
const data = JSON.parse(raw);

// 2) Convert to internal model
const ninja = createNinjaModel(data);

// 3) Create new JSON output based on internal representation
const newScroll = {
  ninja_name: ninja.name,
  ninja_rank: ninja.rank,
  primary_skill: ninja.mainSkill,
  chakra_level: ninja.chakra,
};

fs.writeFileSync(
  __dirname + "/../schemas/ninja_output.json",
  JSON.stringify(newScroll, null, 2)
);

console.log("Output scroll written → schemas/ninja_output.json");

