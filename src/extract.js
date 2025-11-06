const fs = require("fs");

const raw = fs.readFileSync(__dirname + "/../schemas/ninja.json", "utf8");
const data = JSON.parse(raw);

// Safety
console.log("DEBUG:", data);

console.log("Name:", data.name);
console.log("Rank:", data.rank);

if (Array.isArray(data.skills)) {
  console.log("Primary skill:", data.skills[0]);
} else {
  console.log("No skills array found!");
}

console.log("Chakra level:", data.stats?.chakra);

