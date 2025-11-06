const fs = require("fs");
const { createNinjaModel } = require("./model");

const raw = fs.readFileSync(__dirname + "/../schemas/ninja.json", "utf8");
const data = JSON.parse(raw);

const ninja = createNinjaModel(data);

console.log("Ninja Model:", ninja);
