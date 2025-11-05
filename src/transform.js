const fs = require("fs");

const raw = fs.readFileSync(__dirname + "/../schemas/sample.json", "utf8");
const data = JSON.parse(raw);

const transformed = {
        original_message: data.message,
        new_message: data.message + " - transformed!",
        power_doubled: data.power_level * 2,
        ready: data.ready,
};

fs.writeFileSync(
        __dirname + "/../schemas/transformed.json",
        JSON.stringify(transformed, null, 2)
);

console.log("Transformation complete: schemas/transformed.json");

