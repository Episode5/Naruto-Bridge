const fs = require("fs");

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

const path = __dirname + "/../schemas/ninja.json";
const raw = fs.readFileSync(path, "utf8");
const data = JSON.parse(raw);

console.log("Schema shape of:", path);
describe(data);

