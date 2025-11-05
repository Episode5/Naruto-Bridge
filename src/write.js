const fs = require("fs");

const output = {
        status: "success",
        timestamp: new Date().toISOString(),
        secret: "Shadow Clone",
};

fs.writeFileSync(__dirname + "/../schemas/output.json", JSON.stringify(output, null, 2));

console.log("Output scroll written to schemas/output.json");
