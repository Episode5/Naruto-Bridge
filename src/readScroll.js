const fs = require("fs");

function readScroll(path) {
  try {
    const raw = fs.readFileSync(path, "utf8");
    return JSON.parse(raw);
  } catch (err) {
    console.error("Error reading scroll:", err.message);
    return null;
  }
}

function writeScroll(path, data) {
  fs.writeFileSync(path, JSON.stringify(data, null, 2));
  console.log("Scroll written →", path);
}

module.exports = { readScroll, writeScroll };

