const https = require("https");

function fetchScroll(url) {
  return new Promise((resolve, reject) => {
    https
      .get(url, (res) => {
        let data = "";
        res.on("data", (chunk) => (data += chunk));
        res.on("end", () => {
          try {
            const json = JSON.parse(data);
            resolve(json);
          } catch (err) {
            reject(new Error("Failed to parse JSON: " + err.message));
          }
        });
      })
      .on("error", (err) => reject(err));
  });
}

module.exports = { fetchScroll };

