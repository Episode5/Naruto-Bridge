function summarizeData(data) {
  let keyCount = 0,
    arrayCount = 0,
    objectCount = 0,
    numberCount = 0,
    numberSum = 0;

  function walk(value) {
    if (Array.isArray(value)) {
      arrayCount++;
      value.forEach((v) => walk(v));
    } else if (value && typeof value === "object") {
      objectCount++;
      for (const [k, v] of Object.entries(value)) {
        keyCount++;
        walk(v);
      }
    } else if (typeof value === "number") {
      numberCount++;
      numberSum += value;
    }
  }

  walk(data);
  const avg = numberCount ? (numberSum / numberCount).toFixed(2) : 0;
  return { keyCount, arrayCount, objectCount, numberCount, avg };
}

module.exports = { summarizeData };

