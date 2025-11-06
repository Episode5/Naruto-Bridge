function describe(value, indent = 0) {
  const pad = " ".repeat(indent);
  if (Array.isArray(value)) {
    console.log(pad + "array");
    if (value[0]) describe(value[0], indent + 2);
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

module.exports = { describe };

