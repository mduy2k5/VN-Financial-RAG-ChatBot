try {
  const current = rs.status();
  if (current.ok === 1) {
    print("Financial MongoDB replica set is already initialized.");
  }
} catch (_error) {
  const result = rs.initiate({
    _id: "financial-rs",
    members: [{ _id: 0, host: "financial-mongodb:27017" }],
  });
  if (result.ok !== 1 && result.codeName !== "AlreadyInitialized") {
    throw new Error(`Could not initialize financial-rs: ${JSON.stringify(result)}`);
  }
}

for (let attempt = 0; attempt < 60; attempt += 1) {
  if (db.hello().isWritablePrimary) {
    print("Financial MongoDB replica set is ready.");
    quit(0);
  }
  sleep(1000);
}

throw new Error("financial-rs did not elect a writable primary within 60 seconds");
