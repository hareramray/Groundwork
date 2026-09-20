// Large PNGs live in IndexedDB; extension settings storage is never used for images.
const DATABASE = "groundwork-dataset-captures";
let pendingDatabase;

function database() {
  if (!pendingDatabase) {
    pendingDatabase = new Promise((resolve, reject) => {
      const request = indexedDB.open(DATABASE, 1);
      request.onupgradeneeded = () => {
        const db = request.result;
        db.createObjectStore("batches", {keyPath: "id"});
        db.createObjectStore("images", {keyPath: "id"});
      };
      request.onerror = () => { pendingDatabase = undefined; reject(request.error); };
      request.onsuccess = () => {
        const db = request.result;
        db.onversionchange = () => { db.close(); pendingDatabase = undefined; };
        resolve(db);
      };
    });
  }
  return pendingDatabase;
}

async function read(store, key) {
  const db = await database();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(store, "readonly");
    const request = key === undefined ? transaction.objectStore(store).getAll() : transaction.objectStore(store).get(key);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function write(store, record) {
  const db = await database();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(store, "readwrite");
    transaction.objectStore(store).put(record);
    transaction.oncomplete = () => resolve(record);
    transaction.onerror = () => reject(transaction.error || new Error("Could not save this capture."));
    transaction.onabort = () => reject(transaction.error || new Error("Capture storage was interrupted."));
  });
}

export function saveBatch(batch) {
  if (!batch || typeof batch.id !== "string" || !batch.id) throw new Error("A capture batch ID is required.");
  return write("batches", batch);
}

export async function getBatch(id) { return (await read("batches", id)) || null; }

export async function listBatches() {
  const batches = await read("batches");
  return batches.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || "") || b.id.localeCompare(a.id));
}

export function saveImage(id, dataUrl) {
  if (typeof id !== "string" || !id || typeof dataUrl !== "string" || !dataUrl.startsWith("data:image/png;base64,")) {
    throw new Error("A PNG capture and image ID are required.");
  }
  return write("images", {id, data_url: dataUrl});
}

export async function getImage(id) { return (await read("images", id))?.data_url || null; }

export async function deleteBatch(id) {
  const db = await database();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(["batches", "images"], "readwrite");
    const batches = transaction.objectStore("batches");
    const request = batches.get(id);
    let refused;
    request.onsuccess = () => {
      const batch = request.result;
      if (batch?.status === "capturing") {
        refused = new Error("Stop the active capture before deleting its batch.");
        transaction.abort();
        return;
      }
      for (const capture of batch?.captures || []) transaction.objectStore("images").delete(capture.id);
      batches.delete(id);
    };
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error || new Error("Could not delete this batch."));
    transaction.onabort = () => reject(refused || transaction.error || new Error("Batch deletion was interrupted."));
  });
}
