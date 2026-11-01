/** Promise forms of the IndexedDB calls the project store uses. */

export function request<T>(r: IDBRequest<T>): Promise<T> {
  const { promise, resolve, reject } = Promise.withResolvers<T>();
  r.onsuccess = () => resolve(r.result);
  r.onerror = () => reject(r.error);
  return promise;
}

/** Resolves when the transaction commits; rejects with its error (e.g. QuotaExceededError) when it aborts. */
export function committed(tx: IDBTransaction): Promise<void> {
  const { promise, resolve, reject } = Promise.withResolvers<void>();
  tx.oncomplete = () => resolve();
  tx.onerror = () => reject(tx.error);
  tx.onabort = () => reject(tx.error ?? new DOMException('Transaction aborted', 'AbortError'));
  return promise;
}

export function openDb(name: string, version: number, upgrade: (db: IDBDatabase) => void): Promise<IDBDatabase> {
  const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>();
  if (typeof indexedDB === 'undefined') {
    reject(new Error('This browser has no IndexedDB'));
    return promise;
  }
  const r = indexedDB.open(name, version);
  r.onupgradeneeded = () => upgrade(r.result);
  r.onsuccess = () => resolve(r.result);
  r.onerror = () => reject(r.error);
  r.onblocked = () => reject(new Error('Another tab holds an older version of the studio’s storage open: close it and reload'));
  return promise;
}
