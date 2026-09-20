// PNGs are already compressed. A small standard ZIP writer avoids remote code
// and dependencies; the resulting archive is readable by Python's zipfile.
const encoder = new TextEncoder();
const MAX_ARCHIVE_BYTES = 250 * 1024 * 1024;
const crcTable = Uint32Array.from({length: 256}, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit++) value = (value & 1) ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  return value >>> 0;
});

function crc32(data) {
  let crc = 0xffffffff;
  for (const byte of data) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function header(length) {
  const bytes = new Uint8Array(length);
  return {bytes, view: new DataView(bytes.buffer)};
}

export function buildZip(entries) {
  if (!Array.isArray(entries) || !entries.length || entries.length > 64) {
    throw new Error('Export between 1 and 64 files in one capture ZIP.');
  }
  const files = [], directory = [], names = new Set();
  let offset = 0, directorySize = 0;
  for (const entry of entries) {
    const name = entry.name;
    if (typeof name !== 'string' || !name || /[\\:\x00-\x1f]/.test(name) ||
        name.split('/').some(part => !part || part === '.' || part === '..') || names.has(name)) {
      throw new Error('Capture ZIP contains an invalid or duplicate filename.');
    }
    names.add(name);
    const filename = encoder.encode(name);
    if (filename.length > 65535) throw new Error('Capture filename is too long.');
    const data = typeof entry.data === 'string' ? encoder.encode(entry.data) : entry.data;
    if (!(data instanceof Uint8Array)) throw new Error('Capture ZIP entries must contain bytes or text.');
    if (offset + data.byteLength + filename.length * 2 + directorySize + 98 > MAX_ARCHIVE_BYTES) {
      throw new Error('The capture ZIP exceeds 250 MiB. Export fewer or smaller captures.');
    }
    const checksum = crc32(data);
    const local = header(30);
    local.view.setUint32(0, 0x04034b50, true);
    local.view.setUint16(4, 20, true);
    local.view.setUint16(6, 0x0800, true); // UTF-8 names; stored entries.
    local.view.setUint16(12, 33, true); // DOS date 1980-01-01; actual times in manifest.
    local.view.setUint32(14, checksum, true);
    local.view.setUint32(18, data.byteLength, true);
    local.view.setUint32(22, data.byteLength, true);
    local.view.setUint16(26, filename.length, true);
    files.push(local.bytes, filename, data);

    const central = header(46);
    central.view.setUint32(0, 0x02014b50, true);
    central.view.setUint16(4, 20, true);
    central.view.setUint16(6, 20, true);
    central.view.setUint16(8, 0x0800, true);
    central.view.setUint16(14, 33, true);
    central.view.setUint32(16, checksum, true);
    central.view.setUint32(20, data.byteLength, true);
    central.view.setUint32(24, data.byteLength, true);
    central.view.setUint16(28, filename.length, true);
    central.view.setUint32(42, offset, true);
    directory.push(central.bytes, filename);
    directorySize += central.bytes.length + filename.length;
    offset += local.bytes.length + filename.length + data.byteLength;
  }
  const end = header(22);
  end.view.setUint32(0, 0x06054b50, true);
  end.view.setUint16(8, entries.length, true);
  end.view.setUint16(10, entries.length, true);
  end.view.setUint32(12, directorySize, true);
  end.view.setUint32(16, offset, true);
  return new Blob([...files, ...directory, end.bytes], {type: 'application/zip'});
}
