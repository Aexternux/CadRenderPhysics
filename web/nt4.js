// Minimal NetworkTables 4 client for the browser (the protocol AdvantageScope and the robot speak).
//
//   const nt = new NT4Client({ host: '127.0.0.1', onUpdate, onTopics, onStatus });
//   nt.connect();  ...  nt.topics (Map name -> {name, type, value, time})
//
// Reads everything the robot publishes, including AdvantageKit (/AdvantageKit/RealOutputs/...,
// /AdvantageKit/<inputs>/...). Struct values (struct:Pose2d, struct:Pose3d[], ...) are decoded
// with the schemas the robot publishes under /.schema/, with built-in fallbacks for common
// geometry types. Read-only: it never publishes.

const PORT = 5810;
const TYPE_BY_ID = { 0: 'boolean', 1: 'double', 2: 'int', 3: 'float', 4: 'string', 5: 'raw',
  16: 'boolean[]', 17: 'double[]', 18: 'int[]', 19: 'float[]', 20: 'string[]' };

// ---------------- MessagePack (decode only; NT4 binary frames are concatenated values) --------
export function msgpackDecodeAll(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let o = 0;
  const td = new TextDecoder();
  const str = (n) => { const s = td.decode(buf.subarray(o, o + n)); o += n; return s; };
  const bin = (n) => { const b = buf.slice(o, o + n); o += n; return b; };
  const arr = (n) => { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = read(); return a; };
  const map = (n) => { const m = {}; for (let i = 0; i < n; i++) { const k = read(); m[k] = read(); } return m; };
  function read() {
    const t = dv.getUint8(o++);
    if (t <= 0x7f) return t;
    if (t >= 0xe0) return t - 0x100;
    if ((t & 0xf0) === 0x80) return map(t & 0x0f);
    if ((t & 0xf0) === 0x90) return arr(t & 0x0f);
    if ((t & 0xe0) === 0xa0) return str(t & 0x1f);
    let v;
    switch (t) {
      case 0xc0: return null;
      case 0xc2: return false;
      case 0xc3: return true;
      case 0xc4: v = dv.getUint8(o); o += 1; return bin(v);
      case 0xc5: v = dv.getUint16(o); o += 2; return bin(v);
      case 0xc6: v = dv.getUint32(o); o += 4; return bin(v);
      case 0xca: v = dv.getFloat32(o); o += 4; return v;
      case 0xcb: v = dv.getFloat64(o); o += 8; return v;
      case 0xcc: v = dv.getUint8(o); o += 1; return v;
      case 0xcd: v = dv.getUint16(o); o += 2; return v;
      case 0xce: v = dv.getUint32(o); o += 4; return v;
      case 0xcf: v = Number(dv.getBigUint64(o)); o += 8; return v;
      case 0xd0: v = dv.getInt8(o); o += 1; return v;
      case 0xd1: v = dv.getInt16(o); o += 2; return v;
      case 0xd2: v = dv.getInt32(o); o += 4; return v;
      case 0xd3: v = Number(dv.getBigInt64(o)); o += 8; return v;
      case 0xd9: v = dv.getUint8(o); o += 1; return str(v);
      case 0xda: v = dv.getUint16(o); o += 2; return str(v);
      case 0xdb: v = dv.getUint32(o); o += 4; return str(v);
      case 0xdc: v = dv.getUint16(o); o += 2; return arr(v);
      case 0xdd: v = dv.getUint32(o); o += 4; return arr(v);
      case 0xde: v = dv.getUint16(o); o += 2; return map(v);
      case 0xdf: v = dv.getUint32(o); o += 4; return map(v);
      default: throw new Error('msgpack: unsupported type 0x' + t.toString(16));
    }
  }
  const out = [];
  while (o < buf.length) out.push(read());
  return out;
}
export function msgpackEncode(v) {          // only what the client sends: small int arrays
  const out = [];
  const w = (x) => {
    if (Array.isArray(x)) { out.push(0x90 | x.length); x.forEach(w); return; }
    if (Number.isInteger(x) && x >= 0 && x < 128) { out.push(x); return; }
    if (Number.isInteger(x) && x < 0 && x >= -32) { out.push(0x100 + x); return; }
    const b = new DataView(new ArrayBuffer(9)); b.setUint8(0, 0xd3); b.setBigInt64(1, BigInt(Math.round(x)));
    for (let i = 0; i < 9; i++) out.push(b.getUint8(i));
  };
  w(v); return new Uint8Array(out);
}

// ---------------- WPILib structs ----------------
const BUILTIN_SCHEMAS = {
  Translation2d: 'double x;double y', Rotation2d: 'double value',
  Pose2d: 'Translation2d translation;Rotation2d rotation',
  Translation3d: 'double x;double y;double z', Quaternion: 'double w;double x;double y;double z',
  Rotation3d: 'Quaternion q', Pose3d: 'Translation3d translation;Rotation3d rotation',
  Transform2d: 'Translation2d translation;Rotation2d rotation', Transform3d: 'Translation3d translation;Rotation3d rotation',
  Twist2d: 'double dx;double dy;double dtheta', ChassisSpeeds: 'double vx;double vy;double omega',
  SwerveModuleState: 'double speed;Rotation2d angle', SwerveModulePosition: 'double distance;Rotation2d angle',
};
const PRIM = { bool: [1, 'u8'], char: [1, 'u8'], int8: [1, 'i8'], uint8: [1, 'u8'], int16: [2, 'i16'], uint16: [2, 'u16'],
  int32: [4, 'i32'], uint32: [4, 'u32'], int64: [8, 'i64'], uint64: [8, 'u64'], float: [4, 'f32'], float32: [4, 'f32'],
  double: [8, 'f64'], float64: [8, 'f64'] };

export class StructDecoder {
  constructor() { this.schemas = { ...BUILTIN_SCHEMAS }; this.cache = {}; }
  addSchema(name, text) { this.schemas[name] = text; this.cache = {}; }
  layout(name) {                       // -> {size, fields:[{name, prim|struct, count, offset}]}
    if (this.cache[name]) return this.cache[name];
    const text = this.schemas[name]; if (text == null) return null;
    const fields = []; let off = 0;
    for (let decl of text.split(';')) {
      decl = decl.trim(); if (!decl) continue;
      decl = decl.replace(/^enum\s*\{[^}]*\}\s*/, '');           // enum {a=1,b=2} int8 x -> int8 x
      const m = decl.match(/^(\w+)\s+(\w+)\s*(?:\[(\d+)\])?\s*(?::\s*\d+)?$/); if (!m) return null;
      const [, type, fname, cnt] = m, count = cnt ? +cnt : 1;
      if (PRIM[type]) { fields.push({ name: fname, prim: type, count, offset: off }); off += PRIM[type][0] * count; }
      else { const sub = this.layout(type); if (!sub) return null;
        fields.push({ name: fname, struct: type, count, offset: off }); off += sub.size * count; }
    }
    return (this.cache[name] = { size: off, fields });
  }
  decodeOne(name, dv, base) {
    const L = this.layout(name); if (!L) return null;
    const o = {};
    for (const f of L.fields) {
      const one = (i) => {
        if (f.prim) { const [sz, k] = PRIM[f.prim], at = base + f.offset + i * sz;
          switch (k) { case 'u8': return f.prim === 'bool' ? dv.getUint8(at) !== 0 : dv.getUint8(at);
            case 'i8': return dv.getInt8(at); case 'i16': return dv.getInt16(at, true); case 'u16': return dv.getUint16(at, true);
            case 'i32': return dv.getInt32(at, true); case 'u32': return dv.getUint32(at, true);
            case 'i64': return Number(dv.getBigInt64(at, true)); case 'u64': return Number(dv.getBigUint64(at, true));
            case 'f32': return dv.getFloat32(at, true); default: return dv.getFloat64(at, true); } }
        return this.decodeOne(f.struct, dv, base + f.offset + i * this.layout(f.struct).size);
      };
      o[f.name] = f.count === 1 ? one(0) : Array.from({ length: f.count }, (_, i) => one(i));
    }
    return o;
  }
  // type like "struct:Pose2d" or "struct:Pose3d[]"; bytes: Uint8Array
  decode(type, bytes) {
    const m = type.match(/^struct:(\w+)(\[\])?$/); if (!m) return null;
    const L = this.layout(m[1]); if (!L || !L.size) return null;
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (!m[2]) return bytes.length >= L.size ? this.decodeOne(m[1], dv, 0) : null;
    const out = []; for (let at = 0; at + L.size <= bytes.length; at += L.size) out.push(this.decodeOne(m[1], dv, at));
    return out;
  }
}

// ---------------- client ----------------
export class NT4Client {
  constructor({ host = '127.0.0.1', name = 'CadSim', onUpdate, onTopics, onStatus } = {}) {
    Object.assign(this, { host, name, onUpdate, onTopics, onStatus });
    this.topics = new Map();      // name -> {name, type, value, time}
    this.byId = new Map();
    this.structs = new StructDecoder();
    this.ws = null; this.wantOpen = false; this.status = 'disconnected';
  }
  setStatus(s) { this.status = s; this.onStatus && this.onStatus(s); }
  connect(host) {
    if (host) this.host = host;
    this.wantOpen = true; this.open();
  }
  disconnect() { this.wantOpen = false; clearTimeout(this.retry); if (this.ws) this.ws.close(); this.ws = null; this.setStatus('disconnected'); }
  open() {
    clearTimeout(this.retry);
    if (this.ws) { try { this.ws.close(); } catch (e) {} }
    this.setStatus('connecting');
    let ws;
    try { ws = new WebSocket(`ws://${this.host}:${PORT}/nt/${encodeURIComponent(this.name)}`,
                             ['v4.1.networktables.first.wpi.edu', 'networktables.first.wpi.edu']); }
    catch (e) { this.setStatus('error: ' + e.message); return; }
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => {
      this.topics.clear(); this.byId.clear(); this.onTopics && this.onTopics();
      this.setStatus('connected');
      ws.send(JSON.stringify([{ method: 'subscribe', params: { topics: ['/'], subuid: 1,
        options: { prefix: true, periodic: 0.02 } } }]));
    };
    ws.onmessage = (ev) => (typeof ev.data === 'string' ? this.onText(ev.data) : this.onBinary(new Uint8Array(ev.data)));
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      if (this.wantOpen) { this.setStatus('waiting for robot…'); this.retry = setTimeout(() => this.open(), 1500); }
      else this.setStatus('disconnected');
    };
    ws.onerror = () => {};
  }
  onText(text) {
    let msgs; try { msgs = JSON.parse(text); } catch (e) { return; }
    let changed = false;
    for (const m of msgs) {
      const p = m.params || {};
      if (m.method === 'announce') {
        const t = { name: p.name, id: p.id, type: p.type, value: undefined, time: 0 };
        this.topics.set(p.name, t); this.byId.set(p.id, t); changed = true;
      } else if (m.method === 'unannounce') {
        this.topics.delete(p.name); this.byId.delete(p.id); changed = true;
      }
    }
    if (changed && this.onTopics) this.onTopics();
  }
  onBinary(buf) {
    let vals; try { vals = msgpackDecodeAll(buf); } catch (e) { return; }
    for (const v of vals) {
      if (!Array.isArray(v) || v.length < 4) continue;
      const [id, time, typeId, raw] = v;
      const t = this.byId.get(id); if (!t) continue;
      let value = raw;
      if (t.type === 'structschema' && raw instanceof Uint8Array) {
        this.structs.addSchema(t.name.replace(/^\/\.schema\/struct:/, ''), new TextDecoder().decode(raw));
      } else if (t.type.startsWith('struct:') && raw instanceof Uint8Array) {
        value = this.structs.decode(t.type, raw) ?? raw;
      } else if (TYPE_BY_ID[typeId] === 'raw' && t.type === 'json' && raw instanceof Uint8Array) {
        try { value = JSON.parse(new TextDecoder().decode(raw)); } catch (e) {}
      }
      t.value = value; t.time = time;
      this.onUpdate && this.onUpdate(t);
    }
  }
}

// ---------------- helpers for consumers ----------------
// Value at a dotted path inside a topic value ("translation.x", "2.rotation.value").
export function pick(value, path) {
  if (!path) return value;
  let v = value;
  for (const k of path.split('.')) { if (v == null) return undefined; v = v[k]; }
  return v;
}
// A robot pose from whatever was dropped: struct Pose2d/Pose3d, or a legacy double[] [x, y, rad]
// (or [x,y,z,qw,qx,qy,qz]). Returns {x, y, z, yaw, q?} in metres / radians, or null.
export function asPose(v) {
  if (v == null) return null;
  if (Array.isArray(v) && typeof v[0] === 'number') {
    if (v.length >= 7) return { x: v[0], y: v[1], z: v[2], q: { w: v[3], x: v[4], y: v[5], z: v[6] } };
    if (v.length >= 3) return { x: v[0], y: v[1], z: 0, yaw: v[2] };
    return null;
  }
  if (Array.isArray(v)) return asPose(v[0]);
  const tr = v.translation;
  if (tr && v.rotation) {
    if (v.rotation.q) return { x: tr.x, y: tr.y, z: tr.z || 0, q: v.rotation.q };
    if ('value' in v.rotation) return { x: tr.x, y: tr.y, z: 0, yaw: v.rotation.value };
  }
  return null;
}
