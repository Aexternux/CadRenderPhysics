// Guess which parts swing about a hinge, from geometry alone.
//
// Used when the CAD has no mate for a mechanism (e.g. a hood fastened in place): the user clicks
// the hinge shaft/bolt and one part of the thing that swings. We then
//   1. set aside the "axle stack" -- parts that sit on the hinge axis and stay close to it
//      (shaft, bearings, spacers, washers, collars, bolt + nut). Spinning those about their own
//      axis looks the same either way, and they are what joins the moving side to the fixed side;
//   2. set aside belts/chains (they wrap from a pulley on the moving side to one on the frame);
//   3. flood-fill from the clicked part through parts that touch (vertices within TOL).
// Whatever the flood reaches is the guess. Bolts and brackets elsewhere can still bridge to the
// frame, so the UI shows the guess for the user to correct.
//
// parts: [{ key, name, pts: Float32Array of world-space xyz }]  (pts in metres)
// axis:  { origin:[x,y,z], dir:[x,y,z] }

export const TOUCH_TOL = 0.0015;      // m: vertices this close count as touching
const HARDWARE = 0.05;                // m: parts smaller than this are fasteners/spacers (cost no link)
const MIN_CELLS = 6;                  // contact patches smaller than this are grazes (real joints: 30+)
const FLEX_RE = /\b(belt|chain)\b/i;

function radial(pts, o, d) {           // min/max distance of the points from the axis line
  let rmin = Infinity, rmax = 0;
  for (let i = 0; i < pts.length; i += 3) {
    const x = pts[i] - o[0], y = pts[i + 1] - o[1], z = pts[i + 2] - o[2];
    const t = x * d[0] + y * d[1] + z * d[2];
    const r = Math.hypot(x - t * d[0], y - t * d[1], z - t * d[2]);
    if (r < rmin) rmin = r; if (r > rmax) rmax = r;
  }
  return { rmin, rmax };
}

// groupOf (optional): key -> id of the sub-assembly it belongs to; a sub-assembly moves as a unit
export function guessSwinging({ parts, axis, hingeKeys, seedKey, seedKeys, groupOf, tol = TOUCH_TOL }) {
  const seeds = new Set(seedKeys || [seedKey]);
  const n = Math.hypot(...axis.dir), d = axis.dir.map(x => x / n), o = axis.origin;
  const byKey = new Map(parts.map(p => [p.key, p]));
  // how thick the axle is: the clicked hinge part's radius about the axis
  // (a bearing/spacer/washer was clicked: its bore is the axle; a shaft or bolt: its outside)
  let rHinge = 0;
  for (const k of hingeKeys) { const p = byKey.get(k); if (!p) continue;
    const { rmin, rmax } = radial(p.pts, o, d); rHinge = Math.max(rHinge, rmin > 0.002 ? rmin : rmax); }
  rHinge = Math.min(Math.max(rHinge, 0.003), 0.03);
  // an axle-stack part hugs the axle (touches it) and stays within a bearing's size of the axis
  const AXLE_R = Math.max(2.6 * rHinge, 0.02);
  const axle = new Set(hingeKeys), flex = new Set();
  for (const p of parts) {
    if (seeds.has(p.key)) continue;
    if (FLEX_RE.test(p.name || '')) { flex.add(p.key); continue; }
    const { rmin, rmax } = radial(p.pts, o, d);
    if (rmin < rHinge + 0.003 && rmax < AXLE_R) axle.add(p.key);
  }
  // contact graph: spatial hash of every vertex (cell = tol), recording WHERE each pair touches
  const active = parts.filter(p => !axle.has(p.key));     // belts kept for the cleanup pass
  const idx = new Map(active.map((p, i) => [p.key, i]));
  const N = active.length, cell = tol, grid = new Map();
  const hash = (x, y, z) => (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
  active.forEach((p, i) => {
    const P = p.pts;
    for (let v = 0; v < P.length; v += 3) {
      const ix = Math.floor(P[v] / cell), iy = Math.floor(P[v + 1] / cell), iz = Math.floor(P[v + 2] / cell);
      const h = hash(ix, iy, iz);
      let c = grid.get(h); if (!c) grid.set(h, c = { ix, iy, iz, parts: [] });
      if (c.parts[c.parts.length - 1] !== i && !c.parts.includes(i)) c.parts.push(i);
    }
  });
  // pair -> extent of its contact patch along the axis (t) and about it (r)
  const patch = new Map();
  for (const c of grid.values()) {
    const x = (c.ix + 0.5) * cell - o[0], y = (c.iy + 0.5) * cell - o[1], z = (c.iz + 0.5) * cell - o[2];
    const t = x * d[0] + y * d[1] + z * d[2], r = Math.hypot(x - t * d[0], y - t * d[1], z - t * d[2]);
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const nb = grid.get(hash(c.ix + dx, c.iy + dy, c.iz + dz)); if (!nb) continue;
      for (const i of c.parts) for (const j of nb.parts) {
        if (i >= j) continue;
        const k = i * N + j; let e = patch.get(k);
        if (!e) patch.set(k, e = [t, t, r, r, 1]);
        else { if (t < e[0]) e[0] = t; if (t > e[1]) e[1] = t; if (r < e[2]) e[2] = r; if (r > e[3]) e[3] = r; e[4]++; }
      }
    }
  }
  // A contact that survives the hinge turning is either a flat face square to the axis (thrust
  // face: plate rubbing on the pivot holder, a screw head flush against the frame) or a ring
  // around it (bushing). Those join the moving side to the fixed side, so they don't connect.
  // Only applied near the hinge: elsewhere bolted plates also touch in flat patches (a mesh bolt
  // only has vertices at its ends, so it "touches" its plate just under the head).
  const SLIDE = 3 * cell, NEAR = Math.max(8 * rHinge, 0.06);
  const adj = active.map(() => new Set()), raw = active.map(() => new Set());
  const isFlex = active.map(p => flex.has(p.key));
  for (const [k, e] of patch) {
    if (e[4] < MIN_CELLS) continue;                 // a graze (CAD near-miss), not a joint
    { const i = Math.floor(k / N), j = k % N; raw[i].add(j); raw[j].add(i); }
    if (isFlex[Math.floor(k / N)] || isFlex[k % N]) continue;
    if (e[2] < NEAR && (e[1] - e[0] < SLIDE || e[3] - e[2] < SLIDE)) continue;
    const i = Math.floor(k / N), j = k % N; adj[i].add(j); adj[j].add(i);
  }
  const starts = [...seeds].map(k => idx.get(k)).filter(i => i !== undefined);
  const moving = new Set(), hop = new Map();
  if (starts.length) {
    // breadth-first, counting "links": stepping onto a structural part costs 1, onto small
    // hardware (screws, nuts, spacers, rivets) costs 0 -- so plate -> bolt -> plate is one link
    const small = active.map(p => { let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9]; const P = p.pts;
      for (let v = 0; v < P.length; v += 3) for (let a = 0; a < 3; a++) { if (P[v + a] < mn[a]) mn[a] = P[v + a]; if (P[v + a] > mx[a]) mx[a] = P[v + a]; }
      return Math.hypot(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]) < HARDWARE; });
    const members = new Map();             // sub-assembly id -> active indices
    if (groupOf) active.forEach((p, i) => { const g = groupOf(p.key); if (g) { let a = members.get(g); if (!a) members.set(g, a = []); a.push(i); } });
    let frontier = [];
    const reach = (j, h, list) => {         // mark j (and the rest of its sub-assembly) at link h
      const g = groupOf && groupOf(active[j].key), all = g ? members.get(g) : [j];
      for (const x of all) if (!moving.has(x)) { moving.add(x); hop.set(x, h); list.push(x); }
    };
    for (const st of starts) reach(st, 0, frontier);
    while (frontier.length) {                 // 0-1 BFS: hardware first within a level
      const next = [];
      for (let q = 0; q < frontier.length; q++) { const i = frontier[q];
        for (const j of adj[i]) { if (moving.has(j)) continue;
          if (small[j]) reach(j, hop.get(i), frontier); else reach(j, hop.get(i) + 1, next); } }
      frontier = next;
    }
    // cleanup: a part left out (a screw flush against the frame at the hinge, a belt inside the
    // arm) joins if at least half of what it touches is moving
    const add = [];
    active.forEach((_, i) => {
      if (moving.has(i) || !raw[i].size) return;
      let mv = 0; for (const j of raw[i]) if (moving.has(j)) mv++;
      if (mv && mv * 2 >= raw[i].size) add.push(i);
    });
    for (const i of add) { moving.add(i);
      let h = Infinity; for (const j of raw[i]) if (hop.has(j)) h = Math.min(h, hop.get(j)); hop.set(i, h); }
  }
  return {
    moving: [...moving].map(i => active[i].key),
    hops: Object.fromEntries([...hop].map(([i, h]) => [active[i].key, h])),   // links from the clicked part
    axle: [...axle], flex: [...flex],
    leaked: moving.size > 0.6 * active.length,     // probably flooded into the frame too
  };
}

// ---- hinge axis from the clicked part(s): principal axis + circle fit (no THREE needed) ----
function eig3(C) {                       // Jacobi eigen-decomposition of a symmetric 3x3
  const A = C.map(r => r.slice()), V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 30; sweep++) for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
    if (Math.abs(A[p][q]) < 1e-15) continue;
    const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
    const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
    for (let k = 0; k < 3; k++) { const a = A[k][p], b = A[k][q]; A[k][p] = c * a - s * b; A[k][q] = s * a + c * b; }
    for (let k = 0; k < 3; k++) { const a = A[p][k], b = A[q][k]; A[p][k] = c * a - s * b; A[q][k] = s * a + c * b; }
    for (let k = 0; k < 3; k++) { const a = V[k][p], b = V[k][q]; V[k][p] = c * a - s * b; V[k][q] = s * a + c * b; }
  }
  return [0, 1, 2].map(i => ({ val: A[i][i], vec: [V[0][i], V[1][i], V[2][i]] })).sort((x, y) => y.val - x.val);
}
// A shaft/bolt is long along its axis; a bearing/washer/spacer is disc-like (axis = thinnest
// direction). The axis passes through the centre of the circle fitted to its outer surface.
export function fitHingeAxis(ptArrays) {
  const P = []; for (const a of ptArrays) for (let i = 0; i < a.length; i += 3) P.push([a[i], a[i + 1], a[i + 2]]);
  if (P.length < 6) return null;
  const c = [0, 1, 2].map(k => P.reduce((s, p) => s + p[k], 0) / P.length);
  const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const p of P) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) C[i][j] += (p[i] - c[i]) * (p[j] - c[j]);
  const [e1, e2, e3] = eig3(C);
  let d = (e1.val - e2.val) >= (e2.val - e3.val) ? e1.vec : e3.vec;
  const big = [0, 1, 2].reduce((b, i) => Math.abs(d[i]) > Math.abs(d[b]) ? i : b, 0);
  if (d[big] < 0) d = d.map(x => -x);
  // circle fit in the plane square to d (Kasa), refit on the outer shell
  let u = Math.abs(d[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const ud = u[0] * d[0] + u[1] * d[1] + u[2] * d[2]; u = u.map((x, i) => x - ud * d[i]);
  const un = Math.hypot(...u); u = u.map(x => x / un);
  const v = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];
  const xy = P.map(p => { const r = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; return [r[0] * u[0] + r[1] * u[1] + r[2] * u[2], r[0] * v[0] + r[1] * v[1] + r[2] * v[2]]; });
  const fit = (sel) => {
    let sxx = 0, sxy = 0, syy = 0, sx = 0, sy = 0, n = 0, sxz = 0, syz = 0, sz = 0;
    for (const [x, y] of sel) { const z = x * x + y * y; sxx += x * x; sxy += x * y; syy += y * y; sx += x; sy += y; n++; sxz += x * z; syz += y * z; sz += z; }
    const M = [[2 * sxx, 2 * sxy, sx], [2 * sxy, 2 * syy, sy], [2 * sx, 2 * sy, n]], b = [sxz, syz, sz];
    const det = m => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const D = det(M); if (Math.abs(D) < 1e-18) return null;
    const col = i => M.map((r, ri) => r.map((x, ci) => ci === i ? b[ri] : x));
    return [det(col(0)) / D, det(col(1)) / D];
  };
  let ctr = fit(xy) || [0, 0];
  for (let it = 0; it < 3; it++) {
    const rr = xy.map(([x, y]) => Math.hypot(x - ctr[0], y - ctr[1])), R = Math.max(...rr);
    const outer = xy.filter((_, i) => rr[i] > 0.85 * R); if (outer.length < 8) break;
    const nc = fit(outer); if (!nc) break; ctr = nc;
  }
  const o = [0, 1, 2].map(k => c[k] + u[k] * ctr[0] + v[k] * ctr[1]);
  return { origin: o, dir: d };
}

// The same part mirrored to the other side (a hood/arm has a left and right plate): same name,
// position reflected across the plane square to the hinge axis through the middle of `parts`.
export function mirrorTwins(parts, keys, axis) {
  const n = Math.hypot(...axis.dir), d = axis.dir.map(x => x / n), o = axis.origin;
  const info = new Map(parts.map(p => { const P = p.pts; let c = [0, 0, 0];
    const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
    for (let v = 0; v < P.length; v += 3) for (let a = 0; a < 3; a++) { if (P[v + a] < mn[a]) mn[a] = P[v + a]; if (P[v + a] > mx[a]) mx[a] = P[v + a]; }
    c = [0, 1, 2].map(a => (mn[a] + mx[a]) / 2 - o[a]);
    const t = c[0] * d[0] + c[1] * d[1] + c[2] * d[2];
    return [p.key, { name: p.name, t, rad: [c[0] - t * d[0], c[1] - t * d[1], c[2] - t * d[2]] }]; }));
  const ts = [...info.values()].map(x => x.t), mid = (Math.min(...ts) + Math.max(...ts)) / 2;
  const out = [];
  for (const k of keys) { const a = info.get(k); if (!a || !a.name) continue;
    let best = null, bd = 0.01;
    for (const [k2, b] of info) { if (k2 === k || b.name !== a.name) continue;
      const dist = Math.hypot(b.t - (2 * mid - a.t), b.rad[0] - a.rad[0], b.rad[1] - a.rad[1], b.rad[2] - a.rad[2]);
      if (dist < bd) { bd = dist; best = k2; } }
    if (best) out.push(best); }
  return out;
}
