// Rigid-body physics for an imported robot, on Rapier (WASM).
//
// Model
//   * chassis      : every part not moved by a joint -> one FIXED body (robot sits on the floor)
//   * mechanisms   : each joint's parts -> one DYNAMIC body, attached to its parent (chassis or a
//                    bigger mechanism it rides on) by a revolute/prismatic joint with the CAD limits
//   * actuation    : force-based position motors (a stiff PD spring) follow the UI sliders, so a
//                    mechanism that runs into something pushes it -- or gets held back -- instead of
//                    passing through. Rollers use a velocity motor and grab pieces by friction.
//   * game pieces  : dynamic spheres you can drop in.
//
// Collision shapes: a convex hull per part; parts whose hull is much bigger than the part itself
// (an L-shaped bumper, a lattice plate...) are split into a few convex chunks instead.
//
// What collides: game pieces and the floor hit everything; MECHANISMS HIT EACH OTHER (an intake
// swinging down shoves a free hopper wall out). Mechanism-vs-chassis contact is opt-in
// (opts.chassisCollide): a CAD packs parts millimetres apart and hull approximations would jam.
// Robot parts that already touch/overlap in the assembled CAD start out as "ghosts" to each other
// and become solid as soon as they separate, so the solver never fights the CAD's rest pose.
//
// Joints are either DRIVEN (a motor follows the slider) or FREE (no motor: pushed around by
// contacts, gravity and its limits -- e.g. a hopper wall the intake pushes out).
//
// All bodies start at the identity pose, with colliders placed at their assembled (rest) positions,
// so a body's pose IS the motion of its parts relative to the CAD: partWorld = bodyPose * rest.

const TINY_PART = 0.012;        // m: parts smaller than this get no collider (screws, washers)
const HULL_POINTS = 400;        // max vertices sampled per part for its hull
const CONCAVE_RATIO = 2.5;      // hull volume / mesh volume above this -> split into convex chunks
const CHUNK_SIZE = 0.06;        // m: target chunk size for concave parts
const CHUNK_MAX = 3;            // max chunks per axis per part
const DENSITY = 600;            // kg/m^3 of hull volume (hulls are "filled", real parts are not)
const PAYLOAD = 3;              // kg of extra load each motor is sized for (game pieces it lifts)
const OVERLAP_TOL = 0.004;      // m: pairs closer than this in the CAD pose start as ghosts. They
                                // can't start solid: a convex hull fills notches/pockets, so parts
                                // nested in the CAD (a tube in a bracket's notch) would be welded.
const TOUCH_TOL = 0.004;        // m: a ghost pair farther apart than this has separated
const DEEP = 0.002;             // m: solid robot pairs sinking deeper than this are hull artefacts
                                // (a part re-entering a notch the hull fills in) -> ghost again
const ROBOT_FRICTION = 0.15;    // metal/polycarb sliding on each other; a grippy value jams
                                // sandwiched parts (a hopper bracket riding in an intake pocket)
const PIECE_FRICTION = 0.7;     // foam game piece. Averaged with what it touches: ~0.43 on robot
                                // walls/plates (foam on polycarb), ~1.0 on rollers, 0.8 on carpet
const ROLLER_FRICTION = 2.0;    // grip-tape rollers on foam. Max-combined so the piece's lower value
                                // doesn't dilute it; static walls stay slick -- a grippy ramp jams a ball
const ROLLER_DENSITY = 500;     // kg/m^3 for a roller's solid cylinder (stands in for a tube + hubs)
const COMPRESS = 0.015;         // m: how far a foam game piece can squash (~0.6 in) before its core
const FOAM_K = 1500;            // N/m foam stiffness (its own weight squashes it ~1.4 mm)
const FOAM_C = 16;              // N·s/m foam damping (~half critical: piles settle, stays explicit-stable)
const MERGE_COS = 0.9;          // foam contacts within ~25 deg of each other count as one surface
const FRICTION_PASSES = 6;      // Gauss-Seidel passes over a ball's contacts per step
const ROLL_RESIST = 0.15;       // foam rolling-resistance coefficient (torque = c * N * R)
const SOFT_CCD = 0.03;          // m: mechanisms predict contacts this far ahead (anti-tunnelling)
const SHRINK = 0.0015;          // m: hulls are shrunk this much so designed-to-touch parts don't lock
const MOTOR_HZ = 5;             // position-motor stiffness, as a natural frequency
const DT = 1 / 120;
const MAX_SUBSTEPS = 3;

// Collision groups (membership << 16 | filter).
const G_CHASSIS = 0x0001, G_PIECE = 0x0002, G_FLOOR = 0x0004, G_MECH = 0x0008, G_SHELL = 0x0010;
const groups = (member, filter) => ((member << 16) | filter) >>> 0;
const FREE_DAMPING = 0.5;       // free joints: light viscous damping (bearing friction), x inertia

export function createPhysics(RAPIER, opts = {}) {
  const chassisCollide = !!(opts.chassisCollide ?? opts.selfCollide);
  // Pairs nested together in the CAD pose (a tube in a bracket's notch) stay ghosts: their hulls
  // fill each other's notches, so making them solid later jams the mechanism short of its stop.
  const releaseGhosts = opts.releaseGhosts ?? false;
  const overlapTol = opts.overlapTol ?? OVERLAP_TOL;
  const MECH_GROUPS = groups(G_MECH, G_PIECE | G_FLOOR | G_MECH | (chassisCollide ? G_CHASSIS : 0));
  const CHASSIS_GROUPS = groups(G_CHASSIS, G_PIECE | G_FLOOR | (chassisCollide ? G_MECH : 0));
  const world = new RAPIER.World({ x: 0, y: 0, z: -9.81 });       // Onshape is Z-up
  world.timestep = DT;
  world.integrationParameters.numSolverIterations = 4;

  const geomCache = new Map();      // geomId -> analysis (sampled points, indices, concave?)
  const robotColliders = new Set(); // handles of robot colliders
  const disabled = new Set();       // "a:b" pairs touching in the CAD: ghosts until they separate
  const cadTouching = new Set();    // every pair that touched in the CAD pose (may re-ghost)
  const approxMech = new Set();     // mechanism colliders that only APPROXIMATE a concave part
  const bodies = new Map();         // bodyId -> {rb, parts:[], parent, joint, ...}
  const partBody = new Map();       // part key -> bodyId
  const colliderPart = new Map();   // collider handle -> part key (debugging / picking)
  const pieces = [];                // {rb, radius, id}
  const pieceByCollider = new Map(); // core collider handle -> piece
  const foam = opts.foamPieces ?? true;
  let pieceSeq = 0, acc = 0, stepCount = 0;
  const events = new RAPIER.EventQueue(true);   // Rapier only runs contact hooks with a queue

  const pairKey = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);

  const hooks = {
    filterContactPair(c1, c2) {
      // EMPTY = contacts are still computed (so we can see when the pair separates) but not solved
      if (disabled.size && disabled.has(pairKey(c1, c2))) return RAPIER.SolverFlags.EMPTY;
      return RAPIER.SolverFlags.COMPUTE_IMPULSE;
    },
    filterIntersectionPair() { return true; },
  };

  // ---------------- geometry analysis (once per unique mesh) ----------------
  // Hull points from a triangle soup (all of it, or the triangles starting at offsets `tris`):
  // evenly strided vertices, pulled SHRINK metres in toward their centre so CAD parts designed to
  // touch (a hard stop, a stowed intake against a hopper) don't overlap and lock as hulls.
  function hullPoints(pos, tris) {
    const idx = [];
    if (tris) for (const t of tris) idx.push(t, t + 3, t + 6);
    else for (let i = 0; i < pos.length; i += 3) idx.push(i);
    if (idx.length < 4) return null;
    const step = Math.max(1, Math.floor(idx.length / HULL_POINTS));
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const i of idx) for (let a = 0; a < 3; a++) { const v = pos[i + a]; if (v < lo[a]) lo[a] = v; if (v > hi[a]) hi[a] = v; }
    const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2, cz = (lo[2] + hi[2]) / 2;
    const pts = [];
    for (let k = 0; k < idx.length; k += step) {
      const i = idx[k], x = pos[i] - cx, y = pos[i + 1] - cy, z = pos[i + 2] - cz;
      const r = Math.hypot(x, y, z) || 1, f = Math.max(0.8, 1 - SHRINK / r);
      pts.push(cx + x * f, cy + y * f, cz + z * f);
    }
    return RAPIER.ColliderDesc.convexHull(new Float32Array(pts)) ? new Float32Array(pts) : null;
  }
  function hullVolume(points) {
    const tmp = world.createCollider(RAPIER.ColliderDesc.convexHull(points));
    const v = tmp.volume(); world.removeCollider(tmp, false); return v;
  }
  function analyse(geomId, pos) {
    if (geomCache.has(geomId)) return geomCache.get(geomId);
    const n = pos.length / 3;
    let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
      const v = pos[i * 3 + a]; if (v < min[a]) min[a] = v; if (v > max[a]) max[a] = v; }
    const diag = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const info = { diag, points: null, chunks: null, concave: false, meshVol: 0 };
    if (!(diag >= TINY_PART) || n < 4) { geomCache.set(geomId, info); return info; }

    // signed mesh volume (divergence theorem over the triangle soup)
    let vol = 0;
    for (let t = 0; t + 8 < pos.length; t += 9) {
      const ax = pos[t], ay = pos[t + 1], az = pos[t + 2], bx = pos[t + 3], by = pos[t + 4], bz = pos[t + 5],
            cx = pos[t + 6], cy = pos[t + 7], cz = pos[t + 8];
      vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
    }
    info.meshVol = Math.abs(vol);

    const whole = hullPoints(pos, null, min, max);
    const hv = whole && hullVolume(whole);
    if (!hv) { geomCache.set(geomId, info); return info; }
    info.concave = info.meshVol > 0 && hv / info.meshVol > CONCAVE_RATIO && diag > 0.06;
    // Concave parts (L-shaped bumper, notched bracket, lattice plate) are split into a grid of
    // ~CHUNK_SIZE cells, one convex hull per cell. Far cheaper than an exact triangle mesh (which made
    // contacts crawl) while keeping the big notches/pockets a single hull would fill in.
    if (info.concave) {
      const ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
      const dims = ext.map(e => Math.max(1, Math.min(CHUNK_MAX, Math.ceil(e / CHUNK_SIZE))));
      const cells = new Map();
      for (let t = 0; t + 8 < pos.length; t += 9) {
        const c = [0, 1, 2].map(a => (pos[t + a] + pos[t + 3 + a] + pos[t + 6 + a]) / 3);
        const id = [0, 1, 2].map(a => Math.min(dims[a] - 1, Math.floor((c[a] - min[a]) / (ext[a] || 1) * dims[a]))).join();
        let list = cells.get(id); if (!list) cells.set(id, list = []);
        list.push(t);
      }
      info.chunks = [];
      for (const tris of cells.values()) {
        const pts = hullPoints(pos, tris);
        if (pts) info.chunks.push(pts);
      }
      if (!info.chunks.length) info.chunks = [whole];
    } else info.chunks = [whole];
    info.points = info.chunks[0];
    geomCache.set(geomId, info);
    return info;
  }

  // rest = 16-float column-major (THREE.Matrix4.elements), rigid
  function decompose(e) {
    const t = { x: e[12], y: e[13], z: e[14] };
    const m00 = e[0], m10 = e[1], m20 = e[2], m01 = e[4], m11 = e[5], m21 = e[6], m02 = e[8], m12 = e[9], m22 = e[10];
    const tr = m00 + m11 + m22; let x, y, z, w;
    if (tr > 0) { const s = 0.5 / Math.sqrt(tr + 1); w = 0.25 / s; x = (m21 - m12) * s; y = (m02 - m20) * s; z = (m10 - m01) * s; }
    else if (m00 > m11 && m00 > m22) { const s = 2 * Math.sqrt(1 + m00 - m11 - m22); w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
    else if (m11 > m22) { const s = 2 * Math.sqrt(1 + m11 - m00 - m22); w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
    else { const s = 2 * Math.sqrt(1 + m22 - m00 - m11); w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
    const n = Math.hypot(x, y, z, w) || 1;
    return { t, q: { x: x / n, y: y / n, z: z / n, w: w / n } };
  }

  function attachPart(bodyId, key, geomId, localPositions, restElements, dynamic, noCollider) {
    const b = bodies.get(bodyId);
    b.parts.push(key); partBody.set(key, bodyId);
    if (noCollider) return;
    const info = analyse(geomId, localPositions);
    if (!info.points) return;
    const { t, q } = decompose(restElements);
    for (const pts of info.chunks) {
      const desc = RAPIER.ColliderDesc.convexHull(pts);
      if (!desc) continue;
      desc.setDensity(DENSITY).setTranslation(t.x, t.y, t.z).setRotation(q).setFriction(ROBOT_FRICTION)
          .setCollisionGroups(dynamic ? MECH_GROUPS : CHASSIS_GROUPS);
      // contact hooks (JS callbacks, costly) only where a ghost pair can exist: mechanisms
      if (dynamic || chassisCollide) desc.setActiveHooks(RAPIER.ActiveHooks.FILTER_CONTACT_PAIRS);
      const c = world.createCollider(desc, b.rb);
      robotColliders.add(c.handle);
      colliderPart.set(c.handle, key);
      if (dynamic && info.concave) approxMech.add(c.handle);
    }
  }

  // exact cylinder along a joint's axis: shape = {type:'cylinder', radius, tmin, tmax} where
  // tmin/tmax are the roller's extent along the axis measured from axis.origin
  function addCylinder(b, j) {
    const { radius, tmin, tmax } = j.shape, d = norm(j.axis.dir), o = j.axis.origin;
    const half = Math.max((tmax - tmin) / 2, 0.005), mid = (tmin + tmax) / 2;
    // rotate the cylinder's local +Y onto the axis
    const y = [0, 1, 0], c = y[0] * d[0] + y[1] * d[1] + y[2] * d[2];
    let q;
    if (c > 0.999999) q = { x: 0, y: 0, z: 0, w: 1 };
    else if (c < -0.999999) q = { x: 1, y: 0, z: 0, w: 0 };
    else { const ax = [y[1] * d[2] - y[2] * d[1], y[2] * d[0] - y[0] * d[2], y[0] * d[1] - y[1] * d[0]];
      const w = 1 + c, n = Math.hypot(ax[0], ax[1], ax[2], w); q = { x: ax[0] / n, y: ax[1] / n, z: ax[2] / n, w: w / n }; }
    const desc = RAPIER.ColliderDesc.cylinder(half, Math.max(radius - SHRINK, 0.002))
      .setTranslation(o[0] + d[0] * mid, o[1] + d[1] * mid, o[2] + d[2] * mid).setRotation(q)
      .setDensity(ROLLER_DENSITY).setFriction(j.shape.friction ?? ROLLER_FRICTION)
      .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max).setCollisionGroups(MECH_GROUPS)
      .setActiveHooks(RAPIER.ActiveHooks.FILTER_CONTACT_PAIRS);
    const col = world.createCollider(desc, b.rb);
    robotColliders.add(col.handle);
    colliderPart.set(col.handle, b.parts[0]);
  }

  // ---------------- build ----------------
  // parts:  [{key, geomId, positions(Float32Array local triangle soup), rest(16 col-major)}]
  // joints: [{id, type:'pivot'|'slider', axis:{origin,dir}, members:[keys], rest, min, max,
  //           continuous, unit:'deg'|'m'}]   (values in the joint's own units)
  function build(parts, joints) {
    const js = joints.filter(j => j.members && j.members.length && j.axis)
                     .map(j => ({ ...j, set: new Set(j.members) }))
                     .sort((a, b) => a.members.length - b.members.length);
    // each part belongs to the SMALLEST joint containing it; a joint's parent is the smallest
    // bigger joint containing all of its parts (a roller on an arm), else the chassis.
    const owner = new Map();
    for (const j of js) for (const k of j.members) if (!owner.has(k)) owner.set(k, j.id);
    for (const j of js) {
      const p = js.find(o => o !== j && o.members.length > j.members.length
                             && j.members.every(k => o.set.has(k)));
      j.parent = p ? p.id : 'chassis';
    }
    const mk = (id, dynamic) => {
      // Soft CCD: an intake tip moves ~1 cm per step -- more than a polycarbonate wall is thick --
      // so contacts are predicted SOFT_CCD ahead; otherwise a mechanism can tunnel through a thin
      // plate and end up wedged on its far side. (Full CCD is far too slow on 100+ shape bodies.)
      // mechanisms may sleep when idle (so resting game pieces touching them can sleep too);
      // every command wakes them -- see wake()
      const rb = world.createRigidBody(dynamic ? RAPIER.RigidBodyDesc.dynamic()
                                                   .setSoftCcdPrediction(SOFT_CCD)
                                               : RAPIER.RigidBodyDesc.fixed());
      bodies.set(id, { id, rb, parts: [], dynamic, extraMass: 0 });
    };
    mk('chassis', false);
    for (const j of js) mk(j.id, true);
    // Rollers with a fitted shape get ONE exact cylinder instead of a hull per part: a hull built
    // from mesh vertices is a slightly lumpy polygon, and at 60-100 rps that lump acts like a cam,
    // kicking each ball differently (drops 2 mm apart ended up metres apart).
    const shaped = new Map(js.filter(j => j.shape && j.shape.type === 'cylinder').map(j => [j.id, j]));
    for (const p of parts) {
      const bid = owner.get(p.key) || 'chassis';
      attachPart(bid, p.key, p.geomId, p.positions, p.rest, bid !== 'chassis', shaped.has(bid));
    }
    for (const j of shaped.values()) addCylinder(bodies.get(j.id), j);
    for (const b of bodies.values()) if (b.dynamic) {
      const m = b.rb.mass();
      if (b.extraMass || m < 0.05) b.rb.setAdditionalMass(Math.max(b.extraMass, m < 0.05 ? 0.5 : 0), true);
    }

    // joints (created parent-first so mass props are final)
    const byId = new Map(js.map(j => [j.id, j]));
    for (const j of [...js].reverse()) {
      const parent = bodies.get(j.parent).rb, child = bodies.get(j.id).rb;
      const o = j.axis.origin, d = norm(j.axis.dir);
      const a = { x: o[0], y: o[1], z: o[2] }, ax = { x: d[0], y: d[1], z: d[2] };
      const data = j.type === 'slider' ? RAPIER.JointData.prismatic(a, a, ax)
                                       : RAPIER.JointData.revolute(a, a, ax);
      const joint = world.createImpulseJoint(data, parent, child, true);
      joint.configureMotorModel(RAPIER.MotorModel.ForceBased);
      const k = toSim(j, 1) - toSim(j, 0);   // unit conversion factor
      if (!j.continuous && j.min != null && j.max != null)
        joint.setLimits(Math.min(j.min, j.max) * k - j.rest * k, Math.max(j.min, j.max) * k - j.rest * k);
      // effective inertia/mass about the joint for this body AND everything riding on it
      let inertia = 0;
      const subtree = [j.id, ...js.filter(o => isDescendant(o, j, byId)).map(o => o.id)];
      for (const id of subtree) {
        const rb = bodies.get(id).rb, mass = rb.mass(), c = rb.worldCom();
        if (j.type === 'slider') inertia += mass;
        else {
          const r = [c.x - o[0], c.y - o[1], c.z - o[2]], t = r[0] * d[0] + r[1] * d[1] + r[2] * d[2];
          const perp2 = r[0] ** 2 + r[1] ** 2 + r[2] ** 2 - t * t;
          inertia += mass * (perp2 + 0.0025);
        }
      }
      // headroom: size the motor for PAYLOAD kg riding on the mechanism too (pieces on an arm)
      if (j.type === 'slider') inertia += PAYLOAD;
      else {
        const c = bodies.get(j.id).rb.worldCom(), r = [c.x - o[0], c.y - o[1], c.z - o[2]];
        const t = r[0] * d[0] + r[1] * d[1] + r[2] * d[2];
        inertia += PAYLOAD * Math.max(r[0] ** 2 + r[1] ** 2 + r[2] ** 2 - t * t, 0.01);
      }
      const w = 2 * Math.PI * MOTOR_HZ;
      j.stiffness = inertia * w * w; j.damping = 2 * inertia * w; j.inertia = inertia;
      j.joint = joint; j.k = k;
      bodies.get(j.id).joint = j;
      if (j.free) applyFree(j);
      else if (j.continuous) joint.configureMotorVelocity(0, inertia * 30);
      else joint.configureMotorPosition(0, j.stiffness, j.damping);
    }
    sim.joints = byId;
    calibrateContacts();
  }

  function isDescendant(o, j, byId) {
    let p = o.parent;
    while (p && p !== 'chassis') { if (p === j.id) return true; p = byId.get(p)?.parent; }
    return false;
  }

  // Robot parts already touching in the CAD never collide with each other.
  function calibrateContacts() {
    const cand = [];
    const calHooks = {
      filterContactPair(c1, c2) {
        if (robotColliders.has(c1) && robotColliders.has(c2)) { cand.push([c1, c2]); return RAPIER.SolverFlags.EMPTY; }
        return RAPIER.SolverFlags.COMPUTE_IMPULSE;
      },
      filterIntersectionPair() { return true; },
    };
    const g = world.gravity; world.gravity = { x: 0, y: 0, z: 0 };
    world.timestep = 1e-6;
    world.step(events, calHooks);
    for (const [h1, h2] of cand) {
      let touching = false;
      world.contactPair(world.getCollider(h1), world.getCollider(h2), (man) => {
        for (let i = 0; i < man.numContacts(); i++) if (man.contactDist(i) < overlapTol) touching = true;
      });
      if (touching) { disabled.add(pairKey(h1, h2)); cadTouching.add(pairKey(h1, h2)); }
    }
    world.gravity = g; world.timestep = DT;
    sim.stats = { candidates: cand.length, ignoredTouching: disabled.size, colliders: robotColliders.size,
                  chassisCollide, selfCollide: chassisCollide };
  }
  // A deep overlap (beyond what the solver allows a real contact) between robot parts is taken as
  // an approximation error -- a part nesting into a notch/pocket that a hull fills in -- and that
  // pair becomes a ghost until it separates. Only where an approximation is involved: pairs that
  // touched in the CAD pose, or a CONCAVE (chunk-approximated) mechanism part. Two convex parts
  // are modelled exactly, so they stay solid however hard they're pushed: no phasing.
  function ghostIfDeep(c1, c2) {
    const key = pairKey(c1.handle, c2.handle);
    if (disabled.has(key)) return;
    let deep = false;
    world.contactPair(c1, c2, (man) => {
      for (let k = 0; k < man.numContacts(); k++) if (man.contactDist(k) < -DEEP) deep = true;
    });
    if (deep) disabled.add(key);
  }
  function reghostDeep() {
    for (const key of cadTouching) {
      if (disabled.has(key)) continue;
      const [h1, h2] = key.split(':').map(Number);
      const c1 = world.getCollider(h1), c2 = world.getCollider(h2);
      if (c1 && c2) ghostIfDeep(c1, c2);
    }
    for (const h of approxMech) {
      const c = world.getCollider(h);
      world.contactPairsWith(c, (o) => {
        if (robotColliders.has(o.handle) && o.parent()?.handle !== c.parent()?.handle) ghostIfDeep(c, o);
      });
    }
  }
  // ghost pairs that have moved apart become solid again
  function releaseSeparated() {
    for (const key of disabled) {
      const [h1, h2] = key.split(':').map(Number);
      const c1 = world.getCollider(h1), c2 = world.getCollider(h2);
      if (!c1 || !c2) { disabled.delete(key); continue; }
      let touching = false;
      world.contactPair(c1, c2, (man) => {
        for (let i = 0; i < man.numContacts(); i++) if (man.contactDist(i) < TOUCH_TOL) touching = true;
      });
      if (!touching) disabled.delete(key);
    }
  }

  // ---------------- control ----------------
  function toSim(j, v) { return j.type === 'slider' ? v : v * Math.PI / 180; }
  // Targets are ramped at a mechanism-like top speed (instead of snapping), so a fast slider
  // drag swings the arm the way a real motor would rather than teleporting its setpoint.
  const MAX_RATE = { pivot: Math.PI, slider: 1.0 };   // rad/s, m/s
  function applyFree(j) { j.joint.configureMotor(0, 0, 0, j.inertia * FREE_DAMPING); }
  // free = no motor: the joint is moved only by contacts, gravity and its limits
  function setFree(id, free) {
    const j = sim.joints.get(id); if (!j) return;
    j.free = !!free; wake(id);
    if (j.free) { applyFree(j); return; }
    j.cur = undefined; j.goal = undefined;          // re-grab from wherever it got pushed to
    const now = (jointValue(id) - j.rest) * j.k;
    if (j.continuous) j.joint.configureMotorVelocity(0, j.inertia * 30);
    else { j.cur = j.goal = now; j.joint.configureMotorPosition(now, j.stiffness, j.damping); }
  }
  function wake(id) { const b = bodies.get(id); if (b && b.dynamic) b.rb.wakeUp(); }
  function setTarget(id, value) {
    const j = sim.joints.get(id); if (!j || j.continuous || j.free) return;
    wake(id);
    j.goal = (value - j.rest) * j.k;
    if (j.cur === undefined) j.cur = 0;
  }
  function rampTargets() {
    for (const j of sim.joints.values()) {
      if (j.continuous || j.free || j.goal === undefined || j.cur === j.goal) continue;
      const rate = MAX_RATE[j.type === 'slider' ? 'slider' : 'pivot'];
      const d = j.goal - j.cur, s = rate * DT;
      j.cur = Math.abs(d) <= s ? j.goal : j.cur + Math.sign(d) * s;
      const vel = j.cur === j.goal ? 0 : Math.sign(d) * rate;
      j.joint.configureMotor(j.cur, vel, j.stiffness, j.damping);
      wake(j.id);
    }
  }
  function setSpin(id, degPerSec) {
    const j = sim.joints.get(id); if (!j || j.free) return;
    wake(id);
    j.joint.configureMotorVelocity(degPerSec * Math.PI / 180, j.inertia * 30);
    j.spinning = degPerSec !== 0;
  }
  function addFloor(z) {
    const rb = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    world.createCollider(RAPIER.ColliderDesc.cuboid(20, 20, 0.05).setTranslation(0, 0, z - 0.05).setFriction(0.9)
      .setCollisionGroups(groups(G_FLOOR, 0xffff)), rb);
  }
  // Game pieces are FOAM: a rigid core COMPRESS smaller than the ball (so nothing can squash it
  // flat) inside a springy shell handled by foamForces(). Squeezing a ball between rollers and a
  // wall presses it into the rollers far harder than its weight -- that pressure is what lets
  // grip tape actually move it. A rigid ball only ever presses with its own ~2 N.
  function spawnPiece(pos, radius, mass = 0.215) {
    const rb = world.createRigidBody(RAPIER.RigidBodyDesc.dynamic().setTranslation(pos[0], pos[1], pos[2])
      .setCcdEnabled(true).setLinearDamping(0.05).setAngularDamping(0.2));
    const vol = 4 / 3 * Math.PI * radius ** 3;
    const core = foam ? Math.max(radius - COMPRESS, radius * 0.5) : radius;
    const col = world.createCollider(RAPIER.ColliderDesc.ball(core).setFriction(PIECE_FRICTION)
      .setRestitution(foam ? 0.1 : 0.35).setCollisionGroups(groups(G_PIECE, 0xffff))
      .setMassProperties(mass, { x: 0, y: 0, z: 0 },                       // solid foam ball inertia
                         { x: 0.4 * mass * radius ** 2, y: 0.4 * mass * radius ** 2, z: 0.4 * mass * radius ** 2 },
                         { x: 0, y: 0, z: 0, w: 1 }), rb);
    const p = { id: `piece${pieceSeq++}`, rb, radius, mass, shell: new RAPIER.Ball(radius), col: col.handle };
    pieces.push(p); pieceByCollider.set(col.handle, p);
    if (foam) {
      // Ball-to-ball contact uses a massless near-full-size ball that only touches other balls,
      // through Rapier's own (implicit, stable) contact solver. Foam springs between balls made a
      // pile buzz forever -- explicit springs stacked several deep never settle or sleep.
      const outer = world.createCollider(RAPIER.ColliderDesc.ball(radius - 0.002).setDensity(0)
        .setFriction(PIECE_FRICTION).setRestitution(0.1).setCollisionGroups(groups(G_SHELL, G_SHELL)), rb);
      p.outer = outer.handle; pieceByCollider.set(outer.handle, p);
    }
    return p;
  }
  const IDENT_Q = { x: 0, y: 0, z: 0, w: 1 };
  const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
  function velAt(rb, pt) {      // velocity of a rigid body's material at a world point
    if (!rb || !rb.isDynamic()) return { x: 0, y: 0, z: 0 };
    const v = rb.linvel(), w = rb.angvel(), c = rb.worldCom();
    const r = { x: pt.x - c.x, y: pt.y - c.y, z: pt.z - c.z }, wr = cross(w, r);
    return { x: v.x + wr.x, y: v.y + wr.y, z: v.z + wr.z };
  }
  function contactMu(col) {
    const f = col.friction(), rule = col.frictionCombineRule();
    if (rule === RAPIER.CoefficientCombineRule.Max) return Math.max(f, PIECE_FRICTION);
    if (rule === RAPIER.CoefficientCombineRule.Min) return Math.min(f, PIECE_FRICTION);
    if (rule === RAPIER.CoefficientCombineRule.Multiply) return f * PIECE_FRICTION;
    return (f + PIECE_FRICTION) / 2;
  }
  // Foam shell: for everything the full-size ball overlaps, a spring normal force from how far
  // it's squashed, plus friction against the slip between the ball's surface and the other
  // surface (a spinning roller drags it along). This is what lets grip tape move a squeezed ball.
  //  * contacts whose normals agree are merged (deepest wins): a ball on a plate split into chunks,
  //    or a plate + its screws, would otherwise feel several stacked springs;
  //  * normal damping may stop the approach but never push back out (no overshoot);
  //  * friction is solved for ALL of a ball's contacts together (a few Gauss-Seidel passes on the
  //    ball's own velocity, each contact clamped to mu * its normal impulse). Sizing each contact
  //    to cancel the whole slip on its own over-corrects with 2+ contacts and makes balls rattle.
  // Other bodies are treated as much heavier than the ball (rollers are motor-driven, the chassis
  // is fixed); they still receive the equal and opposite impulses.
  // (Impulses are applied AFTER the queries: Rapier holds the world borrowed during
  // intersectionsWithShape, and mutating a body inside the callback throws -- silently.)
  function foamForces() {
    const queued = [];
    for (const p of pieces) {
      if (p.rb.isSleeping()) continue;          // a resting ball may sleep; a bump to its core wakes it
      const c = p.rb.translation();
      const found = [];
      world.intersectionsWithShape(c, IDENT_Q, p.shell, (col) => {
        if (pieceByCollider.has(col.handle)) return true;        // balls: handled by their outer colliders
        const other = null;
        const ct = col.contactShape(p.shell, c, IDENT_Q, 0);
        if (!ct || ct.distance >= 0) return true;
        found.push({ col, other, pen: -ct.distance, n: { ...ct.normal1 }, pt: { ...ct.point1 } });
        return true;
      });
      if (!found.length) continue;
      found.sort((a, b) => b.pen - a.pen);
      const cs = [];
      for (const f of found) {
        if (!f.other && cs.some(m => !m.other && m.n.x * f.n.x + m.n.y * f.n.y + m.n.z * f.n.z > MERGE_COS)) continue;
        cs.push(f);
      }
      // the ball's velocity, updated locally as impulses are decided
      const m = p.mass, invI = 1 / (0.4 * m * p.radius * p.radius);
      const lv = p.rb.linvel(), av = p.rb.angvel();
      const v = { x: lv.x, y: lv.y, z: lv.z }, w = { x: av.x, y: av.y, z: av.z };
      const push = (f, J) => {         // apply impulse J at f.pt to the local ball state
        v.x += J.x / m; v.y += J.y / m; v.z += J.z / m;
        const t = cross(f.r, J); w.x += t.x * invI; w.y += t.y * invI; w.z += t.z * invI;
      };
      const slip = (f) => {            // ball surface velocity minus the other surface's, at the contact
        const wr = cross(w, f.r);
        return { x: v.x + wr.x - f.vo.x, y: v.y + wr.y - f.vo.y, z: v.z + wr.z - f.vo.z };
      };
      for (const f of cs) {
        f.r = { x: f.pt.x - c.x, y: f.pt.y - c.y, z: f.pt.z - c.z };
        f.ob = f.col.parent();
        f.vo = velAt(f.ob, f.pt);
        f.mu = f.other ? PIECE_FRICTION : contactMu(f.col);
        f.jt = { x: 0, y: 0, z: 0 };
        // normal impulse: foam spring + damping that can at most cancel the approach speed
        const pen = Math.min(f.pen, (f.other ? 2 : 1) * COMPRESS * 1.2);
        const k = f.other ? FOAM_K / 2 : FOAM_K;                 // two foam springs in series
        const vr = slip(f), vn = vr.x * f.n.x + vr.y * f.n.y + vr.z * f.n.z;
        const mN = f.other ? m / 2 : m;
        f.jn = k * pen * DT + (vn < 0 ? Math.min(FOAM_C * -vn * DT, mN * -vn) : 0);
        push(f, { x: f.n.x * f.jn, y: f.n.y * f.jn, z: f.n.z * f.jn });
      }
      const mT = m / 3.5;                // solid sphere: impulse at its surface vs. surface speed change
      for (let it = 0; it < FRICTION_PASSES; it++) {
        for (const f of cs) {
          const vr = slip(f), vn = vr.x * f.n.x + vr.y * f.n.y + vr.z * f.n.z;
          const vt = { x: vr.x - vn * f.n.x, y: vr.y - vn * f.n.y, z: vr.z - vn * f.n.z };
          // impulse that would stop this contact's slip, added to what it already applies,
          // then clamped to the Coulomb limit
          const mt = f.other ? mT / 2 : mT;
          let nx = f.jt.x - vt.x * mt, ny = f.jt.y - vt.y * mt, nz = f.jt.z - vt.z * mt;
          const mag = Math.hypot(nx, ny, nz), lim = f.mu * f.jn;
          if (mag > lim) { const s2 = lim / mag; nx *= s2; ny *= s2; nz *= s2; }
          const d = { x: nx - f.jt.x, y: ny - f.jt.y, z: nz - f.jt.z };
          f.jt = { x: nx, y: ny, z: nz };
          push(f, d);
        }
      }
      for (const f of cs) {
        const J = { x: f.n.x * f.jn + f.jt.x, y: f.n.y * f.jn + f.jt.y, z: f.n.z * f.jn + f.jt.z };
        queued.push([p.rb, J, f.pt, false]);                    // don't keep resting balls awake
        if (f.ob && f.ob.isDynamic()) queued.push([f.ob, { x: -J.x, y: -J.y, z: -J.z }, f.pt, true]);
        // rolling resistance: foam deforms as it rolls, so rolling over a surface costs energy.
        // Without it a ball between two parallel rollers rolls along them forever, bouncing
        // wall to wall ("jostling" with every roller stopped).
        const wo = f.ob && f.ob.isDynamic() ? f.ob.angvel() : { x: 0, y: 0, z: 0 };
        const wr = { x: w.x - wo.x, y: w.y - wo.y, z: w.z - wo.z };
        const wn = wr.x * f.n.x + wr.y * f.n.y + wr.z * f.n.z;   // spin about the normal: not rolling
        const roll = { x: wr.x - wn * f.n.x, y: wr.y - wn * f.n.y, z: wr.z - wn * f.n.z };
        const rmag = Math.hypot(roll.x, roll.y, roll.z);
        if (rmag > 1e-6) {
          const cap = (0.4 * m * p.radius * p.radius) * rmag / cs.length;   // at most stop it (shared)
          const L = Math.min(ROLL_RESIST * f.jn * p.radius, cap) / rmag;
          queued.push([p.rb, { x: -roll.x * L, y: -roll.y * L, z: -roll.z * L }, null, false]);
        }
      }
    }
    for (const [rb, J, pt, wake] of queued) {
      if (pt) rb.applyImpulseAtPoint(J, pt, wake); else rb.applyTorqueImpulse(J, wake);
    }
  }
  function removePiece(p) {
    const i = pieces.indexOf(p); if (i >= 0) pieces.splice(i, 1);
    pieceByCollider.delete(p.col); pieceByCollider.delete(p.outer);
    world.removeRigidBody(p.rb);
  }
  function step(dt) {
    // at most MAX_SUBSTEPS per frame: if the machine can't keep up, run in slow motion instead of
    // spiralling (each catch-up step making the next frame even slower)
    acc = Math.min(acc + dt, DT * MAX_SUBSTEPS);
    let n = 0;
    while (acc >= DT) {
      rampTargets(); if (foam && pieces.length) foamForces(); world.step(events, hooks); acc -= DT; n++;
      if (++stepCount % 6 === 0) {
        if (releaseGhosts && disabled.size) releaseSeparated();
        reghostDeep();
      }
    }
    for (const p of [...pieces]) if (p.rb.translation().z < -10) removePiece(p);   // fell off
    return n;
  }
  // pose of a body as {t:{x,y,z}, q:{x,y,z,w}}
  function bodyPose(id) { const rb = bodies.get(id)?.rb; return rb ? { t: rb.translation(), q: rb.rotation() } : null; }
  function jointValue(id) {   // current value in the joint's units (for the UI readout)
    const j = sim.joints.get(id); if (!j) return null;
    const rb = bodies.get(id).rb;
    if (j.type === 'slider') { const t = rb.translation(), d = norm(j.axis.dir); return j.rest + (t.x * d[0] + t.y * d[1] + t.z * d[2]); }
    const q = rb.rotation(), d = norm(j.axis.dir);
    const s = q.x * d[0] + q.y * d[1] + q.z * d[2];
    return j.rest + 2 * Math.atan2(s, q.w) * 180 / Math.PI;
  }
  function free() { world.free(); }

  // Is a ball of radius r at pos clear of the robot, and boxed in by it? "Boxed in" = a ray
  // straight down and rays in the four horizontal directions all hit robot geometry within
  // reach -- i.e. the spot is inside a container (the hopper), not out in open air.
  const isRobot = (col) => robotColliders.has(col.handle);
  function probe(pos, r, reach = 1.0) {
    const c = { x: pos[0], y: pos[1], z: pos[2] };
    let clear = true;
    world.intersectionsWithShape(c, IDENT_Q, new RAPIER.Ball(r), () => { clear = false; return false; },
      undefined, undefined, undefined, undefined, isRobot);
    if (!clear) return { clear: false, enclosed: false };
    const dirs = [[0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]];
    const enclosed = dirs.every(d => world.castRay(new RAPIER.Ray(c, { x: d[0], y: d[1], z: d[2] }), reach, true,
      undefined, undefined, undefined, undefined, isRobot) !== null);
    return { clear, enclosed };
  }

  const sim = { world, build, setTarget, setSpin, setFree, addFloor, spawnPiece, removePiece, probe, step, bodyPose,
                jointValue, partBody, colliderPart, bodies, pieces, joints: new Map(), stats: null, free };
  return sim;
}

function norm(v) { const n = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / n, v[1] / n, v[2] / n]; }
