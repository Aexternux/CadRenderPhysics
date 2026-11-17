// Offline test for web/physics.js (run: node web/test_physics.mjs, needs @dimforge/rapier3d-compat
// resolvable, e.g. NODE_PATH or a local node_modules). Uses boxes instead of CAD meshes.
import RAPIER from '@dimforge/rapier3d-compat';
import { createPhysics } from './physics.js';

await RAPIER.init();
let failures = 0;
const check = (cond, msg) => { console.log(`${cond ? 'PASS' : 'FAIL'}  ${msg}`); if (!cond) failures++; };

// triangle soup of an axis-aligned box (local coords)
function box([x0, y0, z0], [x1, y1, z1]) {
  const v = [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]];
  const f = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]];
  return new Float32Array(f.flatMap(t => t.flatMap(i => v[i])));
}
const I = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
const part = (key, lo, hi) => ({ key, geomId: key, positions: box(lo, hi), rest: I });
const run = (sim, secs) => { for (let t = 0; t < secs; t += 1 / 60) sim.step(1 / 60); };

// 1. revolute sign + limits: vertical axis, arm along +x; +90 deg must swing it to +y
{
  const sim = createPhysics(RAPIER);
  sim.build([part('base', [-0.1,-0.1,0], [0.1,0.1,0.4]), part('arm', [0.12,-0.02,0.42], [0.5,0.02,0.46])],
            [{ id: 'pivot', type: 'pivot', axis: { origin: [0,0,0.44], dir: [0,0,1] }, members: ['arm'],
               rest: 0, min: -180, max: 180 }]);
  sim.addFloor(0);
  sim.setTarget('pivot', 90); run(sim, 2);
  const c = sim.bodies.get('pivot').rb.worldCom();
  check(Math.abs(sim.jointValue('pivot') - 90) < 3, `revolute reaches +90 deg (got ${sim.jointValue('pivot').toFixed(1)})`);
  check(c.y > 0.2 && Math.abs(c.x) < 0.05, `+angle is right-handed about +axis (arm COM -> +y: ${c.x.toFixed(2)}, ${c.y.toFixed(2)})`);
  sim.free();
}

// 2. gravity hold: horizontal axis, arm sticking out along +x must not sag much at target 0
{
  const sim = createPhysics(RAPIER);
  sim.build([part('base', [-0.1,-0.1,0], [0.1,0.1,0.4]), part('arm', [0.12,-0.02,0.38], [0.6,0.02,0.42])],
            [{ id: 'arm', type: 'pivot', axis: { origin: [0.11,0,0.4], dir: [0,1,0] }, members: ['arm'],
               rest: 0, min: -90, max: 90 }]);
  run(sim, 2);
  check(Math.abs(sim.jointValue('arm')) < 5, `motor holds the arm against gravity (sag ${sim.jointValue('arm').toFixed(1)} deg)`);
  // 3. limits: ask beyond the CAD limit
  sim.setTarget('arm', -170); run(sim, 2);
  check(sim.jointValue('arm') > -95, `CAD limit respected (asked -170, got ${sim.jointValue('arm').toFixed(1)})`);
  sim.free();
}

// 4. pushing: a swinging paddle shoves a ball across the floor
{
  const sim = createPhysics(RAPIER);
  sim.build([part('post', [-0.03,-0.03,0], [0.03,0.03,0.3]), part('paddle', [0.04,-0.01,0.0], [0.6,0.01,0.25])],
            [{ id: 'swing', type: 'pivot', axis: { origin: [0,0,0.15], dir: [0,0,1] }, members: ['paddle'],
               rest: 0, min: -180, max: 180 }]);
  sim.addFloor(0);
  const ball = sim.spawnPiece([0.4, 0.12, 0.075], 0.075);
  run(sim, 0.5);
  const before = ball.rb.translation();
  sim.setTarget('swing', 80); run(sim, 2);
  const after = ball.rb.translation();
  const moved = Math.hypot(after.x - before.x, after.y - before.y);
  check(moved > 0.1, `moving paddle pushes the ball (moved ${moved.toFixed(2)} m)`);
  check(after.z > 0.05, `ball stays on the floor, not through it (z ${after.z.toFixed(3)})`);
  sim.free();
}

// 5. slider sign + (with self-collision on) parts touching in the CAD don't jam or explode
{
  const sim = createPhysics(RAPIER, { selfCollide: true });
  // carriage overlaps the rail at rest (like a bearing block on a tube)
  sim.build([part('rail', [-0.5,-0.02,0.2], [0.5,0.02,0.24]), part('carriage', [-0.06,-0.04,0.19], [0.06,0.04,0.25])],
            [{ id: 'slide', type: 'slider', axis: { origin: [0,0,0.22], dir: [1,0,0] }, members: ['carriage'],
               rest: 0, min: -0.3, max: 0.3 }]);
  run(sim, 0.5);
  const still = sim.bodies.get('slide').rb.translation();
  check(Math.hypot(still.x, still.y, still.z) < 0.01, `overlapping CAD parts stay put (drift ${Math.hypot(still.x, still.y, still.z).toFixed(4)} m)`);
  check(sim.stats.ignoredTouching >= 1, `touching pair detected and ignored (${sim.stats.ignoredTouching})`);
  sim.setTarget('slide', 0.2); run(sim, 2);
  check(Math.abs(sim.jointValue('slide') - 0.2) < 0.02, `slider reaches +0.2 m along +axis (got ${sim.jointValue('slide').toFixed(3)})`);
  sim.free();
}

// 6. spinning roller drags a ball resting on it sideways
{
  const sim = createPhysics(RAPIER);
  sim.build([part('frame', [-0.3,-0.3,0], [0.3,-0.25,0.1]), part('roller', [-0.2,-0.04,0.06], [0.2,0.04,0.14])],
            [{ id: 'roll', type: 'pivot', axis: { origin: [0,0,0.1], dir: [1,0,0] }, members: ['roller'],
               rest: 0, continuous: true }]);
  sim.addFloor(0);
  const ball = sim.spawnPiece([0, 0, 0.14 + 0.076], 0.075);
  sim.setSpin('roll', 720);
  let far = 0;   // a box "roller" knocks the ball around, so track the farthest it got pushed
  for (let t = 0; t < 1.5; t += 1 / 60) { sim.step(1 / 60); far = Math.max(far, Math.abs(ball.rb.translation().y)); }
  check(far > 0.05, `spinning roller pushes the ball off (max |y| ${far.toFixed(2)} m)`);
  sim.free();
}

// 7. mechanism pushes mechanism: a swinging arm shoves a FREE slider (like an intake and a hopper)
{
  const sim = createPhysics(RAPIER);
  sim.build([part('base', [-0.6,-0.6,0], [0.6,0.6,0.02]),
             part('arm', [0.03,-0.02,0.10], [0.45,0.02,0.16]),
             part('wall', [0.25,0.08,0.05], [0.45,0.10,0.30])],
            [{ id: 'arm', type: 'pivot', axis: { origin: [0,0,0.13], dir: [0,0,1] }, members: ['arm'],
               rest: 0, min: -10, max: 100 },
             { id: 'wall', type: 'slider', axis: { origin: [0.35,0.09,0.2], dir: [0,1,0] }, members: ['wall'],
               rest: 0, min: 0, max: 0.4, free: true }]);
  run(sim, 0.3);
  sim.setTarget('arm', 30); run(sim, 2);     // arm ends up parked across the wall's track
  check(sim.jointValue('wall') > 0.05, `swinging arm pushes the free slider (moved ${sim.jointValue('wall').toFixed(3)} m)`);
  check(sim.jointValue('arm') > 25, `arm keeps moving while pushing (at ${sim.jointValue('arm').toFixed(1)} deg)`);
  // the wall is solid to the arm: it never ends up on the arm's far side
  sim.setFree('wall', false); sim.setTarget('wall', 0); run(sim, 2);
  // both are motor-driven, so the wall may shove the arm back -- but they must never overlap:
  // the wall's near face (y = 0.08 + v) has to stay on the far side of the arm at x = 0.25
  const a = sim.jointValue('arm') * Math.PI / 180, v = sim.jointValue('wall');
  const armY = 0.25 * Math.tan(a) + 0.02 / Math.cos(a);
  check(0.08 + v >= armY - 0.01, `driven slider stops at the arm, no phasing (wall face ${(0.08 + v).toFixed(3)} m, arm edge ${armY.toFixed(3)} m, arm ${(a * 180 / Math.PI).toFixed(1)} deg)`);
  sim.free();
}

// 8. a fast roller with an exact cylinder shape treats nearly identical drops identically
{
  const ends = [];
  for (const dx of [-0.001, 0, 0.001]) {
    const sim = createPhysics(RAPIER);
    // roller part is a coarse box, but the joint's cylinder shape replaces its collider
    sim.build([part('frame', [-0.4,-0.4,0], [0.4,-0.35,0.05]), part('roller', [-0.2,-0.03,0.17], [0.2,0.03,0.23])],
              [{ id: 'roll', type: 'pivot', axis: { origin: [0,0,0.2], dir: [1,0,0] }, members: ['roller'],
                 rest: 0, continuous: true, shape: { type: 'cylinder', radius: 0.03, tmin: -0.2, tmax: 0.2 } }]);
    sim.addFloor(0);
    sim.setSpin('roll', 60 * 360);                 // 60 rps
    const ball = sim.spawnPiece([dx, 0.02, 0.35], 0.075);
    let far = 0;
    for (let k = 0; k < 90; k++) { sim.step(1 / 60); far = Math.max(far, Math.abs(ball.rb.translation().y)); }
    const t = ball.rb.translation(); ends.push([t.x, t.y, far]); sim.free();
  }
  const spread = Math.max(...ends.map(e => Math.hypot(e[0] - ends[1][0], e[1] - ends[1][1])));
  check(spread < 0.05, `cylinder roller at 60 rps: drops 1 mm apart land together (spread ${spread.toFixed(3)} m)`);
  check(ends[1][2] > 0.1, `...and the spinning roller actually throws the ball (max |y| ${ends[1][2].toFixed(2)} m)`);
}

console.log(failures ? `\n${failures} FAILED` : '\nALL PHYSICS CHECKS PASSED');
process.exit(failures ? 1 : 0);
