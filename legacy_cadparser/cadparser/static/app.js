import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';

const state = { model: null, values: {}, objects: new Map(), available: new Set(), geometry: new Map(), token: 0, poller: null };
const assembly = document.querySelector('#assembly'), viewer = document.querySelector('#viewer'), meshButton = document.querySelector('#load-meshes');
const scene = new THREE.Scene(); scene.background = new THREE.Color('#0d1117');
const camera = new THREE.PerspectiveCamera(45, 1, .001, 500); camera.position.set(2, 2, 2);
const renderer = new THREE.WebGLRenderer({ antialias: true }); renderer.setPixelRatio(devicePixelRatio); renderer.setSize(viewer.clientWidth, viewer.clientHeight); renderer.outputColorSpace = THREE.SRGBColorSpace; viewer.append(renderer.domElement);
scene.add(new THREE.HemisphereLight(0xffffff, 0x38404a, 2));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4); keyLight.position.set(3, 5, 4); scene.add(keyLight);
const controls = new OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
const grid = new THREE.GridHelper(3, 30, 0x45515e, 0x242c35); grid.rotation.x = Math.PI / 2; scene.add(grid);
const axes = new THREE.AxesHelper(.25); scene.add(axes);
const cadMaterial = new THREE.MeshStandardMaterial({ color: 0xb7c1cc, metalness: .18, roughness: .62, side: THREE.DoubleSide });

async function request(url, options) { const response = await fetch(url, options); const body = await response.text(); let data; try { data = JSON.parse(body); } catch { throw Error(`Server returned ${response.status}. Restart the CadParser viewer.`); } if (!response.ok) throw Error(data.error || response.statusText); return data; }
function partKey(part) { return part.path.join(' / '); }
function applyPose(object, pose) { object.matrix.copy(new THREE.Matrix4().set(...pose.flat())); object.matrixAutoUpdate = false; }
function render() { controls.update(); renderer.render(scene, camera); requestAnimationFrame(render); }
function removePartObject(key) { const old = state.objects.get(key); if (old) scene.remove(old); }
function markerFor(part) { const object = new THREE.Mesh(new THREE.BoxGeometry(.025, .025, .025), new THREE.MeshStandardMaterial({ color: 0x58a6ff })); object.userData = { partObject: true, part, marker: true }; applyPose(object, part.pose); return object; }
function drawMarkers() { for (const part of state.model.parts) { const key = partKey(part), object = markerFor(part); scene.add(object); state.objects.set(key, object); } fitCamera(); }
function geometryFor(mesh, loader) { if (!state.geometry.has(mesh)) state.geometry.set(mesh, loader.loadAsync(`/assets/meshes/${mesh}.stl`).then(geometry => { geometry.computeVertexNormals(); return geometry; })); return state.geometry.get(mesh); }
async function addAvailableMeshes() {
  if (!state.model) return;
  const token = state.token, loader = new STLLoader();
  const parts = state.model.parts.filter(part => part.mesh && state.available.has(part.mesh) && state.objects.get(partKey(part))?.userData.marker);
  let next = 0;
  async function worker() { while (next < parts.length) { const part = parts[next++]; try { const geometry = await geometryFor(part.mesh, loader); if (token !== state.token) return; const key = partKey(part), object = new THREE.Mesh(geometry, cadMaterial); object.userData = { partObject: true, part, marker: false }; applyPose(object, part.pose); removePartObject(key); scene.add(object); state.objects.set(key, object); } catch (error) { console.warn(`Mesh missing for ${partKey(part)}`, error); } } }
  await Promise.all(Array.from({ length: 6 }, worker));
  if (token === state.token) { fitCamera(); updateMeshButton(); }
}
function updateMeshButton(progress) { if (!state.model) return; const total = state.model.meshes.total, cached = state.available.size; if (progress && !progress.done) { meshButton.textContent = `Downloading CAD: ${progress.completed}/${progress.total}`; return; } meshButton.disabled = cached >= total; meshButton.textContent = cached >= total ? `CAD rendered (${total} meshes)` : cached ? `Download remaining CAD (${cached}/${total})` : `Download and render CAD (${total} meshes)`; }
function fitCamera() { const box = new THREE.Box3(); for (const object of state.objects.values()) box.expandByObject(object); if (box.isEmpty()) return; const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()); controls.target.copy(center); const distance = Math.max(size.x, size.y, size.z, .5) * 1.8; camera.position.copy(center).add(new THREE.Vector3(distance, distance, distance)); camera.near = distance / 1000; camera.far = distance * 100; camera.updateProjectionMatrix(); controls.update(); }
function showModel(model) {
  state.token++; state.model = model; state.values = {}; state.available = new Set(model.meshes.cached); state.geometry.clear(); state.objects.clear(); clearInterval(state.poller);
  for (const child of [...scene.children]) if (child.userData.partObject) scene.remove(child);
  document.querySelector('#counts').textContent = `${model.parts.length} parts · ${model.joints.length} joints · ${model.groups.length} rigid groups`;
  document.querySelector('#joints').replaceChildren(...model.joints.map(jointControl));
  document.querySelector('#subsystems').replaceChildren(...model.subsystems.map(subsystem => { const item = document.createElement('div'); item.className = 'subsystem'; item.textContent = `${subsystem.name} · ${subsystem.joints.join(', ')}`; return item; }));
  document.querySelector('#warnings').replaceChildren(...model.warnings.slice(0, 40).map(warning => { const item = document.createElement('li'); item.textContent = warning; return item; }));
  document.querySelector('#empty').style.display = 'none'; drawMarkers(); addAvailableMeshes(); updateMeshButton();
}
function jointControl(joint) { const holder = document.createElement('div'); holder.className = 'joint'; const title = document.createElement('div'); title.className = 'joint-name'; title.innerHTML = `<span></span><small>${joint.kind}</small>`; title.firstChild.textContent = joint.name; const slider = document.createElement('input'); slider.type = 'range'; slider.min = joint.kind === 'hinge' ? -180 : -.5; slider.max = joint.kind === 'hinge' ? 180 : .5; slider.step = joint.kind === 'hinge' ? 1 : .005; slider.addEventListener('input', async () => { const val = Number(slider.value); state.values[joint.name] = joint.kind === 'hinge' ? val * Math.PI / 180 : val; const result = await request('/api/pose', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ file: state.model.filename, values: state.values }) }); for (const part of result.parts) { const object = state.objects.get(partKey(part)); if (object) applyPose(object, part.pose); } }); holder.append(title, slider); return holder; }
async function load(filename) { showModel(await request(`/api/model?file=${encodeURIComponent(filename)}`)); document.querySelector('#connection').textContent = 'Assembly loaded'; }
function pollMeshJob() { state.poller = setInterval(async () => { try { const job = await request(`/api/meshes?file=${encodeURIComponent(state.model.filename)}`); for (const mesh of job.available) state.available.add(mesh); await addAvailableMeshes(); updateMeshButton(job); document.querySelector('#connection').textContent = job.done ? (job.errors.length ? `CAD download stopped: ${job.errors[0]}` : 'CAD rendered') : `Downloading CAD geometry (${job.completed}/${job.total})`; if (job.done) clearInterval(state.poller); } catch (error) { clearInterval(state.poller); showError(error); } }, 1000); }
assembly.addEventListener('change', () => load(assembly.value).catch(showError));
document.querySelector('#reset').addEventListener('click', () => load(assembly.value).catch(showError));
meshButton.addEventListener('click', async () => { try { await request('/api/meshes', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({file: state.model.filename}) }); clearInterval(state.poller); pollMeshJob(); document.querySelector('#connection').textContent = 'Downloading CAD geometry in the background'; } catch (error) { showError(error); } });
document.querySelector('#import-form').addEventListener('submit', async event => { event.preventDefault(); const button = event.currentTarget.querySelector('button'); button.disabled = true; button.textContent = 'Importing…'; try { const model = await request('/api/import', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({url: document.querySelector('#onshape-url').value}) }); assembly.add(new Option(model.filename, model.filename)); assembly.value = model.filename; showModel(model); document.querySelector('#connection').textContent = 'Assembly imported'; } catch (error) { showError(error); } finally { button.disabled = false; button.textContent = 'Import'; } });
function showError(error) { document.querySelector('#connection').textContent = `Error: ${error.message}`; }
renderer.domElement.addEventListener('click', event => { const rect = renderer.domElement.getBoundingClientRect(), mouse = new THREE.Vector2((event.clientX-rect.left)/rect.width*2-1, -(event.clientY-rect.top)/rect.height*2+1), ray = new THREE.Raycaster(); ray.setFromCamera(mouse,camera); const hit = ray.intersectObjects([...state.objects.values()])[0]; if (hit?.object.userData.part) document.querySelector('#selection').textContent = `${hit.object.userData.part.name} · ${partKey(hit.object.userData.part)}`; });
window.addEventListener('resize', () => { camera.aspect = viewer.clientWidth / viewer.clientHeight; camera.updateProjectionMatrix(); renderer.setSize(viewer.clientWidth, viewer.clientHeight); });
(async () => { try { const data = await request('/api/assemblies'); assembly.replaceChildren(...data.assemblies.map(name => new Option(name,name))); if (data.assemblies.length) await load(data.assemblies[0]); else document.querySelector('#empty').style.display = 'grid'; render(); } catch (error) { showError(error); } })();
