// 识海 · 记忆星图：无限的黑 + 一闪一闪的记忆星。
//
// 星       = 记忆节点（事实）与实体枢纽；星温着色（中性=淡白星光，
//            喜悦偏暖金、低落偏冷蓝——像真实恒星的色温，效价信息保留）
// 连接     = 三类真实的边：同源(share) 同事(link) 相似(sim)
// 点亮     = /api/state 的激活事件——真值层真实发生的扩散激活
//           （对话扩散 / 心智游移路径），亮起的正是检索权重变高的记忆
// 动态     = 每 2s 拉 /api/graph，新星从所属枢纽旁生长出来
const $ = (id) => document.getElementById(id);

// ---------------- 场景 ----------------
const canvas = $("sea");
// preserveDrawingBuffer：远程桌面/截图管道下 WebGL 层不至于黑屏
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false,
                                           preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x010104);        // 无限的黑
scene.fog = new THREE.FogExp2(0x010104, 0.0016);
const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 9000);

// 轨道相机（手写，不依赖 OrbitControls）
const orbit = { theta: 0.6, phi: 1.15, r: 260, target: new THREE.Vector3(0, 0, 0) };
function applyCamera() {
  camera.position.set(
    orbit.target.x + orbit.r * Math.sin(orbit.phi) * Math.cos(orbit.theta),
    orbit.target.y + orbit.r * Math.cos(orbit.phi),
    orbit.target.z + orbit.r * Math.sin(orbit.phi) * Math.sin(orbit.theta));
  camera.lookAt(orbit.target);
}
function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

scene.add(new THREE.AmbientLight(0xffffff, 0.9));

// ---------------- 发光贴图 ----------------
function glowTexture(inner, outer) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, inner);
  grad.addColorStop(0.35, outer);
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}
const TEX = {
  star: glowTexture("rgba(255,255,255,1)", "rgba(170,185,255,0.55)"),
  hub: glowTexture("rgba(255,244,214,1)", "rgba(255,205,110,0.6)"),
};

// ---------------- 深空星幕（无限黑里的一闪一闪） ----------------
const BG_LAYERS = [
  { n: 260, rMin: 1200, rMax: 1800, size: 2.4, hue: 0xdfe6ff },
  { n: 160, rMin: 1900, rMax: 2600, size: 1.8, hue: 0xaab6e8 },
  { n: 90,  rMin: 2800, rMax: 3600, size: 1.3, hue: 0x8792c8 },
];
const bgStars = [];                // 每颗远景星：层、序号、相位、频率、基色
const bgLayers = [];
for (const L of BG_LAYERS) {
  const pos = new Float32Array(L.n * 3);
  const col = new Float32Array(L.n * 3);
  for (let i = 0; i < L.n; i++) {
    const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2;
    const rr = L.rMin + Math.random() * (L.rMax - L.rMin);
    const s2 = Math.sqrt(1 - u * u);
    pos[i * 3] = rr * s2 * Math.cos(th);
    pos[i * 3 + 1] = rr * u;
    pos[i * 3 + 2] = rr * s2 * Math.sin(th);
    const b = 0.45 + Math.random() * 0.55;
    const base = new THREE.Color(L.hue).multiplyScalar(b);
    col[i * 3] = base.r; col[i * 3 + 1] = base.g; col[i * 3 + 2] = base.b;
    bgStars.push({ col, i, phase: Math.random() * Math.PI * 2,
                   freq: 0.4 + Math.random() * 1.8, base });
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pts = new THREE.Points(geo, new THREE.PointsMaterial({
    size: L.size, map: TEX.star, vertexColors: true, transparent: true,
    opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false,
    sizeAttenuation: true,
  }));
  scene.add(pts);
  bgLayers.push({ pts, col });
}
function twinkleBackdrop(now) {
  for (const tw of bgStars) {
    const k = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(now * 0.001 * tw.freq + tw.phase));
    const i = tw.i * 3;
    tw.col[i] = tw.base.r * k; tw.col[i + 1] = tw.base.g * k; tw.col[i + 2] = tw.base.b * k;
  }
  for (const L of bgLayers) L.pts.geometry.attributes.color.needsUpdate = true;
}

// ---------------- 星温着色 ----------------
const STAR_NEUTRAL = new THREE.Color(0xcdd6ff);
const STAR_WARM = new THREE.Color(0xffd9a0);
const STAR_COLD = new THREE.Color(0x8fb8ff);
const SILVER = new THREE.Color(0xf2f6ff);        // 被想到时的银白色闪光
function valenceColor(v) {
  if (v === null || v === undefined) return STAR_NEUTRAL.clone();
  const c = STAR_NEUTRAL.clone();
  return c.lerp(v >= 0 ? STAR_WARM : STAR_COLD, Math.min(1, Math.abs(v) * 1.2));
}

// ---------------- 图数据 ----------------
const nodes = new Map();      // id -> 节点对象
let edges = [];               // {a, b, kind, w}
let lineObj = null;
let edgeBase = [];

function nodeRadius(n) {
  return n.kind === "entity" ? 7 + Math.min(10, (n.degree || 1) * 0.6) : 3.4;
}

function spawnPosFor(n) {
  // 事实星：出生在自己的实体枢纽旁边（像神经元长出突起）。
  // 用实体字段找枢纽——曾错用 label 前缀（那是谓词），新星随机落地。
  const hubId = "e:" + (n.entity || (n.label || "").split("=")[0]);
  const hub = nodes.get(hubId);
  if (hub) {
    const p = hub.sprite.position;
    return new THREE.Vector3(p.x + (Math.random() - 0.5) * 36,
                             p.y + (Math.random() - 0.5) * 36,
                             p.z + (Math.random() - 0.5) * 36);
  }
  return new THREE.Vector3((Math.random() - 0.5) * 60, (Math.random() - 0.5) * 60,
                           (Math.random() - 0.5) * 60);
}

function makeNode(n) {
  const color = n.kind === "entity" ? new THREE.Color(0xffe9a8) : valenceColor(n.valence);
  const mat = new THREE.SpriteMaterial({
    map: n.kind === "entity" ? TEX.hub : TEX.star,
    color, transparent: true, opacity: 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false,
  });
  const baseColor = color.clone();               // 星温基色（点亮闪银后沉回这里）
  const sp = new THREE.Sprite(mat);
  const r = nodeRadius(n);
  sp.scale.set(r * 2.4, r * 2.4, 1);
  sp.position.copy(spawnPosFor(n));
  scene.add(sp);
  return { id: n.id, kind: n.kind, label: n.label, tip: n.tip || "",
           entity: n.entity, valence: n.valence, degree: n.degree || 1, t: n.t,
           sprite: sp, baseColor, glow: 0, born: performance.now(),
           vel: new THREE.Vector3(),
           twPhase: Math.random() * Math.PI * 2,
           twFreq: n.kind === "entity" ? 0.4 : 0.9 + Math.random() * 1.6 };
}

// ---------------- 力导向（3D，紧凑版）----------------
const SPRING = 0.012, REPULSE = 900, CENTER = 0.0035, DAMP = 0.86;
function simulate() {
  const arr = [...nodes.values()];
  for (let i = 0; i < arr.length; i++) {
    const a = arr[i];
    for (let j = i + 1; j < arr.length; j++) {
      const b = arr[j];
      const d = a.sprite.position.distanceToSquared(b.sprite.position);
      if (d > 30000) continue;
      const f = REPULSE / Math.max(d, 25);
      const dir = new THREE.Vector3().subVectors(a.sprite.position, b.sprite.position).normalize();
      const wA = a.kind === "entity" ? 0.4 : 1;     // 枢纽重，动得少
      a.vel.addScaledVector(dir, f * wA);
      b.vel.addScaledVector(dir, -f * (b.kind === "entity" ? 0.4 : 1));
    }
  }
  for (const e of edges) {
    const a = nodes.get(e.a), b = nodes.get(e.b);
    if (!a || !b) continue;
    const dir = new THREE.Vector3().subVectors(b.sprite.position, a.sprite.position);
    const len = dir.length() || 1;
    const rest = e.kind === "share" ? 34 : 46;
    const f = SPRING * (len - rest) / len;
    a.vel.addScaledVector(dir, f);
    b.vel.addScaledVector(dir, -f);
  }
  for (const n of arr) {
    n.vel.addScaledVector(n.sprite.position, -CENTER);
    n.vel.multiplyScalar(DAMP);
    n.sprite.position.add(n.vel);
  }
}

// ---------------- 边渲染 ----------------
const EDGE_COLOR = { share: new THREE.Color(0x5a6db8), link: new THREE.Color(0xd8b86a),
                     sim: new THREE.Color(0xa86ad8), rel: new THREE.Color(0xffb27a),
                     co: new THREE.Color(0x4a5588) };
function rebuildLines() {
  if (lineObj) { scene.remove(lineObj); lineObj.geometry.dispose(); }
  const n = edges.length;
  const pos = new Float32Array(n * 6);
  const col = new Float32Array(n * 6);
  edgeBase = edges.map(e => EDGE_COLOR[e.kind] || EDGE_COLOR.share);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  lineObj = new THREE.LineSegments(geo,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true,
                                  opacity: 0.6, blending: THREE.AdditiveBlending,
                                  depthWrite: false }));
  scene.add(lineObj);
}

function updateLinePositions() {
  if (!lineObj) return;
  const pos = lineObj.geometry.attributes.position.array;
  const col = lineObj.geometry.attributes.color.array;
  edges.forEach((e, i) => {
    const a = nodes.get(e.a), b = nodes.get(e.b);
    const o = i * 6;
    if (!a || !b) { pos.fill(0, o, o + 6); return; }
    pos[o] = a.sprite.position.x; pos[o + 1] = a.sprite.position.y; pos[o + 2] = a.sprite.position.z;
    pos[o + 3] = b.sprite.position.x; pos[o + 4] = b.sprite.position.y; pos[o + 5] = b.sprite.position.z;
    // 点亮：两端 glow -> 边变亮变白（神经元导电的感觉）
    const glow = Math.min(1, (a.glow + b.glow) * 0.5);
    const base = edgeBase[i] || EDGE_COLOR.share;
    const w = 0.45 + 0.55 * Math.min(1, e.w);
    const c = base.clone().lerp(new THREE.Color(0xffffff), glow * 0.8)
                  .multiplyScalar(w + glow * 1.6);
    col[o] = c.r; col[o + 1] = c.g; col[o + 2] = c.b;
    col[o + 3] = c.r; col[o + 4] = c.g; col[o + 5] = c.b;
  });
  lineObj.geometry.attributes.position.needsUpdate = true;
  lineObj.geometry.attributes.color.needsUpdate = true;
}

// ---------------- 拉取图（动态增加）----------------
let graphVersion = -1;
async function pullGraph() {
  try {
    const r = await fetch("/api/graph");
    const g = await r.json();
    if (g.version === graphVersion) return;
    graphVersion = g.version;
    for (const n of g.nodes) {
      if (nodes.has(n.id)) {
        Object.assign(nodes.get(n.id), { label: n.label, tip: n.tip || "", degree: n.degree });
        continue;
      }
      const node = makeNode(n);
      node.sprite.position.copy(spawnPosFor(n));
      nodes.set(n.id, node);
      node._homeHub = "e:" + n.entity;          // 盲测用：出生时应靠近的枢纽
    }
    const alive = new Set(g.nodes.map(n => n.id));
    for (const [id, node] of nodes) {
      if (!alive.has(id)) { scene.remove(node.sprite); nodes.delete(id); }
    }
    edges = g.edges;
    rebuildLines();
    $("stats").textContent = `记忆 ${g.nodes.filter(n => n.kind === "fact").length} 颗星 · 连接 ${g.edges.length} 缕`;
  } catch (e) { /* 服务没起 */ }
}

// ---------------- 激活点亮 ----------------
// 基线 = 页面加载时刻（本机服务，时钟同源）：只播这之后的新点亮
let lastActT = Date.now() / 1000 - 2;
async function pullActivations() {
  try {
    const r = await fetch("/api/state");
    const s = await r.json();
    $("moodLine").textContent = "心情：" + s.mood.label;
    for (const act of (s.activations || [])) {
      if (act.t <= lastActT) continue;
      lastActT = act.t;
      for (const [nid, v] of Object.entries(act.acc)) {
        const n = nodes.get(nid);          // 键已是 "f:<id>"（push_activation 规范化）
        if (n) n.glow = Math.min(1.4, n.glow + 0.5 + v);
      }
    }
  } catch (e) { /* 忽略 */ }
}

// ---------------- 交互 ----------------
let dragging = false, lastX = 0, lastY = 0;
canvas.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", () => { dragging = false; });
window.addEventListener("mousemove", (e) => {
  if (dragging) {
    orbit.theta += (e.clientX - lastX) * 0.006;   // 向左拖=图像向左转（抓取约定）
    orbit.phi = Math.min(3.0, Math.max(0.15, orbit.phi - (e.clientY - lastY) * 0.006));
    lastX = e.clientX; lastY = e.clientY;
  } else {
    hover(e.clientX, e.clientY);
  }
});
let zoomK = 1.0;                     // 用户缩放偏移，乘在自动取景上
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomK = Math.min(4, Math.max(0.3, zoomK * (e.deltaY > 0 ? 1.12 : 0.89)));
}, { passive: false });

const raycaster = new THREE.Raycaster();
function hover(x, y) {
  const px = (x / window.innerWidth) * 2 - 1;
  const py = -(y / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera({ x: px, y: py }, camera);
  const hit = raycaster.intersectObjects([...nodes.values()].map(n => n.sprite), false)[0];
  const tip = $("tip");
  if (!hit) { tip.hidden = true; return; }
  const node = [...nodes.values()].find(n => n.sprite === hit.object);
  if (!node) { tip.hidden = true; return; }
  if (node.kind === "entity") {
    tip.innerHTML = `<b>${node.label}</b><div class="meta">实体枢纽 · 名下 ${node.degree} 条记忆</div>`;
  } else {
    const date = node.t ? new Date(node.t * 1000).toLocaleString("zh-CN", { hour12: false }) : "";
    tip.innerHTML = `<b>${node.label}</b>` +
      (node.tip ? `<div>${node.tip}</div>` : "") +
      `<div class="meta">${date}${node.valence !== null && node.valence !== undefined ? " · 效价 " + node.valence.toFixed(2) : ""}</div>`;
  }
  tip.style.left = Math.min(x + 14, window.innerWidth - 320) + "px";
  tip.style.top = (y + 14) + "px";
  tip.hidden = false;
}

// ---------------- 自动取景 ----------------
let framedR = null;
function autoFrame() {
  const arr = [...nodes.values()];
  if (!arr.length) return;
  const box = new THREE.Box3();
  for (const n of arr) box.expandByPoint(n.sprite.position);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const fit = Math.max(90, Math.min(600, sphere.radius / Math.tan(camera.fov * Math.PI / 360) + 40));
  framedR = framedR === null ? fit : framedR + (fit - framedR) * 0.04;
  orbit.r = framedR * zoomK;
}

// ---------------- 主循环 ----------------
function tick(now) {
  simulate();
  autoFrame();
  twinkleBackdrop(now);
  // 新星生成 + 一闪一闪（每星自己的相位/频率；点亮时盖过闪烁）
  for (const n of nodes.values()) {
    const age = (now - n.born) / 600;
    const tw = 0.72 + 0.28 * (0.5 + 0.5 * Math.sin(now * 0.001 * n.twFreq + n.twPhase));
    const r = nodeRadius(n) * Math.min(1, age) * (1 + n.glow * 0.25) * (0.9 + 0.1 * tw);
    n.sprite.scale.set(r * 2.4, r * 2.4, 1);
    n.glow *= 0.99;                              // 点亮后的余晖（半衰期约 3.5s）
    n.sprite.material.opacity = Math.min(1, tw * (0.75 + 0.6 * n.glow));
    // 被想到 = 那个点闪银色的光：颜色向银白跳变，随余晖沉回星温色
    if (n.glow > 0.01) {
      // 闪光峰值要明显亮过常态星（不是换个颜色而已）
      n.sprite.material.color.copy(n.baseColor)
        .lerp(SILVER, Math.min(1, n.glow))
        .multiplyScalar(1 + 0.9 * Math.min(1, n.glow));
      n._silvered = true;
    } else if (n._silvered) {
      n.sprite.material.color.copy(n.baseColor);
      n._silvered = false;
    }
  }
  updateLinePositions();
  applyCamera();
  renderer.render(scene, camera);
}

pullGraph();
pullActivations();
setInterval(pullGraph, 2000);
setInterval(pullActivations, 1500);
// 后台标签页会把 rAF 冻结——看门狗：rAF 超 600ms 没跑就用低频 interval 步进
let lastTickAt = performance.now();
const stepTick = (now) => { lastTickAt = performance.now(); tick(now); };
const rafLoop = (now) => { stepTick(now); requestAnimationFrame(rafLoop); };
requestAnimationFrame(rafLoop);
setInterval(() => {
  if (performance.now() - lastTickAt > 600) stepTick(performance.now());
}, 400);
