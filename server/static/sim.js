/**
 * Sim tab — canvas renderer + WebSocket client for /sim/events.
 *
 * Coordinate system: world cm (x right, y up). The canvas has y flipped
 * (pixel y down). We compute a viewport transform once per resize that
 * maps world coordinates to canvas pixels with a fixed margin.
 */
'use strict';
console.log('[sim.js] v20260423 loaded');

(function () {

// --- State -----------------------------------------------------------------
let ws = null;
let walls = [];          // [[ax,ay],[bx,by]] in world cm
let robots = [];         // from WebSocket
let lastMsg = null;      // most recent sim frame
let vp = null;           // viewport transform {ox, oy, scale}

const canvas = document.getElementById('sim-canvas');
const ctx    = canvas.getContext('2d');

let selectedRobotId = null;
let dragMode = null; // 'move', 'rotate', null
let dragStart = null;

// --- Playback controls -------------------------------------------------------
let playing = true;
let timeScale = 1.0;

// Keyboard drive state
const keys = {};
window.addEventListener('keydown', e => { keys[e.code] = true; });
window.addEventListener('keyup',   e => { keys[e.code] = false; });
let driveTick = null;

// JWT — stored in sessionStorage so the page survives soft refreshes.
function getJWT() { return sessionStorage.getItem('pysaac_jwt') || ''; }

// --- WebSocket -------------------------------------------------------------
function connectSimWS() {
  const jwt = getJWT();
  if (!jwt) {
    pysaac.log('No JWT in sessionStorage["pysaac_jwt"] — sim WS skipped', 'err');
    return;
  }
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url   = `${proto}://${location.host}/sim/events?token=${encodeURIComponent(jwt)}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    pysaac.setConnected(true);
    pysaac.log('Sim WS connected', 'ok');
    fetchWalls();
    fetchRobotsList();
    startDriveLoop();
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.kind && msg.kind.startsWith('training_')) {
          if (window.handleLiveTrainingMsg) window.handleLiveTrainingMsg(msg);
          return;
      }
      if (msg.kind === 'sim') {
        handleSimMsg(msg);
      }
    } catch (err) {}
  };

  ws.onclose = (ev) => {
    pysaac.setConnected(false);
    pysaac.log(`Sim WS closed (${ev.code})`, ev.wasClean ? '' : 'err');
    stopDriveLoop();
    setTimeout(connectSimWS, 3000);
  };

  ws.onerror = () => pysaac.log('Sim WS error', 'err');
}

async function fetchWalls() {
  const jwt = getJWT();
  try {
    const r = await fetch('/gui/track', { headers: { Authorization: 'Bearer ' + jwt } });
    if (!r.ok) return;
    const data = await r.json();
    walls = data.walls || [];
    vp = null;   // force recompute
    pysaac.log(`Loaded ${walls.length} wall segments`);
  } catch (e) {
    pysaac.log('fetchWalls failed: ' + e, 'err');
  }
}

async function fetchRobotsList() {
    const r = await fetch('/gui/robots', { headers: { Authorization: 'Bearer ' + getJWT() } });
    if (!r.ok) return;
    const data = await r.json();
    const sel = document.getElementById('sim-robot-list');
    sel.innerHTML = '';
    data.forEach(rob => {
        const opt = document.createElement('option');
        opt.value = rob.id;
        opt.textContent = `#${rob.id} (${rob.x.toFixed(1)}, ${rob.y.toFixed(1)}, ${(rob.theta*180/Math.PI).toFixed(1)}°) — ${rob.controller_id}`;
        if (rob.id === selectedRobotId) opt.selected = true;
        sel.appendChild(opt);
    });
}

// --- REST Actions ----------------------------------------------------------

async function spawnRobot(x, y, theta) {
    const ctrl = document.getElementById('sim-robot-ctrl').value || 'web-manual';
    const r = await fetch('/gui/robots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + getJWT() },
        body: JSON.stringify({ x, y, theta, controller_id: ctrl })
    });
    if (r.ok) {
        const data = await r.json();
        selectedRobotId = data.robot.id;
        fetchRobotsList();
        updateSelectedUI();
    }
}

async function patchRobot(id, patch) {
    await fetch(`/gui/robots/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + getJWT() },
        body: JSON.stringify(patch)
    });
    fetchRobotsList();
}

async function deleteRobot(id) {
    await fetch(`/gui/robots/${id}`, { method: 'DELETE', headers: { Authorization: 'Bearer ' + getJWT() } });
    if (selectedRobotId === id) selectedRobotId = null;
    fetchRobotsList();
    updateSelectedUI();
}

async function clearRobots() {
    await fetch('/gui/robots/clear', { method: 'POST', headers: { Authorization: 'Bearer ' + getJWT() } });
    selectedRobotId = null;
    fetchRobotsList();
    updateSelectedUI();
}

async function updatePlayback(opts) {
    await fetch('/gui/playback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + getJWT() },
        body: JSON.stringify(opts)
    });
}

async function resetPlayback() {
    await fetch('/gui/playback/reset', { method: 'POST', headers: { Authorization: 'Bearer ' + getJWT() } });
}

// --- Keyboard drive --------------------------------------------------------
function startDriveLoop() {
  if (driveTick) return;
  driveTick = setInterval(sendDriveCmd, 80);
}
function stopDriveLoop() {
  if (driveTick) { clearInterval(driveTick); driveTick = null; }
}

async function sendDriveCmd() {
  const fwd = keys['KeyW'] || keys['ArrowUp'];
  const rev = keys['KeyS'] || keys['ArrowDown'];
  const lft = keys['KeyA'] || keys['ArrowLeft'];
  const rgt = keys['KeyD'] || keys['ArrowRight'];
  if (!fwd && !rev && !lft && !rgt) return;

  const throttle = fwd ? 0.8 : rev ? -0.5 : 0.0;
  const steer    = lft ? -0.6 : rgt ? 0.6 : 0.0;
  try {
    await fetch('/sim/command', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + getJWT() },
      body: JSON.stringify({ throttle, steer }),
    });
  } catch (_) {}
}

// --- Canvas Interactions ---------------------------------------------------

function canvasToWorld(cx, cy) {
    if (!vp) return {x:0, y:0};
    const x = (cx - vp.ox) / vp.scale;
    const y = ((canvas.height - cy) - vp.oy) / vp.scale;
    return {x, y};
}

function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function findRobotAt(x, y) {
    for (let i = robots.length - 1; i >= 0; i--) {
        const r = robots[i];
        if (!r.pose) continue;
        if (dist({x,y}, r.pose) < 30) return r; // rough bounding circle
    }
    return null;
}

canvas.addEventListener('mousedown', e => {
    const pt = canvasToWorld(e.offsetX, e.offsetY);
    
    // Right click -> delete
    if (e.button === 2) {
        const r = findRobotAt(pt.x, pt.y);
        if (r) deleteRobot(r.id);
        return;
    }
    
    // Check if rotating selected robot
    if (selectedRobotId !== null) {
        const r = robots.find(x => x.id === selectedRobotId);
        if (r && r.pose) {
            const d = dist(pt, r.pose);
            if (d > 15 && d < 45) { // rotation ring hit
                dragMode = 'rotate';
                return;
            }
        }
    }
    
    const r = findRobotAt(pt.x, pt.y);
    if (r) {
        selectedRobotId = r.id;
        dragMode = 'move';
        updateSelectedUI();
        fetchRobotsList();
        return;
    }
    
    // Clicked empty space -> spawn
    spawnRobot(pt.x, pt.y, 0.0);
});

canvas.addEventListener('mousemove', e => {
    if (!dragMode || selectedRobotId === null) return;
    const pt = canvasToWorld(e.offsetX, e.offsetY);
    
    if (dragMode === 'move') {
        patchRobot(selectedRobotId, {x: pt.x, y: pt.y});
    } else if (dragMode === 'rotate') {
        const r = robots.find(x => x.id === selectedRobotId);
        if (r && r.pose) {
            const th = Math.atan2(pt.y - r.pose.y, pt.x - r.pose.x);
            patchRobot(selectedRobotId, {theta: th});
        }
    }
});

canvas.addEventListener('mouseup', () => dragMode = null);
canvas.addEventListener('contextmenu', e => e.preventDefault());

window.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.code === 'KeyR' && selectedRobotId !== null) {
        const r = robots.find(x => x.id === selectedRobotId);
        if (r && r.pose) {
            const dir = e.shiftKey ? +1 : -1;
            patchRobot(selectedRobotId, {theta: r.pose.theta + dir * (15 * Math.PI / 180)});
        }
    }
});

// --- UI Bindings -----------------------------------------------------------

document.getElementById('sim-robot-list').addEventListener('change', e => {
    selectedRobotId = parseInt(e.target.value);
    updateSelectedUI();
});

document.getElementById('sim-robot-ctrl').addEventListener('change', e => {
    if (selectedRobotId !== null) patchRobot(selectedRobotId, {controller_id: e.target.value});
});

document.getElementById('btn-sim-clear').addEventListener('click', clearRobots);

const inX = document.getElementById('sim-sel-x');
const inY = document.getElementById('sim-sel-y');
const inTh = document.getElementById('sim-sel-th');

function onFormEdit() {
    if (selectedRobotId !== null) {
        patchRobot(selectedRobotId, {
            x: parseFloat(inX.value),
            y: parseFloat(inY.value),
            theta: parseFloat(inTh.value) * Math.PI / 180
        });
    }
}
inX.addEventListener('change', onFormEdit);
inY.addEventListener('change', onFormEdit);
inTh.addEventListener('change', onFormEdit);

const btnPlay = document.getElementById('btn-sim-play');
btnPlay.addEventListener('click', () => {
    playing = !playing;
    btnPlay.innerHTML = playing ? '&#10074;&#10074; Pause' : '&#9654; Play';
    updatePlayback({ 
      time_scale: playing ? parseFloat(document.getElementById('sim-speed').value) : 0.0,
      playing: playing
    });
});

document.getElementById('btn-sim-reset').addEventListener('click', resetPlayback);

document.getElementById('sim-speed').addEventListener('input', e => {
    document.getElementById('sim-speed-lbl').textContent = parseFloat(e.target.value).toFixed(1) + 'x';
    if (playing) updatePlayback({ time_scale: parseFloat(e.target.value) });
});

document.getElementById('chk-sim-interact').addEventListener('change', e => {
    updatePlayback({ cars_interact: e.target.checked });
});

function updateSelectedUI() {
    if (selectedRobotId === null) return;
    const r = robots.find(x => x.id === selectedRobotId);
    if (!r) return;
    if (dragMode !== 'move' && document.activeElement !== inX && document.activeElement !== inY && document.activeElement !== inTh) {
        inX.value = r.pose.x.toFixed(1);
        inY.value = r.pose.y.toFixed(1);
        inTh.value = (r.pose.theta * 180 / Math.PI).toFixed(1);
    }
    const cSel = document.getElementById('sim-robot-ctrl');
    if (cSel.value !== r.controller_id) {
        // if not in list, maybe add it
        let found = false;
        for (let i=0; i<cSel.options.length; i++) if (cSel.options[i].value === r.controller_id) found = true;
        if (!found) {
            const opt = document.createElement('option');
            opt.value = opt.textContent = r.controller_id;
            cSel.appendChild(opt);
        }
        cSel.value = r.controller_id;
    }
}

// --- Render ----------------------------------------------------------------

function handleSimMsg(msg) {
  if (msg.kind !== 'sim') return;
  lastMsg = msg;
  robots = msg.robots || [];
  
  updateSelectedUI();
  updateTelemetry();
  draw();
}

function updateTelemetry() {
    const term = document.getElementById('sim-readings');
    if (selectedRobotId === null) {
        term.textContent = "(no robot selected — click on one in the list or track)";
        return;
    }
    const r = robots.find(x => x.id === selectedRobotId);
    if (!r || !r.pose) {
        term.textContent = "(selection invalid)";
        return;
    }
    
    let lines = [
        `robot #${r.id}  ctrl=${r.controller_id}  ${playing ? 'RUNNING' : 'stopped'}`,
        `pose   x=${r.pose.x.toFixed(2)} cm  y=${r.pose.y.toFixed(2)} cm  θ=${(r.pose.theta*180/Math.PI).toFixed(2)}°`,
        `v=${(r.v||0).toFixed(2)} cm/s  ω=${(r.omega||0).toFixed(3)} rad/s  steer=${((r.steer||0)*180/Math.PI).toFixed(1)}°  collided=${r.collided}`,
        ""
    ];
    
    if (r.lidar && r.lidar.center) {
        lines.push("Lidar (cm):");
        lines.push(`  center = ${r.lidar.center.distance_cm.toFixed(1).padStart(5)}   valid=${r.lidar.center.valid}`);
        lines.push(`  left   = ${r.lidar.left.distance_cm.toFixed(1).padStart(5)}   valid=${r.lidar.left.valid}`);
        lines.push(`  right  = ${r.lidar.right.distance_cm.toFixed(1).padStart(5)}   valid=${r.lidar.right.valid}`);
        lines.push("");
    }
    if (r.ir && r.ir.left) {
        lines.push("IR (cm):");
        lines.push(`  left   = ${r.ir.left.distance_cm.toFixed(2).padStart(5)}  valid=${r.ir.left.valid}`);
        lines.push(`  right  = ${r.ir.right.distance_cm.toFixed(2).padStart(5)}  valid=${r.ir.right.valid}`);
    }
    
    term.textContent = lines.join("\n");
}

function ensureViewport() {
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  if (!walls.length) {
    vp = { ox: 20, oy: 20, scale: Math.min((W - 40) / 490, (H - 40) / 240) };
    return;
  }
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const w of walls) {
    for (const p of w) {
      if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1];
    }
  }
  const ww = maxX - minX || 1, wh = maxY - minY || 1;
  const margin = 20;
  const scale = Math.min((W - 2*margin) / ww, (H - 2*margin) / wh);
  vp = { ox: margin - minX * scale, oy: margin - minY * scale, scale };
}

function wx(x) { return vp.ox + x * vp.scale; }
function wy(y) { return canvas.height - (vp.oy + y * vp.scale); }  // flip y

function draw() {
  ensureViewport();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Walls
  ctx.strokeStyle = '#555';
  ctx.lineWidth = Math.max(1, vp.scale * 2);
  for (const w of walls) {
    ctx.beginPath();
    ctx.moveTo(wx(w[0][0]), wy(w[0][1]));
    ctx.lineTo(wx(w[1][0]), wy(w[1][1]));
    ctx.stroke();
  }

  for (const r of robots) {
      const pose = r.pose;
      if (!pose) continue;

      // Draw sensors if selected
      if (r.id === selectedRobotId) {
          const SENSORS = [
            { key: 'center', color: '#4a9eff' },
            { key: 'left',   color: '#3ecf8e' },
            { key: 'right',  color: '#3ecf8e' },
          ];
          for (const { key, color } of SENSORS) {
            const s = r.lidar?.[key];
            if (!s) continue;
            const o = s.origin, h = s.hit;
            if (!o) continue;
            const endX = h ? h[0] : (o[0] + s.distance_cm * Math.cos(pose.theta));
            const endY = h ? h[1] : (o[1] + s.distance_cm * Math.sin(pose.theta));
            const frac = Math.min(1, s.distance_cm / 800);
            ctx.strokeStyle = `rgba(${hexToRgb(color)}, ${0.3 + 0.7 * (1 - frac)})`;
            ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(wx(o[0]), wy(o[1])); ctx.lineTo(wx(endX), wy(endY)); ctx.stroke();
            if (h) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(wx(h[0]), wy(h[1]), 2, 0, 2*Math.PI); ctx.fill(); }
          }
          for (const { key, color } of [{key:'left',color:'#f0c040'},{key:'right',color:'#f0c040'}]) {
            const s = r.ir?.[key];
            if (!s || !s.valid) continue;
            const o = s.origin, h = s.hit;
            if (!o) continue;
            ctx.strokeStyle = 'rgba(240,192,64,0.35)';
            ctx.lineWidth = 1;
            const endX = h ? h[0] : o[0]; const endY = h ? h[1] : o[1];
            ctx.beginPath(); ctx.moveTo(wx(o[0]), wy(o[1])); ctx.lineTo(wx(endX), wy(endY)); ctx.stroke();
          }
          
          // Selection rotation ring
          ctx.beginPath();
          ctx.arc(wx(pose.x), wy(pose.y), 30 * vp.scale, 0, 2*Math.PI);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.lineWidth = 1;
          ctx.stroke();
      }

      // Robot rectangle (chassis 29.5 × 19 cm)
      const L = 29.5, W = 19, hl = L/2, hw = W/2;
      const corners = [
        rotPt(+hl, +hw, pose.theta), rotPt(+hl, -hw, pose.theta),
        rotPt(-hl, -hw, pose.theta), rotPt(-hl, +hw, pose.theta),
      ].map(([bx, by]) => [pose.x + bx, pose.y + by]);
      ctx.fillStyle   = r.id === selectedRobotId ? 'rgba(74,158,255,0.45)' : 'rgba(74,158,255,0.25)';
      ctx.strokeStyle = r.collided ? '#ff4d4d' : r.color || '#4a9eff';
      ctx.lineWidth   = r.id === selectedRobotId ? 2 : 1.5;
      ctx.beginPath();
      ctx.moveTo(wx(corners[0][0]), wy(corners[0][1]));
      for (let i = 1; i < 4; i++) ctx.lineTo(wx(corners[i][0]), wy(corners[i][1]));
      ctx.closePath(); ctx.fill(); ctx.stroke();

      // Heading arrow
      const hlen = 18;
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(wx(pose.x), wy(pose.y));
      ctx.lineTo(wx(pose.x + hlen * Math.cos(pose.theta)),
                 wy(pose.y + hlen * Math.sin(pose.theta)));
      ctx.stroke();
  }
}

function rotPt(bx, by, theta) {
  return [bx * Math.cos(theta) - by * Math.sin(theta),
          bx * Math.sin(theta) + by * Math.cos(theta)];
}

function hexToRgb(hex) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `${r},${g},${b}`;
}

// --- Init ------------------------------------------------------------------
window.loadPolicyIntoSim = async function loadPolicyIntoSim() {
  const jobId = window._lastJobId;
  if (!jobId) { pysaac.log('No completed job to load', 'err'); return; }
  const jwt = getJWT();
  const r = await fetch('/sim/policy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + jwt },
    body: JSON.stringify({ job_id: jobId }),
  });
  if (r.ok) { pysaac.log('Policy loaded into sim ✓', 'ok'); fetchWalls(); }
  else       { pysaac.log('Load policy failed: ' + r.status, 'err'); }
};

// Kick off WS connection on page load.
window.addEventListener('load', () => {
  if (getJWT()) {
    connectSimWS();
  } else {
    const username = prompt('Enter username:');
    if (username) {
      const password = prompt('Enter password:');
      fetch('/auth/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ username, password })
      })
      .then(r => {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(data => {
        sessionStorage.setItem('pysaac_jwt', data.access_token);
        connectSimWS();
      })
      .catch(e => {
        pysaac.log('Login failed: ' + e.message, 'err');
        alert('Login failed. Please refresh the page to try again.');
      });
    }
  }
});

})();
