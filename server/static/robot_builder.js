let rbCanvas, rbCtx;
let rbDims = { chassis_length_cm: 20, chassis_width_cm: 10 };
let rbCal = { lidar_placements: [], ir_placements: [], lidar: {}, ir: {} };
let rbDragging = null;

let rbScale = 1;
let rbOffsetX = 0;
let rbOffsetY = 0;

function rbInit() {
    rbCanvas = document.getElementById('robot-canvas');
    if (!rbCanvas) return;
    rbCtx = rbCanvas.getContext('2d');
    
    rbCanvas.addEventListener('mousedown', rbOnMouseDown);
    rbCanvas.addEventListener('mousemove', rbOnMouseMove);
    rbCanvas.addEventListener('mouseup', rbOnMouseUp);
    
    window.addEventListener('resize', () => {
        if (!document.getElementById('robot-panel').classList.contains('hidden')) rbResize();
    });
}

function rbResize() {
    if (!rbCanvas) return;
    const wrap = document.getElementById('robot-canvas-wrap');
    rbCanvas.width = wrap.clientWidth;
    rbCanvas.height = wrap.clientHeight;
    rbRender();
}

async function rbLoad() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    if (!jwt) return;
    try {
        const [rDims, rCal] = await Promise.all([
            fetch("/gui/dims", { headers: { "Authorization": `Bearer ${jwt}` } }).then(r => r.json()),
            fetch("/gui/calibration", { headers: { "Authorization": `Bearer ${jwt}` } }).then(r => r.json())
        ]);
        rbDims = rDims;
        rbCal = rCal;
        rbBuildForms();
        rbRender();
    } catch(e) {
        console.error("Failed to load robot state", e);
    }
}

async function rbSave(type) {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    if (!jwt) return;
    const url = `/gui/export/${type}.yaml`;
    fetch(url, { headers: { 'Authorization': `Bearer ${jwt}` } })
        .then(r => r.blob())
        .then(blob => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `${type}.yaml`;
            a.click();
        });
}

async function rbLoadFile(event, type) {
    const file = event.target.files[0];
    if (!file) return;
    const jwt = sessionStorage.getItem('pysaac_jwt');
    const formData = new FormData();
    formData.append('file', file);
    try {
        await fetch(`/gui/import/${type}`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${jwt}` },
            body: formData
        });
        await rbLoad();
        window.pysaac.log(`${type} YAML imported`);
    } catch (e) {
        window.pysaac.log('Import failed', 'err');
    }
    event.target.value = '';
}

async function rbReset() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    await fetch('/gui/dims/reset', { method: 'POST', headers: { 'Authorization': `Bearer ${jwt}` } });
    await fetch('/gui/calibration/reset', { method: 'POST', headers: { 'Authorization': `Bearer ${jwt}` } });
    await rbLoad();
}

async function rbSyncLive() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    if (!jwt) return;
    try {
        await Promise.all([
            fetch("/gui/dims", { method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${jwt}` }, body: JSON.stringify(rbDims) }),
            fetch("/gui/calibration", { method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${jwt}` }, body: JSON.stringify(rbCal) })
        ]);
    } catch(e) {}
}

function rbBuildForms() {
    // Build Dims Form
    const dForm = document.getElementById('rb-dims-form');
    dForm.innerHTML = '';
    for (let [k, v] of Object.entries(rbDims)) {
        const div = document.createElement('div');
        div.style.marginBottom = '8px';
        div.innerHTML = `<label style="display:inline-block;width:150px;">${k}</label> <input type="number" id="rb-d-${k}" value="${v}" step="0.1" style="width:80px;" onchange="rbSyncDim('${k}')">`;
        dForm.appendChild(div);
    }
    
    // Build Placements Form
    const pForm = document.getElementById('rb-placements-form');
    pForm.innerHTML = '';
    const addPlacement = (p, listType, idx) => {
        const div = document.createElement('div');
        div.style.marginBottom = '12px';
        div.innerHTML = `<b>${p.id}</b><br>
            x: <input type="number" id="rb-px-${p.id}" value="${p.x}" step="0.1" style="width:60px;" onchange="rbSyncP('${listType}', ${idx}, 'x')"> 
            y: <input type="number" id="rb-py-${p.id}" value="${p.y}" step="0.1" style="width:60px;" onchange="rbSyncP('${listType}', ${idx}, 'y')"> 
            θ: <input type="number" id="rb-pt-${p.id}" value="${(p.theta * 180 / Math.PI).toFixed(1)}" step="1" style="width:60px;" onchange="rbSyncP('${listType}', ${idx}, 'theta')">`;
        pForm.appendChild(div);
    };
    (rbCal.lidar_placements || []).forEach((p, i) => addPlacement(p, 'lidar_placements', i));
    (rbCal.ir_placements || []).forEach((p, i) => addPlacement(p, 'ir_placements', i));
    
    // Build IR / Lidar forms
    const buildCalForm = (id, obj, name) => {
        const form = document.getElementById(id);
        form.innerHTML = '';
        for (let [k, v] of Object.entries(obj || {})) {
            const div = document.createElement('div');
            div.style.marginBottom = '8px';
            div.innerHTML = `<label style="display:inline-block;width:150px;">${k}</label> <input type="number" id="rb-c-${name}-${k}" value="${v}" step="0.01" style="width:80px;" onchange="rbSyncC('${name}', '${k}')">`;
            form.appendChild(div);
        }
    };
    buildCalForm('rb-ir-form', rbCal.ir, 'ir');
    buildCalForm('rb-lidar-form', rbCal.lidar, 'lidar');
}

function rbSyncDim(k) {
    rbDims[k] = parseFloat(document.getElementById(`rb-d-${k}`).value);
    rbRender();
    rbSyncLive();
}

function rbSyncP(listType, idx, field) {
    let val = parseFloat(document.getElementById(`rb-p${field[0]}-${rbCal[listType][idx].id}`).value);
    if (field === 'theta') val = val * Math.PI / 180;
    rbCal[listType][idx][field] = val;
    rbRender();
    rbSyncLive();
}

function rbSyncC(name, k) {
    rbCal[name][k] = parseFloat(document.getElementById(`rb-c-${name}-${k}`).value);
    rbRender();
    rbSyncLive();
}

function rbUpdateFormPlacement(p) {
    const elx = document.getElementById(`rb-px-${p.id}`);
    if (elx) elx.value = p.x.toFixed(1);
    const ely = document.getElementById(`rb-py-${p.id}`);
    if (ely) ely.value = p.y.toFixed(1);
}

function rbRender() {
    if (!rbCtx) return;
    rbCtx.clearRect(0, 0, rbCanvas.width, rbCanvas.height);
    
    const L = rbDims.chassis_length_cm || 20;
    const W = rbDims.chassis_width_cm || 10;
    
    rbScale = Math.min(rbCanvas.width / (L + 40), rbCanvas.height / (W + 40));
    rbOffsetX = rbCanvas.width / 2;
    rbOffsetY = rbCanvas.height / 2;
    
    const w2c = (x, y) => [x * rbScale + rbOffsetX, -y * rbScale + rbOffsetY];
    
    // Draw Chassis
    rbCtx.fillStyle = 'rgba(200, 220, 240, 0.8)';
    rbCtx.strokeStyle = '#111';
    rbCtx.lineWidth = 1;
    let [ltx, lty] = w2c(-L/2, W/2);
    let [rtx, rty] = w2c(L/2, -W/2);
    rbCtx.fillRect(ltx, lty, rtx-ltx, rty-lty);
    rbCtx.strokeRect(ltx, lty, rtx-ltx, rty-lty);
    
    // Draw Centerline Arrow
    rbCtx.strokeStyle = '#333';
    let [cx, cy] = w2c(0, 0);
    let [ax, ay] = w2c(L/2, 0);
    rbCtx.beginPath(); rbCtx.moveTo(cx, cy); rbCtx.lineTo(ax, ay);
    rbCtx.stroke();
    
    // Draw Sensors
    const drawSensors = (placements, color, length) => {
        placements.forEach(p => {
            let [x, y] = w2c(p.x, p.y);
            let ex = p.x + length * Math.cos(p.theta);
            let ey = p.y + length * Math.sin(p.theta);
            let [exC, eyC] = w2c(ex, ey);
            
            rbCtx.strokeStyle = color;
            rbCtx.setLineDash([4, 4]);
            rbCtx.beginPath(); rbCtx.moveTo(x, y); rbCtx.lineTo(exC, eyC); rbCtx.stroke();
            rbCtx.setLineDash([]);
            
            rbCtx.fillStyle = color;
            rbCtx.beginPath(); rbCtx.arc(x, y, 6, 0, Math.PI*2); rbCtx.fill();
            rbCtx.strokeStyle = '#111'; rbCtx.stroke();
        });
    };
    
    const lMax = (rbCal.lidar && rbCal.lidar.max_cm) ? rbCal.lidar.max_cm * 0.06 : 10;
    const iMax = (rbCal.ir && rbCal.ir.max_cm) ? rbCal.ir.max_cm * 0.4 : 10;
    
    drawSensors(rbCal.lidar_placements || [], '#4fc3f7', lMax);
    drawSensors(rbCal.ir_placements || [], '#ff9800', iMax);
}

function rbOnMouseDown(e) {
    const rect = rbCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    let wx = (cx - rbOffsetX) / rbScale;
    let wy = -(cy - rbOffsetY) / rbScale;
    
    const HIT_R = 10 / rbScale;
    
    const checkPlacements = (listType) => {
        const list = rbCal[listType] || [];
        for (let i = 0; i < list.length; i++) {
            if (Math.hypot(wx - list[i].x, wy - list[i].y) < HIT_R) {
                rbDragging = { listType, i };
                return true;
            }
        }
        return false;
    };
    
    if (checkPlacements('lidar_placements')) return;
    if (checkPlacements('ir_placements')) return;
}

function rbOnMouseMove(e) {
    if (!rbDragging) return;
    const rect = rbCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    let wx = (cx - rbOffsetX) / rbScale;
    let wy = -(cy - rbOffsetY) / rbScale;
    
    const p = rbCal[rbDragging.listType][rbDragging.i];
    p.x = wx;
    p.y = wy;
    
    rbUpdateFormPlacement(p);
    rbRender();
}

function rbOnMouseUp(e) {
    if (rbDragging) {
        rbDragging = null;
        rbSyncLive();
    }
}

window.addEventListener('DOMContentLoaded', rbInit);
