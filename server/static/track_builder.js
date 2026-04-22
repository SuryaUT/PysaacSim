let tbCanvas, tbCtx;
let tbWalls = [];
let tbDrawing = null;
let tbDragging = null;
let tbBounds = { min_x: 0, max_x: 100, min_y: 0, max_y: 100 };

// World space to canvas coords
let tbScale = 1;
let tbOffsetX = 0;
let tbOffsetY = 0;

function tbInit() {
    tbCanvas = document.getElementById('track-canvas');
    if (!tbCanvas) return;
    tbCtx = tbCanvas.getContext('2d');
    
    tbCanvas.addEventListener('mousedown', tbOnMouseDown);
    tbCanvas.addEventListener('mousemove', tbOnMouseMove);
    tbCanvas.addEventListener('mouseup', tbOnMouseUp);
    tbCanvas.addEventListener('contextmenu', e => { e.preventDefault(); tbOnRightClick(e); });
    
    window.addEventListener('resize', () => {
        if (!document.getElementById('track-panel').classList.contains('hidden')) tbResize();
    });
}

function tbResize() {
    if (!tbCanvas) return;
    const wrap = document.getElementById('track-canvas-wrap');
    tbCanvas.width = wrap.clientWidth;
    tbCanvas.height = wrap.clientHeight;
    tbRender();
}

async function tbLoad() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    if (!jwt) return;
    try {
        const res = await fetch("/gui/track", { headers: { "Authorization": `Bearer ${jwt}` } });
        const data = await res.json();
        tbWalls = data.walls || [];
        tbRender();
        tbUpdateInfo();
    } catch(e) {
        console.error("Failed to load track", e);
    }
}

async function tbLoadFile(event) {
    const file = event.target.files[0];
    if (!file) return;
    const jwt = sessionStorage.getItem('pysaac_jwt');
    const formData = new FormData();
    formData.append('file', file);
    try {
        await fetch('/gui/import/track', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${jwt}` },
            body: formData
        });
        await tbLoad();
        window.pysaac.log('Track YAML imported');
    } catch (e) {
        window.pysaac.log('Import failed', 'err');
    }
    event.target.value = '';
}

function tbSave() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    if (!jwt) return;
    // Download the YAML
    const url = `/gui/export/track.yaml`;
    fetch(url, { headers: { 'Authorization': `Bearer ${jwt}` } })
        .then(r => r.blob())
        .then(blob => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'track.yaml';
            a.click();
        });
}

async function tbReset() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    await fetch('/gui/track/reset', { method: 'POST', headers: { 'Authorization': `Bearer ${jwt}` } });
    await tbLoad();
}

async function tbClear() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    await fetch('/gui/track', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${jwt}` },
        body: JSON.stringify({ walls: [] })
    });
    await tbLoad();
}

function tbRender() {
    if (!tbCtx) return;
    tbCtx.clearRect(0, 0, tbCanvas.width, tbCanvas.height);
    
    // Calculate bounds
    if (tbWalls.length > 0) {
        let xs = [], ys = [];
        for (let w of tbWalls) { xs.push(w[0][0], w[1][0]); ys.push(w[0][1], w[1][1]); }
        tbBounds = { min_x: Math.min(...xs)-20, max_x: Math.max(...xs)+20, min_y: Math.min(...ys)-20, max_y: Math.max(...ys)+20 };
    } else {
        tbBounds = { min_x: -100, max_x: 100, min_y: -100, max_y: 100 };
    }
    
    // Compute scale and offset
    const w = tbBounds.max_x - tbBounds.min_x;
    const h = tbBounds.max_y - tbBounds.min_y;
    tbScale = Math.min(tbCanvas.width / w, tbCanvas.height / h) * 0.9;
    tbOffsetX = tbCanvas.width / 2 - ((tbBounds.min_x + tbBounds.max_x) / 2) * tbScale;
    tbOffsetY = tbCanvas.height / 2 + ((tbBounds.min_y + tbBounds.max_y) / 2) * tbScale; // + because y points down on canvas
    
    const w2c = (x, y) => [x * tbScale + tbOffsetX, -y * tbScale + tbOffsetY];
    
    // Draw Grid
    if (document.getElementById('tb-show-grid').checked) {
        const cellSize = parseFloat(document.getElementById('tb-cell').value) || 10;
        tbCtx.strokeStyle = '#333';
        tbCtx.lineWidth = 1;
        for (let x = Math.floor(tbBounds.min_x/cellSize)*cellSize; x <= tbBounds.max_x; x += cellSize) {
            let [cx1, cy1] = w2c(x, tbBounds.min_y);
            let [cx2, cy2] = w2c(x, tbBounds.max_y);
            tbCtx.beginPath(); tbCtx.moveTo(cx1, cy1); tbCtx.lineTo(cx2, cy2); tbCtx.stroke();
        }
        for (let y = Math.floor(tbBounds.min_y/cellSize)*cellSize; y <= tbBounds.max_y; y += cellSize) {
            let [cx1, cy1] = w2c(tbBounds.min_x, y);
            let [cx2, cy2] = w2c(tbBounds.max_x, y);
            tbCtx.beginPath(); tbCtx.moveTo(cx1, cy1); tbCtx.lineTo(cx2, cy2); tbCtx.stroke();
        }
    }
    
    // Draw Walls
    tbCtx.strokeStyle = '#aaa';
    tbCtx.lineWidth = 4;
    tbCtx.lineCap = 'round';
    for (let w of tbWalls) {
        let [x1, y1] = w2c(w[0][0], w[0][1]);
        let [x2, y2] = w2c(w[1][0], w[1][1]);
        tbCtx.beginPath(); tbCtx.moveTo(x1, y1); tbCtx.lineTo(x2, y2); tbCtx.stroke();
    }
    
    // Draw Live Preview Line
    if (tbDrawing) {
        let [x1, y1] = w2c(tbDrawing.start[0], tbDrawing.start[1]);
        let [x2, y2] = w2c(tbDrawing.curr[0], tbDrawing.curr[1]);
        tbCtx.strokeStyle = 'rgba(233, 30, 99, 0.8)';
        tbCtx.setLineDash([5, 5]);
        tbCtx.beginPath(); tbCtx.moveTo(x1, y1); tbCtx.lineTo(x2, y2); tbCtx.stroke();
        tbCtx.setLineDash([]);
    }
    
    // Draw Handles
    tbCtx.fillStyle = '#ffeb3b';
    for (let w of tbWalls) {
        let [x1, y1] = w2c(w[0][0], w[0][1]);
        let [x2, y2] = w2c(w[1][0], w[1][1]);
        tbCtx.beginPath(); tbCtx.arc(x1, y1, 6, 0, Math.PI*2); tbCtx.fill();
        tbCtx.beginPath(); tbCtx.arc(x2, y2, 6, 0, Math.PI*2); tbCtx.fill();
    }
}

function tbUpdateInfo() {
    const el = document.getElementById('tb-info');
    if (!el) return;
    let len = 0;
    for (let w of tbWalls) {
        len += Math.sqrt(Math.pow(w[1][0]-w[0][0], 2) + Math.pow(w[1][1]-w[0][1], 2));
    }
    el.textContent = `walls  : ${tbWalls.length}\nbounds : x [${tbBounds.min_x.toFixed(1)}, ${tbBounds.max_x.toFixed(1)}] cm\n         y [${tbBounds.min_y.toFixed(1)}, ${tbBounds.max_y.toFixed(1)}] cm\nperim. : ${len.toFixed(1)} cm`;
}

function tbSnap(wx, wy) {
    if (document.getElementById('tb-snap').checked) {
        const cs = parseFloat(document.getElementById('tb-cell').value) || 10;
        return [Math.round(wx / cs) * cs, Math.round(wy / cs) * cs];
    }
    return [wx, wy];
}

function tbOnMouseDown(e) {
    const rect = tbCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    let wx = (cx - tbOffsetX) / tbScale;
    let wy = -(cy - tbOffsetY) / tbScale;
    let [sx, sy] = tbSnap(wx, wy);
    
    // Check handles
    const HIT_R = 10 / tbScale;
    for (let i = 0; i < tbWalls.length; i++) {
        for (let j = 0; j < 2; j++) {
            let px = tbWalls[i][j][0], py = tbWalls[i][j][1];
            if (Math.hypot(wx - px, wy - py) < HIT_R) {
                tbDragging = { i, j };
                return;
            }
        }
    }
    
    // Otherwise start drawing
    tbDrawing = { start: [sx, sy], curr: [sx, sy] };
    tbRender();
}

function tbOnMouseMove(e) {
    const rect = tbCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    let wx = (cx - tbOffsetX) / tbScale;
    let wy = -(cy - tbOffsetY) / tbScale;
    let [sx, sy] = tbSnap(wx, wy);
    
    if (tbDragging) {
        tbWalls[tbDragging.i][tbDragging.j] = [sx, sy];
        tbRender();
    } else if (tbDrawing) {
        tbDrawing.curr = [sx, sy];
        tbRender();
    }
}

function tbOnMouseUp(e) {
    if (tbDragging) {
        tbDragging = null;
        tbUpdateInfo();
        tbSyncLive();
    } else if (tbDrawing) {
        const d = tbDrawing;
        tbDrawing = null;
        if (Math.hypot(d.curr[0]-d.start[0], d.curr[1]-d.start[1]) > 2) {
            tbWalls.push([d.start, d.curr]);
            tbUpdateInfo();
            tbSyncLive();
        }
        tbRender();
    }
}

function tbOnRightClick(e) {
    const rect = tbCanvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    let wx = (cx - tbOffsetX) / tbScale;
    let wy = -(cy - tbOffsetY) / tbScale;
    
    // Find closest wall
    let minDist = 10 / tbScale;
    let best = -1;
    for (let i = 0; i < tbWalls.length; i++) {
        let a = tbWalls[i][0], b = tbWalls[i][1];
        let px = b[0] - a[0], py = b[1] - a[1];
        let norm = px*px + py*py;
        let u =  ((wx - a[0]) * px + (wy - a[1]) * py) / norm;
        if (u > 1) u = 1; else if (u < 0) u = 0;
        let x = a[0] + u * px;
        let y = a[1] + u * py;
        let dist = Math.hypot(x - wx, y - wy);
        if (dist < minDist) {
            minDist = dist;
            best = i;
        }
    }
    
    if (best >= 0) {
        tbWalls.splice(best, 1);
        tbUpdateInfo();
        tbRender();
        tbSyncLive();
    }
}

async function tbSyncLive() {
    const jwt = sessionStorage.getItem('pysaac_jwt');
    if (!jwt) return;
    try {
        await fetch("/gui/track", {
            method: "POST",
            headers: { "Content-Type": "application/json", "Authorization": `Bearer ${jwt}` },
            body: JSON.stringify({ walls: tbWalls })
        });
    } catch(e) {}
}

window.addEventListener('DOMContentLoaded', tbInit);
