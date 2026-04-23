/**
 * Training tab — submits /jobs/train, subscribes to the job WS, charts
 * the reward curve with Chart.js, handles cancel + policy load.
 */
'use strict';
console.log('[training.js] v20260423 loaded');

(function () {

// Chart instance
let chart = null;
let jobWs = null;
window._lastJobId = null;

function getJWT() { return sessionStorage.getItem('pysaac_jwt') || ''; }

// --- Chart setup -----------------------------------------------------------
function initChart() {
  if (chart) return;
  const ctx = document.getElementById('reward-chart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'mean reward',
        data: [],
        borderColor: '#4a9eff',
        backgroundColor: 'rgba(74,158,255,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: {
          ticks: { color: '#777', font: { size: 10 } },
          grid:  { color: '#222' },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function pushRewardPoint(step, reward) {
  if (!chart) initChart();
  chart.data.labels.push(step);
  chart.data.datasets[0].data.push(reward);
  // Keep last 200 points to avoid chart blowup.
  if (chart.data.labels.length > 200) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update('none');
}

function resetChart() {
  if (chart) {
    chart.data.labels = [];
    chart.data.datasets[0].data = [];
    chart.update('none');
  }
}

// --- Stats panel -----------------------------------------------------------
function setStats({ step, reward, fps, state }) {
  if (step   != null) document.getElementById('ts-step').textContent   = step.toLocaleString();
  if (reward != null) document.getElementById('ts-reward').textContent = reward.toFixed(3);
  if (fps    != null) document.getElementById('ts-fps').textContent    = fps.toFixed(0);
  if (state  != null) document.getElementById('ts-state').textContent  = state;
}

// --- Job WS ---------------------------------------------------------------
window.handleLiveTrainingMsg = function(msg) {
  switch (msg.kind) {
    case 'training_progress':
      pushRewardPoint(msg.step, msg.mean_reward);
      setStats({ step: msg.step, reward: msg.mean_reward, fps: msg.fps });
      break;

    case 'training_state':
      setStats({ state: msg.state });
      pysaac.log(`Training state → ${msg.state}`, msg.state === 'failed' ? 'err' : 'info');
      if (msg.state !== 'running') setJobInFlight(false);
      if (msg.state === 'done') {
        document.getElementById('btn-load-policy').disabled = false;
        pysaac.log('Training complete!', 'ok');
      }
      break;

    case 'training_log':
      pysaac.log(`Training: ${msg.msg}`, 'info');
      break;

    case 'training_weights':
      // Currently weights are automatically applied to the engine.
      // We could add a visual heatmap here in the future.
      break;

    default: break;
  }
};

// --- UI helpers ------------------------------------------------------------
function setJobInFlight(inFlight) {
  document.getElementById('btn-start').style.display = inFlight ? 'none' : '';
  document.getElementById('btn-stop').style.display  = inFlight ? '' : 'none';
}

// --- Public API (called from index.html onclick) ---------------------------
window.startJob = async function startJob() {
  const totalSteps = parseInt(document.getElementById('f-steps').value, 10);
  const nEnvs   = parseInt(document.getElementById('f-n-envs').value, 10);
  const lr      = parseFloat(document.getElementById('f-lr').value);
  const device  = document.getElementById('f-device').value;
  const sameScene = document.getElementById('f-same-scene').checked;
  const savePath = document.getElementById('f-save-path').value;
  // const resume = document.getElementById('f-resume')?.checked || false;
  // const saveEvery = parseInt(document.getElementById('f-save-every')?.value || '50000', 10);
  const jwt     = getJWT();

  if (!jwt) { pysaac.log('No JWT — reload page and enter token', 'err'); return; }

  resetChart();
  document.getElementById('eval-card')?.classList.remove('visible');
  setStats({ step: 0, reward: 0, fps: 0, state: 'starting…' });
  setJobInFlight(true);
  pysaac.log(`Starting live training: steps=${totalSteps}, n_envs=${nEnvs}`);

  try {
    const r = await fetch('/gui/training/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + jwt },
      body: JSON.stringify({ 
          total_timesteps: totalSteps, 
          n_envs: nEnvs, 
          learning_rate: lr,
          device: device,
          same_scene: sameScene,
          save_path: savePath,
          resume: false,
          save_every: 50000
      }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      pysaac.log('Start failed: ' + (err.detail || r.status), 'err');
      setJobInFlight(false);
      return;
    }
    pysaac.log('Live training started.', 'info');
  } catch (e) {
    pysaac.log('Network error: ' + e, 'err');
    setJobInFlight(false);
  }
};

window.cancelJob = async function cancelJob() {
  const jwt = getJWT();
  try {
    const r = await fetch('/gui/training/stop', {
      method: 'POST', headers: { Authorization: 'Bearer ' + jwt },
    });
    pysaac.log(`Stop training requested`, 'info');
  } catch (e) {
    pysaac.log('Stop error: ' + e, 'err');
  }
};

// Init chart eagerly so it renders in the correct panel size.
window.addEventListener('load', () => {
  setTimeout(initChart, 100);
  setJobInFlight(false);
});

})();
