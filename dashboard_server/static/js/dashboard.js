// static/js/dashboard.js
const POLL_INTERVAL = 3000; // ms
let lastAlertId = 0;

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error('Fetch failed ' + res.status);
  return res.json();
}

function renderAlerts(alerts) {
  const container = document.getElementById('alertsList');
  container.innerHTML = '';
  alerts.forEach(a => {
    const div = document.createElement('div');
    div.className = 'alert-item';
    div.innerHTML = `<strong>${a.camera || 'Unknown'}</strong><div>${a.message}</div><div style="font-size:12px;color:#7fb0d6">${a.timestamp}</div>
      ${a.snapshot_path ? `<img src="${a.snapshot_path}" style="width:100%;margin-top:6px;border-radius:6px">` : ''}`;
    container.prepend(div);
  });
  if (alerts.length && alerts[alerts.length-1].id > lastAlertId) {
    // new alert arrived
    try { document.getElementById('alarm').play().catch(()=>{}); } catch(e) {}
    lastAlertId = alerts[alerts.length-1].id;
  }
}

async function pollAlerts() {
  try {
    const alerts = await fetchJSON('/alerts/');
    renderAlerts(alerts.reverse());
  } catch (e) {
    console.error('Alerts poll failed', e);
  }
}

async function pollStatus() {
  try {
    const s = await fetchJSON('/status');
    // not rendering camera tiles fully here — simple placeholder
    const cameras = document.getElementById('cameras');
    cameras.innerHTML = '';
    for (const [name, info] of Object.entries(s)) {
      const c = document.createElement('div');
      c.className = 'card';
      c.innerHTML = `<h4>${name} ${info.status==='active'?'<span style="color:#29d37a">●</span>':'<span style="color:#ff6b6b">●</span>'}</h4>
                     ${info.last_snapshot ? `<img src="${info.last_snapshot}">` : `<div style="height:180px;background:#07182a;display:flex;align-items:center;justify-content:center;color:#547ea2">No snapshot</div>`}
                     <div style="font-size:12px;color:#7fb0d6;margin-top:6px">Last seen: ${info.last_seen || '-'}</div>`;
      cameras.appendChild(c);
    }
  } catch (e) {
    console.error('Status poll failed', e);
  }
}

window.onload = function(){
  pollAlerts(); pollStatus();
  setInterval(pollAlerts, POLL_INTERVAL);
  setInterval(pollStatus, POLL_INTERVAL);
  // get username via /me
  fetch('/me').then(r=>r.json()).then(d=>{
    document.getElementById('userLabel').innerText = d.logged_in_as;
  }).catch(()=>{ document.getElementById('userLabel').innerText = ''});
}

