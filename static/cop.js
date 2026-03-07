const map = L.map("map", { zoomControl: false });
L.control.zoom({ position: "topright" }).addTo(map);

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);

map.setView([49.262, -123.25], 12);

const receiverLayer = L.layerGroup().addTo(map);
const uncertaintyLayer = L.layerGroup().addTo(map);
const trackLayer = L.layerGroup().addTo(map);

let initialized = false;

function labelColor(label) {
  if (label === "hostile") return "#c81d4a";
  if (label === "civilian") return "#d97706";
  return "#138a72";
}

function asTime(ts) {
  if (!ts) return "-";
  return ts.replace("T", " ").replace("+00:00", "Z");
}

function setStatus(kind, text) {
  const pill = document.getElementById("statusPill");
  pill.textContent = text;
  pill.className = `pill ${kind}`;
}

function receiverMarker(rx) {
  return L.marker([rx.latitude, rx.longitude], {
    icon: L.divIcon({
      className: "",
      html: '<div class="rx-pin"></div>',
      iconSize: [12, 12],
      iconAnchor: [6, 6],
    }),
  }).bindTooltip(`Receiver ${rx.receiver_id}`, { direction: "top" });
}

function trackMarker(track) {
  const color = labelColor(track.faction);
  const iconHtml = `<div class="track-pin" style="background:${color};color:${color}"></div>`;
  const marker = L.marker([track.lat, track.lon], {
    icon: L.divIcon({
      className: "",
      html: iconHtml,
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    }),
  });

  marker.bindPopup(
    `<strong>${track.track_id}</strong><br>` +
    `Faction: ${track.faction}<br>` +
    `Signal Type: ${track.label}<br>` +
    `Modulation: ${track.modulation}<br>` +
    `Assessment: ${track.assessment}<br>` +
    `Confidence: ${track.confidence.toFixed(3)}<br>` +
    `Uncertainty: ${track.uncertainty_m.toFixed(1)} m<br>` +
    `Hits: ${track.hit_count}<br>` +
    `Last seen: ${asTime(track.last_seen)}`
  );
  return marker;
}

function renderObservations(obs) {
  const body = document.getElementById("obsBody");
  body.innerHTML = "";

  const rows = obs.slice(-30).reverse();
  document.getElementById("obsCount").textContent = `${rows.length}`;

  for (const o of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="mono">${asTime(o.timestamp)}</td>
      <td>${o.receiver_id}</td>
      <td>${o.faction}</td>
      <td>${o.signal_type}</td>
      <td>${o.modulation}</td>
      <td>${o.confidence.toFixed(3)}</td>
      <td>${o.rssi_dbm.toFixed(1)}</td>
      <td>${o.snr_db.toFixed(1)}</td>
    `;
    body.appendChild(tr);
  }
}

function renderState(data) {
  document.getElementById("metricReceivers").textContent = `${data.receiver_count}`;
  document.getElementById("metricTracks").textContent = `${data.track_count}`;
  document.getElementById("metricTime").textContent = asTime(data.server_time);

  receiverLayer.clearLayers();
  uncertaintyLayer.clearLayers();
  trackLayer.clearLayers();

  const bounds = [];
  for (const rx of data.receivers) {
    receiverMarker(rx).addTo(receiverLayer);
    bounds.push([rx.latitude, rx.longitude]);
  }

  for (const track of data.tracks) {
    const color = labelColor(track.faction);
    L.circle([track.lat, track.lon], {
      radius: Math.max(20, track.uncertainty_m),
      color,
      fillColor: color,
      fillOpacity: 0.08,
      weight: 1.5,
    }).addTo(uncertaintyLayer);

    trackMarker(track).addTo(trackLayer);
    bounds.push([track.lat, track.lon]);
  }

  if (!initialized && bounds.length > 0) {
    map.fitBounds(bounds, { padding: [25, 25] });
    initialized = true;
  }

  renderObservations(data.latest_observations);
}

async function refresh() {
  try {
    const res = await fetch("/state");
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    const data = await res.json();
    setStatus("pill-ok", "Live Feed Active");
    renderState(data);
  } catch (err) {
    setStatus("pill-bad", "Reconnect...");
  }
}

refresh();
setInterval(refresh, 1500);
