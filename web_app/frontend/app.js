const $ = (id) => document.getElementById(id);

function getUserId() {
  let uid = localStorage.getItem("race_user_id");
  if (!uid) {
    uid = crypto.randomUUID();
    localStorage.setItem("race_user_id", uid);
  }
  return uid;
}

const stateLabels = {
  idle: "Ready",
  starting: "Starting",
  waiting: "Waiting",
  racing: "Racing",
  finished: "Finished",
  crashed: "Crashed",
  stopped: "Stopped",
  stopping: "Stopping",
  error: "Error",
};

let lastModalRunId = "";

function formatLap(value) {
  if (value === null || value === undefined || Number(value) <= 0) return "--";
  return `${Number(value).toFixed(3)}s`;
}

function formatDistance(value) {
  return `${Math.max(0, Number(value || 0)).toFixed(1)} m`;
}

function formatSpeed(value) {
  if (value === null || value === undefined) return "--";
  return `${Number(value).toFixed(1)} km/h`;
}

function shortDate(value) {
  if (!value) return "--";
  return String(value).replace("T", " ").slice(0, 16);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": getUserId(),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, 3600);
}

function renderStatus(status) {
  const label = stateLabels[status.status] || status.status || "Ready";
  $("statusPill").textContent = label;
  $("heroMessage").textContent = status.message || "Ready";
  $("runId").textContent = status.run_id ? `Run ${status.run_id}` : "No active run";
  $("progressText").textContent = `${Number(status.progress || 0).toFixed(0)}%`;
  $("progressBar").style.width = `${Math.min(100, Number(status.progress || 0))}%`;
  $("lapTime").textContent = formatLap(status.lap_time);
  $("distance").textContent = formatDistance(status.distance);
  $("speed").textContent = formatSpeed(status.speed);
  $("step").textContent = status.step_count ?? "--";

  const titleByStatus = {
    idle: "AI Driver Ready",
    starting: "Warming Up",
    waiting: "Waiting for Simulator",
    racing: "Race in Progress",
    finished: "Clean Finish",
    crashed: "Run Ended",
    error: "Check Simulator",
    stopped: "Race Stopped",
  };
  $("heroTitle").textContent = titleByStatus[status.status] || "AI Driver Ready";

  $("raceButton").disabled = Boolean(status.running);
  $("stopButton").disabled = !status.running;

  if (status.run_id && status.run_id !== lastModalRunId && ["finished", "crashed", "error"].includes(status.status)) {
    lastModalRunId = status.run_id;
    showResultModal(status);
  }
}

function renderSimulator(simulator) {
  if (simulator.torcs_path && !$("torcsPath").value) {
    $("torcsPath").value = simulator.torcs_path;
  }
  if (simulator.torcs_args && !$("torcsArgs").value) {
    $("torcsArgs").value = simulator.torcs_args;
  }
}

function renderLeaderboard(rows) {
  if (!rows.length) {
    $("leaderboardRows").innerHTML = '<tr><td colspan="4">No results yet.</td></tr>';
    return;
  }
  const myId = getUserId();
  $("leaderboardRows").innerHTML = rows
    .map((row, index) => {
      const isMe = row.user_id === myId;
      const highlight = isMe ? ' class="my-row"' : "";
      const name = row.username || "Anonymous";
      return `
        <tr${highlight}>
          <td>${index + 1}</td>
          <td>${name}${isMe ? " ★" : ""}</td>
          <td>${formatLap(row.lap_time)}</td>
          <td>${row.races ?? "--"}</td>
        </tr>
      `;
    })
    .join("");
}

async function loadProfile() {
  try {
    const profile = await api("/api/profile");
    const name = profile.username || "";
    if (name && name !== "Anonymous") {
      $("driverName").value = name;
    }
  } catch (_) {}
}

async function saveProfile() {
  const name = $("driverName").value.trim();
  if (!name) return;
  try {
    await api("/api/profile", {
      method: "POST",
      body: JSON.stringify({ username: name }),
    });
    showToast(`Name saved: ${name}`);
  } catch (err) {
    showToast(err.message);
  }
}

async function refreshGarage() {
  const payload = await api("/api/garage");
  $("setupName").textContent = payload.setup_name || "Rule Fast";
  $("garageStatus").textContent = payload.status ? `${payload.status} and ready` : "Ready";
  $("bestLap").textContent = formatLap(payload.best_lap);
  $("reliability").textContent = payload.reliability || "Stable";
}

async function loadTracks() {
  try {
    const data = await api("/api/tracks");
    const tracks = data.tracks || [];
    if (!tracks.length) return;
    const select = $("trackSelect");
    const currentVal = select.value;
    select.innerHTML = tracks
      .map((t) => `<option value="${t.category}|${t.name}">${t.label}</option>`)
      .join("");
    if (select.querySelector(`option[value="${currentVal}"]`)) {
      select.value = currentVal;
    }
  } catch (_) {
    // keep default option
  }
}

async function loadSettings() {
  try {
    const s = await api("/api/settings");
    if (s.torcs_path) $("torcsPath").value = s.torcs_path;
    if (s.torcs_args) $("torcsArgs").value = s.torcs_args;
    if (s.laps) $("lapsInput").value = s.laps;
    if (s.driver_config) $("driverConfig").value = s.driver_config;
    if (s.track_category && s.track_name) {
      const val = `${s.track_category}|${s.track_name}`;
      if ($("trackSelect").querySelector(`option[value="${val}"]`)) {
        $("trackSelect").value = val;
      }
    }
  } catch (_) {
    // ignore
  }
}

async function refreshAll() {
  try {
    const [status, leaderboardPayload, simulatorPayload] = await Promise.all([
      api("/api/status"),
      api("/api/leaderboard"),
      api("/api/simulator/status"),
    ]);
    renderStatus(status);
    renderSimulator(simulatorPayload);
    renderLeaderboard(leaderboardPayload.leaderboard || []);
    await refreshGarage();
  } catch (error) {
    showToast(error.message);
  }
}

async function launchSimulator() {
  try {
    await saveSettings(false);
    $("launchButton").disabled = true;
    $("heroTitle").textContent = "Opening TORCS...";
    $("heroMessage").textContent = "Launching simulator in race-ready mode. Please wait.";
    const sim = await api("/api/simulator/launch", { method: "POST" });
    renderSimulator(sim);
    showToast("TORCS is ready. Click Race Now to start.");
    await refreshAll();
  } catch (error) {
    showToast(error.message);
    await refreshAll();
  } finally {
    $("launchButton").disabled = false;
  }
}

async function saveSettings(showMessage = true) {
  const torcsPath = $("torcsPath").value.trim();
  const torcsArgs = $("torcsArgs").value.trim();
  const trackVal = $("trackSelect").value || "road|corkscrew";
  const [trackCategory, trackName] = trackVal.split("|");
  const laps = $("lapsInput").value.trim() || "1";
  const driverConfig = $("driverConfig").value;
  await api("/api/settings", {
    method: "POST",
    body: JSON.stringify({
      torcs_path: torcsPath,
      torcs_args: torcsArgs,
      track_category: trackCategory,
      track_name: trackName,
      laps,
      driver_config: driverConfig,
    }),
  });
  if (showMessage) showToast("Settings saved.");
}

async function startRace() {
  try {
    const trackVal = $("trackSelect").value || "road|corkscrew";
    const [trackCategory, trackName] = trackVal.split("|");
    const laps = parseInt($("lapsInput").value, 10) || 1;
    const driverConfig = $("driverConfig").value;

    await saveSettings(false);

    $("raceButton").disabled = true;
    $("heroTitle").textContent = "Preparing Race...";
    $("heroMessage").textContent = "Navigating TORCS to the race screen, then connecting the driver. Please wait.";

    await api("/api/race/start", {
      method: "POST",
      body: JSON.stringify({
        track_category: trackCategory,
        track_name: trackName,
        laps,
        driver_config: driverConfig,
      }),
    });
    await refreshAll();
  } catch (error) {
    $("raceButton").disabled = false;
    showToast(error.message);
    await refreshAll();
  }
}

async function stopRace() {
  try {
    await api("/api/race/stop", { method: "POST" });
    await refreshAll();
  } catch (error) {
    showToast(error.message);
  }
}

$("raceButton").addEventListener("click", startRace);
$("stopButton").addEventListener("click", stopRace);
$("launchButton").addEventListener("click", launchSimulator);
$("saveSettings").addEventListener("click", () => saveSettings(true));
$("saveName").addEventListener("click", saveProfile);
$("raceAgain").addEventListener("click", () => {
  $("resultModal").hidden = true;
  startRace();
});
$("viewResults").addEventListener("click", () => {
  $("resultModal").hidden = true;
  document.querySelector(".leaderboard-panel").scrollIntoView({ behavior: "smooth", block: "start" });
});
$("closeResult").addEventListener("click", () => {
  $("resultModal").hidden = true;
});

function showResultModal(status) {
  const finished = status.status === "finished";
  $("resultBadge").textContent = finished ? "Clean Finish" : "Run Ended";
  $("resultTitle").textContent = finished ? "Race Complete" : "Race Stopped";
  $("resultLap").textContent = formatLap(status.lap_time);
  $("resultDistance").textContent = formatDistance(status.distance);
  $("resultModal").hidden = false;
}

async function init() {
  await loadTracks();
  await Promise.all([loadSettings(), loadProfile()]);
  await refreshAll();
}

init();
setInterval(refreshAll, 1500);
