const state = {
  data: null,
  cropValues: {},
  yieldValues: {},
  fields: [],
  theme: "day",
};

const fieldNames = ["North Field", "Canal Plot", "Orchard Edge", "Dry Ridge", "Trial Bed", "Market Block"];

const qs = (selector) => document.querySelector(selector);
const qsa = (selector) => [...document.querySelectorAll(selector)];
const fmtPct = (value) => value == null ? "Ranked" : `${(value * 100).toFixed(2)}%`;
const fmtNum = (value) => Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });

async function api(path, body) {
  const options = body ? {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  } : {};
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`Request failed: ${path}`);
  return response.json();
}

function setTheme(theme) {
  state.theme = theme;
  document.body.classList.toggle("night", theme === "night");
  qs("#themeToggle").textContent = theme === "night" ? "Day mode" : "Night mode";
}

function metricCard(label, value, detail = "") {
  return `<article class="metric-card"><span>${label}</span><strong>${value}</strong><small>${detail}</small></article>`;
}

function renderOverview() {
  const crop = state.data.cropMetrics[0];
  const yieldModel = state.data.yieldMetrics[0];
  const forecast = state.data.yieldForecast;
  qs("#overviewMetrics").innerHTML = [
    metricCard("Crop accuracy", fmtPct(crop.accuracy), crop.model),
    metricCard("Crop profiles", "22", "Recommendation classes"),
    metricCard("Yield R2", fmtPct(yieldModel.r2), yieldModel.model),
    metricCard("2025 wheat outlook", `${fmtNum(forecast.predicted_hg_per_ha_yield)} Hg/Ha`, "Baseline forecast"),
  ].join("");
  qs("#heroYield").textContent = fmtNum(forecast.predicted_hg_per_ha_yield);
  qs("#cropModelName").textContent = crop.model;
}

function renderPresetPills() {
  const wrap = qs("#presetPills");
  wrap.innerHTML = Object.keys(state.data.cropPresets).map((name, index) => (
    `<button class="preset-pill ${index === 0 ? "active" : ""}" data-preset="${name}" type="button">${name}</button>`
  )).join("");
  wrap.addEventListener("click", (event) => {
    const button = event.target.closest("[data-preset]");
    if (!button) return;
    qsa(".preset-pill").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    state.cropValues = { ...state.data.cropPresets[button.dataset.preset] };
    renderCropControls();
    runCrop();
  });
}

function renderCropControls() {
  const labels = {
    N: "Nitrogen",
    P: "Phosphorus",
    K: "Potassium",
    temperature: "Temperature",
    humidity: "Humidity",
    ph: "Soil pH",
    rainfall: "Rainfall",
  };
  qs("#cropControls").innerHTML = state.data.cropFeatures.map((feature) => {
    const value = state.cropValues[feature];
    const type = ["temperature", "humidity", "ph"].includes(feature) ? "range" : "number";
    const min = feature === "ph" ? 3.5 : feature === "temperature" ? 5 : feature === "humidity" ? 10 : 0;
    const max = feature === "ph" ? 9.5 : feature === "temperature" ? 45 : feature === "humidity" ? 100 : 250;
    const step = feature === "ph" || feature === "temperature" || feature === "humidity" ? 0.1 : 1;
    return `
      <div class="control">
        <label for="crop-${feature}">${labels[feature]} <span id="crop-${feature}-value">${value}</span></label>
        <input id="crop-${feature}" data-crop-feature="${feature}" type="${type}" min="${min}" max="${max}" step="${step}" value="${value}" />
      </div>
    `;
  }).join("");
  qsa("[data-crop-feature]").forEach((input) => {
    input.addEventListener("input", () => {
      state.cropValues[input.dataset.cropFeature] = Number(input.value);
      qs(`#crop-${input.dataset.cropFeature}-value`).textContent = input.value;
    });
  });
}

function renderStressFeatureOptions() {
  qs("#stressFeature").innerHTML = state.data.cropFeatures.map((feature) => (
    `<option value="${feature}" ${feature === "rainfall" ? "selected" : ""}>Stress: ${feature}</option>`
  )).join("");
  qs("#stressFeature").addEventListener("change", runCrop);
}

async function runCrop() {
  const payload = { ...state.cropValues, stressFeature: qs("#stressFeature").value };
  const result = await api("/api/crop", payload);
  qs("#topCrop").textContent = result.topCrop;
  qs("#heroCrop").textContent = result.topCrop;
  qs("#topConfidence").textContent = fmtPct(result.recommendations[0].confidence);
  qs("#readinessLabel").textContent = `${result.readiness.score}/100 ${result.readiness.band}`;
  qs("#heroReadiness").textContent = `${result.readiness.score}/100`;
  qs("#readinessBar").style.width = `${result.readiness.score}%`;
  qs("#cropRanking").innerHTML = result.recommendations.map((item) => (
    `<div class="rank-row"><span>${item.crop}</span><strong>${fmtPct(item.confidence)}</strong></div>`
  )).join("");
  qs("#actionList").innerHTML = result.actions.map((item) => (
    `<div class="action-row"><span><strong>${item.area}</strong><br><small>${item.signal}</small></span><strong>${item.action}</strong></div>`
  )).join("");
  qs("#sensitivityLabel").textContent = qs("#stressFeature").value;
  drawLineChart(qs("#sensitivityChart"), result.sensitivity.map((row) => row.x), result.sensitivity.map((row) => row.confidence || 0), { percent: true });
}

function renderYieldControls() {
  const forecast = state.data.yieldForecast;
  state.yieldValues = {
    Year: Number(forecast.Year),
    average_rain_fall_mm_per_year: Number(forecast.average_rain_fall_mm_per_year),
    pesticides_tonnes: Number(forecast.pesticides_tonnes),
    avg_temp: Number(forecast.avg_temp),
  };
  const labels = {
    Year: "Year",
    average_rain_fall_mm_per_year: "Average rainfall",
    pesticides_tonnes: "Pesticides",
    avg_temp: "Average temperature",
  };
  qs("#yieldControls").innerHTML = state.data.yieldFeatures.map((feature) => `
    <div class="control">
      <label for="yield-${feature}">${labels[feature]}</label>
      <input id="yield-${feature}" data-yield-feature="${feature}" type="number" value="${state.yieldValues[feature].toFixed(feature === "Year" ? 0 : 2)}" ${feature === "Year" ? "disabled" : ""} />
    </div>
  `).join("");
  qsa("[data-yield-feature]").forEach((input) => {
    input.addEventListener("input", () => {
      state.yieldValues[input.dataset.yieldFeature] = Number(input.value);
    });
  });
}

async function runYield() {
  const result = await api("/api/yield", state.yieldValues);
  qs("#yieldPrediction").textContent = `${fmtNum(result.prediction)} Hg/Ha`;
  qs("#yieldDelta").textContent = `${result.delta >= 0 ? "+" : ""}${fmtNum(result.delta)} vs baseline`;
  renderStressGrid(result.stress);
}

function renderStressGrid(rows) {
  qs("#stressGrid").innerHTML = rows.map((row) => `
    <div class="stress-cell">
      <span>Rain ${row.rainfallDelta > 0 ? "+" : ""}${row.rainfallDelta}% / Temp ${row.tempDelta > 0 ? "+" : ""}${row.tempDelta}</span>
      <strong>${fmtNum(row.prediction)}</strong>
    </div>
  `).join("");
}

function renderFieldPlanner() {
  const scenarioNames = Object.keys(state.data.cropPresets);
  state.fields = fieldNames.map((field, index) => ({
    field,
    scenario: scenarioNames[index % scenarioNames.length],
    acres: 5 + index,
  }));
  qs("#fieldGrid").innerHTML = state.fields.map((item, index) => `
    <article class="field-card" data-field-card="${index}">
      <h3>${item.field}</h3>
      <div class="control">
        <label>Scenario</label>
        <select data-field-scenario="${index}">
          ${scenarioNames.map((name) => `<option value="${name}" ${name === item.scenario ? "selected" : ""}>${name}</option>`).join("")}
        </select>
      </div>
      <div class="control">
        <label>Acres <span id="field-acres-${index}">${item.acres}</span></label>
        <input type="range" min="1" max="30" value="${item.acres}" data-field-acres="${index}" />
      </div>
      <div class="result-hero"><span id="field-crop-${index}">--</span><strong id="field-ready-${index}">--</strong></div>
    </article>
  `).join("");
  qsa("[data-field-scenario]").forEach((input) => {
    input.addEventListener("change", () => {
      state.fields[Number(input.dataset.fieldScenario)].scenario = input.value;
      runFieldPlan();
    });
  });
  qsa("[data-field-acres]").forEach((input) => {
    input.addEventListener("input", () => {
      const index = Number(input.dataset.fieldAcres);
      state.fields[index].acres = Number(input.value);
      qs(`#field-acres-${index}`).textContent = input.value;
      runFieldPlan();
    });
  });
}

async function runFieldPlan() {
  const result = await api("/api/field-plan", { fields: state.fields });
  qs("#fieldPlanRows").innerHTML = result.fields.map((row) => `
    <tr><td>${row.field}</td><td>${row.scenario}</td><td>${row.acres}</td><td>${row.crop}</td><td>${row.readiness}/100</td></tr>
  `).join("");
  result.fields.forEach((row, index) => {
    qs(`#field-crop-${index}`).textContent = row.crop;
    qs(`#field-ready-${index}`).textContent = `${row.readiness}/100 ${row.band}`;
  });
  drawBarChart(qs("#fieldMixChart"), result.mix.map((row) => row.crop), result.mix.map((row) => row.acres));
}

function renderCropLibrary() {
  const filters = {
    groupFilter: ["All", ...new Set(state.data.cropProfiles.map((crop) => crop.group))],
    seasonFilter: ["All", ...new Set(state.data.cropProfiles.map((crop) => crop.season))],
    waterFilter: ["All", ...new Set(state.data.cropProfiles.map((crop) => crop.water))],
  };
  Object.entries(filters).forEach(([id, values]) => {
    qs(`#${id}`).innerHTML = values.map((value) => `<option value="${value}">${value}</option>`).join("");
    qs(`#${id}`).addEventListener("change", drawCropGallery);
  });
  drawCropGallery();
}

function drawCropGallery() {
  const group = qs("#groupFilter").value;
  const season = qs("#seasonFilter").value;
  const water = qs("#waterFilter").value;
  const crops = state.data.cropProfiles.filter((crop) => (
    (group === "All" || crop.group === group) &&
    (season === "All" || crop.season === season) &&
    (water === "All" || crop.water === water)
  ));
  qs("#cropGallery").innerHTML = crops.map((crop) => `
    <article class="crop-card">
      <img src="${crop.image}" alt="${crop.name} crop" loading="lazy" />
      <div class="crop-body">
        <div class="tag-row"><span class="tag">${crop.group}</span><span class="tag">${crop.water} water</span><span class="tag">${crop.season}</span></div>
        <h3>${crop.name}</h3>
        <p class="muted">${crop.duration} / ${crop.soil}</p>
      </div>
    </article>
  `).join("");
}

function renderMonitor() {
  qs("#cropLeaderboard").innerHTML = state.data.cropMetrics.map((row) => `
    <div class="leader-row"><span>${row.model}<br><small>F1 ${fmtPct(row.f1)}</small></span><strong>${fmtPct(row.accuracy)}</strong></div>
  `).join("");
  qs("#yieldLeaderboard").innerHTML = state.data.yieldMetrics.map((row) => `
    <div class="leader-row"><span>${row.model}<br><small>RMSE ${Number(row.rmse).toFixed(2)}</small></span><strong>${fmtPct(row.r2)}</strong></div>
  `).join("");
  drawBarChart(qs("#cropImportanceChart"), state.data.cropImportance.map((row) => row.feature), state.data.cropImportance.map((row) => row.importance));
  drawBarChart(qs("#yieldImportanceChart"), state.data.yieldImportance.map((row) => row.feature), state.data.yieldImportance.map((row) => row.importance));
}

function setupReveals() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("visible");
    });
  }, { threshold: 0.12 });
  qsa(".reveal").forEach((element) => observer.observe(element));
}

function drawLineChart(canvas, labels, values, options = {}) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width = canvas.clientWidth * devicePixelRatio;
  const height = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const pad = 28;
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--line");
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.stroke();
  ctx.strokeStyle = getComputedStyle(document.body).getPropertyValue("--green");
  ctx.lineWidth = 3;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = pad + (index / Math.max(values.length - 1, 1)) * (w - pad * 2);
    const y = h - pad - ((value - min) / Math.max(max - min, 0.001)) * (h - pad * 2);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawBarChart(canvas, labels, values) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width = canvas.clientWidth * devicePixelRatio;
  const height = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const max = Math.max(...values, 1);
  const pad = 28;
  const gap = 10;
  const barW = (w - pad * 2 - gap * (values.length - 1)) / Math.max(values.length, 1);
  ctx.fillStyle = getComputedStyle(document.body).getPropertyValue("--green");
  values.forEach((value, index) => {
    const barH = (value / max) * (h - pad * 2);
    const x = pad + index * (barW + gap);
    const y = h - pad - barH;
    ctx.fillRect(x, y, Math.max(barW, 3), barH);
  });
}

function setupFieldScan() {
  const canvas = qs("#scanCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const nodes = Array.from({ length: 34 }, () => ({
    x: Math.random(),
    y: Math.random(),
    radius: 1.5 + Math.random() * 2.8,
    pulse: Math.random() * Math.PI * 2,
    drift: 0.0006 + Math.random() * 0.0012,
  }));
  let pointer = { x: innerWidth * 0.5, y: innerHeight * 0.5 };

  function resize() {
    canvas.width = innerWidth * devicePixelRatio;
    canvas.height = innerHeight * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }

  function draw() {
    const time = performance.now();
    ctx.clearRect(0, 0, innerWidth, innerHeight);

    const styles = getComputedStyle(document.body);
    const green = styles.getPropertyValue("--green").trim();
    const cyan = styles.getPropertyValue("--cyan").trim();
    const gold = styles.getPropertyValue("--gold").trim();
    const rose = styles.getPropertyValue("--rose").trim();

    const scanY = (time * 0.035) % (innerHeight + 180) - 90;
    const scan = ctx.createLinearGradient(0, scanY - 70, 0, scanY + 70);
    scan.addColorStop(0, "rgba(255,255,255,0)");
    scan.addColorStop(0.48, "rgba(66,198,200,0.18)");
    scan.addColorStop(0.52, "rgba(240,167,52,0.24)");
    scan.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = scan;
    ctx.fillRect(0, scanY - 70, innerWidth, 140);

    ctx.strokeStyle = cyan;
    ctx.globalAlpha = 0.16;
    ctx.lineWidth = 1;
    for (let x = ((time * 0.012) % 72) - 72; x < innerWidth + 72; x += 72) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + innerHeight * 0.18, innerHeight);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    nodes.forEach((node, index) => {
      node.y += node.drift;
      if (node.y > 1.06) {
        node.y = -0.06;
        node.x = Math.random();
      }
      const x = node.x * innerWidth + Math.sin(time / 1200 + index) * 18;
      const y = node.y * innerHeight;
      const distance = Math.hypot(pointer.x - x, pointer.y - y);
      const pull = Math.max(0, 1 - distance / 220);
      const radius = node.radius + Math.sin(time / 500 + node.pulse) * 0.8 + pull * 5;
      ctx.fillStyle = index % 3 === 0 ? gold : index % 3 === 1 ? cyan : green;
      ctx.globalAlpha = 0.22 + pull * 0.34;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(1, radius), 0, Math.PI * 2);
      ctx.fill();

      if (pull > 0.08) {
        ctx.strokeStyle = rose;
        ctx.globalAlpha = pull * 0.28;
        ctx.beginPath();
        ctx.moveTo(pointer.x, pointer.y);
        ctx.lineTo(x, y);
        ctx.stroke();
      }
    });
    ctx.globalAlpha = 1;
    requestAnimationFrame(draw);
  }

  addEventListener("pointermove", (event) => {
    pointer = { x: event.clientX, y: event.clientY };
    document.body.style.setProperty("--cursor-x", `${event.clientX}px`);
    document.body.style.setProperty("--cursor-y", `${event.clientY}px`);
  });
  addEventListener("resize", resize);
  resize();
  draw();
}

function setupCardTilt() {
  qsa(".product-card, .metric-card, .field-card, .crop-card").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      const box = card.getBoundingClientRect();
      const x = (event.clientX - box.left) / box.width - 0.5;
      const y = (event.clientY - box.top) / box.height - 0.5;
      card.style.setProperty("--tilt-x", `${(-y * 3).toFixed(2)}deg`);
      card.style.setProperty("--tilt-y", `${(x * 4).toFixed(2)}deg`);
    });
    card.addEventListener("pointerleave", () => {
      card.style.setProperty("--tilt-x", "0deg");
      card.style.setProperty("--tilt-y", "0deg");
    });
  });
}

async function init() {
  state.data = await api("/api/state");
  state.cropValues = { ...state.data.cropPresets["Balanced field"] };
  renderOverview();
  renderPresetPills();
  renderCropControls();
  renderStressFeatureOptions();
  renderYieldControls();
  renderFieldPlanner();
  renderCropLibrary();
  renderMonitor();
  setupReveals();
  setupFieldScan();
  setupCardTilt();
  qs("#runCrop").addEventListener("click", runCrop);
  qs("#runYield").addEventListener("click", runYield);
  qs("#themeToggle").addEventListener("click", () => setTheme(state.theme === "day" ? "night" : "day"));
  await runCrop();
  await runYield();
  await runFieldPlan();
}

init().catch((error) => {
  console.error(error);
  document.body.insertAdjacentHTML("afterbegin", `<div style="padding:1rem;background:#7b1e1e;color:white">Failed to load AgriSense: ${error.message}</div>`);
});
