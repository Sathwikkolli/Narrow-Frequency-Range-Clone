/* Spot the Deepfake - classroom demo.
   Every clip in the game is synthetic; "correct" means calling it AI.
   No dependencies, no network beyond this folder. */

(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = {
    screens:   [...document.querySelectorAll(".screen")],
    setupWhy:  $("#setup-why"),
    speaker:   $("#title-speaker"),
    start:     $("#btn-start"),
    pips:      $("#pips"),
    roundN:    $("#round-n"),
    roundTot:  $("#round-total"),
    viz:       $("#viz"),
    play:      $("#btn-play"),
    playCount: $("#play-count"),
    clipDur:   $("#clip-dur"),
    conf:      $("#confidence"),
    confOut:   $("#confidence-out"),
    next:      $("#btn-next"),
    lab:       $("#btn-lab"),
    again:     $("#btn-again"),
    back:      $("#btn-back"),
    labGrid:   $("#lab-grid"),
    tally:     $("#tally"),
    confetti:  $("#confetti"),
  };

  const state = {
    config: null,
    game: [],        // clips with use:true, in order
    i: 0,
    plays: 0,
    answers: [],     // {clip, guess, confidence, correct}
  };

  /* ---------------- screens ---------------- */

  function show(id) {
    el.screens.forEach((s) => { s.hidden = s.id !== id; });
    window.scrollTo(0, 0);
  }

  function fail(why) {
    el.setupWhy.textContent = why;
    show("screen-setup");
  }

  /* ---------------- audio ---------------- */

  const audio = new Audio();
  audio.preload = "auto";
  audio.crossOrigin = "anonymous";

  let ctx = null, analyser = null, freq = null, srcNode = null;

  function initAudioGraph() {
    if (ctx) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;                          // visualiser is optional, playback is not
    try {
      ctx = new AC();
      analyser = ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.8;
      freq = new Uint8Array(analyser.frequencyBinCount);
      srcNode = ctx.createMediaElementSource(audio);
      srcNode.connect(analyser);
      analyser.connect(ctx.destination);
    } catch (e) {
      ctx = analyser = null;                  // fall back to the idle wave
    }
  }

  /* ---------------- visualiser ---------------- */

  const cv = el.viz, g = cv.getContext("2d");
  let t0 = performance.now();

  function sizeCanvas() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect();
    cv.width = Math.max(1, Math.round(r.width * dpr));
    cv.height = Math.max(1, Math.round(r.height * dpr));
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  window.addEventListener("resize", sizeCanvas);

  function draw() {
    requestAnimationFrame(draw);
    const r = cv.getBoundingClientRect();
    const w = r.width, h = r.height;
    if (!w || !h) return;

    g.clearRect(0, 0, w, h);
    const mid = h / 2;
    const bars = 72;
    const gap = 3;
    const bw = Math.max(2, w / bars - gap);

    const grad = g.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0,   "#22d3ee");
    grad.addColorStop(0.5, "#7c5cff");
    grad.addColorStop(1,   "#ff4d6d");
    g.fillStyle = grad;

    const live = analyser && !audio.paused;
    if (live) analyser.getByteFrequencyData(freq);
    const now = (performance.now() - t0) / 1000;

    for (let i = 0; i < bars; i++) {
      let v;
      if (live) {
        // log-ish bin spread so the low end does not swamp the picture
        const lo = Math.floor(Math.pow(i / bars, 1.7) * freq.length);
        const hi = Math.max(lo + 1, Math.floor(Math.pow((i + 1) / bars, 1.7) * freq.length));
        let sum = 0;
        for (let k = lo; k < hi; k++) sum += freq[k];
        v = (sum / (hi - lo)) / 255;
        v = Math.pow(v, 0.85);
      } else {
        // idle: a slow breathing wave so the stage never looks dead
        v = 0.06 + 0.05 * (1 + Math.sin(now * 1.4 + i * 0.28));
      }
      const bh = Math.max(3, v * (h * 0.46));
      const x = i * (bw + gap) + gap / 2;
      g.beginPath();
      if (g.roundRect) g.roundRect(x, mid - bh, bw, bh * 2, bw / 2);
      else g.rect(x, mid - bh, bw, bh * 2);
      g.fill();
    }
  }

  /* ---------------- rounds ---------------- */

  const CONF_WORDS = [
    [10, "Certain it is a person"],
    [30, "Leaning human"],
    [45, "Probably human"],
    [56, "Not sure"],
    [70, "Probably AI"],
    [90, "Leaning AI"],
    [101, "Certain it is AI"],
  ];

  function confWord(v) {
    for (const [ceil, word] of CONF_WORDS) if (v < ceil) return word;
    return "Not sure";
  }

  el.conf.addEventListener("input", () => {
    el.confOut.textContent = confWord(Number(el.conf.value));
  });

  function renderPips() {
    el.pips.innerHTML = "";
    state.game.forEach((_, i) => {
      const d = document.createElement("div");
      d.className = "pip" + (i === state.i ? " on" : i < state.i ? " done" : "");
      el.pips.appendChild(d);
    });
  }

  function loadRound() {
    const clip = state.game[state.i];
    state.plays = 0;
    el.playCount.textContent = "0";
    el.roundN.textContent = String(state.i + 1);
    el.clipDur.textContent = clip.duration ? clip.duration.toFixed(1) + "s" : "—";
    el.conf.value = 50;
    el.confOut.textContent = "Not sure";
    el.play.classList.remove("playing");
    audio.pause();
    audio.src = clip.audio;
    renderPips();
    show("screen-round");
    sizeCanvas();
  }

  function togglePlay() {
    initAudioGraph();
    if (ctx && ctx.state === "suspended") ctx.resume();
    if (audio.paused) {
      audio.currentTime = 0;
      audio.play().then(() => {
        state.plays++;
        el.playCount.textContent = String(state.plays);
        el.play.classList.add("playing");
      }).catch(() => {
        fail("The browser refused to play " + audio.getAttribute("src") +
             ". Check the file exists and that you opened this over http://localhost, not as a file:// path.");
      });
    } else {
      audio.pause();
      el.play.classList.remove("playing");
    }
  }

  el.play.addEventListener("click", togglePlay);
  audio.addEventListener("ended", () => el.play.classList.remove("playing"));

  function answer(guess) {
    const clip = state.game[state.i];
    const correct = guess === clip.label;          // every game clip is "fake"
    state.answers.push({
      clip, guess, correct,
      confidence: Number(el.conf.value),
      plays: state.plays,
    });
    audio.pause();
    el.play.classList.remove("playing");
    reveal(clip, correct, guess);
  }

  document.querySelectorAll("[data-guess]").forEach((b) =>
    b.addEventListener("click", () => answer(b.dataset.guess)));

  /* ---------------- reveal ---------------- */

  function reveal(clip, correct, guess) {
    const v = $("#reveal-verdict");
    v.textContent = correct
      ? "You called it"
      : (guess === "real" ? "It fooled you" : "Not this time");
    v.className = "verdict-line " + (correct ? "hit" : "miss");

    $("#reveal-answer").textContent = clip.label === "fake" ? "AI-generated" : "a real recording";
    $("#reveal-system").textContent = clip.system;
    $("#reveal-kind").textContent = clip.kind;
    $("#reveal-listen").textContent = clip.listen;

    const img = $("#reveal-spec");
    img.src = clip.spectrogram;
    img.alt = "Spectrogram of the " + clip.system + " clip";

    el.next.textContent = state.i + 1 < state.game.length ? "Next clip" : "See the results";
    show("screen-reveal");
  }

  el.next.addEventListener("click", () => {
    state.i++;
    if (state.i < state.game.length) loadRound();
    else final();
  });

  /* ---------------- final ---------------- */

  function final() {
    const hits = state.answers.filter((a) => a.correct).length;
    const den = state.answers.length;
    $("#score-num").textContent = String(hits);
    $("#score-den").textContent = String(den);
    const ring = $("#score-ring");
    ring.style.setProperty("--pct", den ? Math.round((hits / den) * 100) : 0);

    $("#final-head").textContent =
      hits === den ? "Clean sweep" :
      hits === 0   ? "They got you every time" :
                     "Here is the twist";

    el.tally.innerHTML = "";
    state.answers.forEach((a, i) => {
      const row = document.createElement("div");
      row.className = "tally-row";
      row.innerHTML =
        '<span class="tally-mark">' + (a.correct ? "✅" : "❌") + "</span>" +
        '<span class="tally-sys">Clip ' + (i + 1) + ": " + a.clip.system + "</span>" +
        '<span class="lab-tag fake">fake</span>' +
        '<span class="tally-guess">you said ' +
          (a.guess === "fake" ? "AI clone" : "real person") +
          " · " + a.plays + (a.plays === 1 ? " play" : " plays") + "</span>";
      el.tally.appendChild(row);
    });

    show("screen-final");
    if (hits > 0) confetti();
  }

  el.lab.addEventListener("click", () => { buildLab(); show("screen-lab"); });
  el.back.addEventListener("click", () => show("screen-final"));
  el.again.addEventListener("click", () => {
    state.i = 0;
    state.answers = [];
    loadRound();
  });

  /* ---------------- lab ---------------- */

  let labBuilt = false;
  function buildLab() {
    if (labBuilt) return;
    labBuilt = true;
    el.labGrid.innerHTML = "";
    state.config.clips.forEach((c) => {
      const card = document.createElement("div");
      card.className = "lab-card";

      const tags = [];
      tags.push('<span class="lab-tag ' + (c.label === "real" ? "real" : "fake") + '">' +
                (c.label === "real" ? "real" : "fake") + "</span>");
      if (c.phone) tags.push('<span class="lab-tag phone">8 kHz phone band</span>');
      if (c.use)   tags.push('<span class="lab-tag used">in the game</span>');

      card.innerHTML =
        '<div class="lab-top"><span class="lab-name">' + esc(c.system) + "</span>" + tags.join("") + "</div>" +
        '<p class="lab-kind">' + esc(c.kind) + "</p>" +
        '<audio controls preload="none" src="' + esc(c.audio) + '"></audio>' +
        '<img loading="lazy" src="' + esc(c.spectrogram) + '" alt="Spectrogram of the ' + esc(c.system) + ' clip">' +
        '<p class="lab-facts">' +
          "<span><b>source</b> " + esc(c.source_file) + "</span>" +
          "<span><b>original</b> " + (c.original_sample_rate || "?") + " Hz</span>" +
          "<span><b>played at</b> " + (c.played_sample_rate || "?") + " Hz</span>" +
          "<span><b>length</b> " + (c.duration != null ? c.duration + "s" : "?") + "</span>" +
        "</p>";
      el.labGrid.appendChild(card);
    });
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ---------------- confetti ---------------- */

  function confetti() {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const c = el.confetti, x = c.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.style.display = "block";
    c.width = innerWidth * dpr;
    c.height = innerHeight * dpr;
    x.setTransform(dpr, 0, 0, dpr, 0, 0);

    const colors = ["#7c5cff", "#22d3ee", "#ff4d6d", "#ffb020", "#2ee6a8"];
    const bits = Array.from({ length: 130 }, () => ({
      x: Math.random() * innerWidth,
      y: -20 - Math.random() * innerHeight * 0.5,
      w: 5 + Math.random() * 7,
      h: 9 + Math.random() * 9,
      vy: 2 + Math.random() * 3.4,
      vx: -1 + Math.random() * 2,
      a: Math.random() * Math.PI,
      va: -0.12 + Math.random() * 0.24,
      col: colors[(Math.random() * colors.length) | 0],
    }));

    const until = performance.now() + 3200;
    (function tick(now) {
      x.clearRect(0, 0, innerWidth, innerHeight);
      bits.forEach((b) => {
        b.y += b.vy; b.x += b.vx; b.a += b.va;
        x.save();
        x.translate(b.x, b.y);
        x.rotate(b.a);
        x.fillStyle = b.col;
        x.fillRect(-b.w / 2, -b.h / 2, b.w, b.h);
        x.restore();
      });
      if (now < until) requestAnimationFrame(tick);
      else { x.clearRect(0, 0, innerWidth, innerHeight); c.style.display = "none"; }
    })(performance.now());
  }

  /* ---------------- keyboard ---------------- */

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, audio")) return;
    const round  = !$("#screen-round").hidden;
    const revealed = !$("#screen-reveal").hidden;
    if (e.code === "Space" && round) { e.preventDefault(); togglePlay(); }
    else if (round && (e.key === "r" || e.key === "R")) answer("real");
    else if (round && (e.key === "f" || e.key === "F")) answer("fake");
    else if (revealed && (e.key === "Enter" || e.code === "Space")) { e.preventDefault(); el.next.click(); }
    else if (!$("#screen-title").hidden && (e.key === "Enter" || e.code === "Space")) {
      e.preventDefault(); el.start.click();
    }
  });

  /* ---------------- boot ---------------- */

  el.start.addEventListener("click", () => {
    state.i = 0;
    state.answers = [];
    loadRound();
  });

  async function boot() {
    requestAnimationFrame(draw);

    if (location.protocol === "file:") {
      return fail("Open this over a local server, not as a file. Run:  python tools/serve.py");
    }

    let cfg;
    try {
      const res = await fetch("clips.json", { cache: "no-store" });
      if (!res.ok) throw new Error("clips.json returned " + res.status);
      cfg = await res.json();
    } catch (err) {
      return fail("Could not read web/clips.json (" + err.message +
                  "). The audio has not been downloaded and built yet.");
    }

    const game = (cfg.clips || []).filter((c) => c.use).sort((a, b) => a.order - b.order);
    if (game.length < 2) {
      return fail("clips.json has only " + game.length +
                  ' clip(s) marked use:true. The game needs three. Re-run the build with --pick.');
    }

    // fail loudly now rather than mid-demo in front of a class
    for (const c of game) {
      const head = await fetch(c.audio, { method: "HEAD" }).catch(() => null);
      if (!head || !head.ok) {
        return fail("clips.json points at " + c.audio + ", which the server cannot find. " +
                    "Re-run:  python tools/build_web_assets.py");
      }
    }

    state.config = cfg;
    state.game = game;
    el.roundTot.textContent = String(game.length);
    el.speaker.textContent = (cfg.speaker || "Unknown speaker") +
                             (cfg.dataset ? " · " + cfg.dataset : "");
    show("screen-title");
  }

  boot();
})();
