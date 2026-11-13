const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Scroll progress bar */
const progress = document.querySelector(".progress");
function updateProgress() {
  const max = document.documentElement.scrollHeight - innerHeight;
  progress.style.setProperty("--p", max > 0 ? scrollY / max : 0);
}
addEventListener("scroll", updateProgress, { passive: true });
updateProgress();

/* Reveal on scroll */
const revealer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      revealer.unobserve(entry.target);
      if (entry.target.classList.contains("versus-card")) countUp(entry.target);
    });
  },
  { rootMargin: "0px 0px -8% 0px", threshold: 0.12 },
);
document.querySelectorAll(".reveal").forEach((el) => revealer.observe(el));

function countUp(root) {
  root.querySelectorAll("[data-count]").forEach((el) => {
    const target = Number(el.dataset.count);
    const unit = el.querySelector("small").outerHTML;
    if (reduceMotion) return;
    const start = performance.now();
    const duration = 1400;
    function frame(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      el.innerHTML = (target * eased).toFixed(1) + unit;
      if (t < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  });
}

/* Roles: one chapter at a time */
const roleIds = ["actor", "judge", "teacher"];
const rolePanels = roleIds.map((id) => document.getElementById(id));
const roleTabs = [...document.querySelectorAll('.role-tabs [role="tab"]')];
const rolesTop = document.getElementById("roles");
let roleIndex = 0;
function showRole(index, { scroll = false, animate = true } = {}) {
  const next = (index + roleIds.length) % roleIds.length;
  const fromLeft = index < roleIndex;
  roleIndex = next;
  rolePanels.forEach((panel, i) => {
    panel.hidden = i !== next;
    panel.classList.remove("enter", "from-left");
    if (i === next && animate) {
      void panel.offsetWidth;
      panel.classList.add("enter");
      panel.classList.toggle("from-left", fromLeft);
    }
    panel.querySelectorAll(".reveal").forEach((el) => el.classList.add("in"));
  });
  roleTabs.forEach((tab, i) => {
    tab.setAttribute("aria-selected", String(i === next));
    tab.tabIndex = i === next ? 0 : -1;
  });
  if (scroll && rolesTop.getBoundingClientRect().top < 0) {
    rolesTop.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
  }
}
document.querySelectorAll("[data-role-go]").forEach((el) =>
  el.addEventListener("click", () => {
    const i = roleIds.indexOf(el.dataset.roleGo);
    showRole(i, { scroll: true });
  }),
);
document.querySelectorAll("[data-role-step]").forEach((el) =>
  el.addEventListener("click", () => showRole(roleIndex + Number(el.dataset.roleStep), { scroll: true })),
);
roleTabs.forEach((tab, i) =>
  tab.addEventListener("keydown", (event) => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[event.key];
    if (!step) return;
    event.preventDefault();
    showRole(i + step);
    roleTabs[roleIndex].focus();
  }),
);
document.querySelectorAll('.nav a[href^="#"]').forEach((link) =>
  link.addEventListener("click", (event) => {
    const i = roleIds.indexOf(link.getAttribute("href").slice(1));
    if (i < 0) return;
    event.preventDefault();
    showRole(i);
    rolesTop.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth" });
    history.replaceState(null, "", `#${roleIds[i]}`);
  }),
);
function followHash() {
  const i = roleIds.indexOf(location.hash.slice(1));
  if (i < 0) return;
  showRole(i, { animate: false });
  rolesTop.scrollIntoView();
}
addEventListener("hashchange", followHash);
showRole(0, { animate: false });
followHash();

/* Active chapter in navigation */
const navLinks = [...document.querySelectorAll('.nav nav a[href^="#"]')];
const spy = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navLinks.forEach((a) =>
        a.classList.toggle("active", a.getAttribute("href") === `#${entry.target.id}`),
      );
    });
  },
  { rootMargin: "-45% 0px -50% 0px" },
);
["actor", "judge", "teacher", "takeaways"].forEach((id) => spy.observe(document.getElementById(id)));

/* Segmented toggle helper */
function segmented(group, onChange) {
  const buttons = [...group.querySelectorAll("button")];
  buttons.forEach((button, i) =>
    button.addEventListener("click", () => {
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      onChange(i, button);
    }),
  );
}

/* Actor: goal masking */
const maskData = [
  { state: 14.6, place: 63.8, note: "With both subgoals in the goal, <b>63.8%</b> of the probability goes to placing the unheated mug." },
  { state: 77.6, place: 2.5, note: "Remove the easier subgoal and <b>77.6%</b> moves to the required state change: the model knew the step all along." },
];
const maskBar = document.querySelector("[data-mask-bar]");
const maskNote = document.querySelector("[data-mask-note]");
segmented(document.querySelector("[data-mask-toggle]"), (i) => {
  const d = maskData[i];
  const segs = maskBar.querySelectorAll(".seg");
  const other = +(100 - d.state - d.place).toFixed(1);
  [d.state, d.place, other].forEach((w, k) => {
    segs[k].style.setProperty("--w", w);
    segs[k].querySelector("span").textContent = k < 2 && w >= 8 ? `${w}%` : "";
  });
  maskNote.innerHTML = d.note;
});

/* Judge: context slider */
const contextNames = ["Prefix", "+ immediate feedback", "+ 25% of the rest", "+ 50% of the rest", "+ 75% of the rest", "Full trajectory"];
const contextNotes = [
  "<b>Prefix:</b> the history up to and including the action. JEV favors the premature placement (0.79), just like the Actor.",
  "<b>+ immediate feedback:</b> seeing the observation after each action reinforces the error: placement rises to 0.90, the microwave steps stay near 0.2.",
  "<b>+ 25%:</b> once the heating step and its feedback are visible, “open microwave” jumps from 0.18 to 0.66.",
  "<b>+ 50%:</b> the repair steps (take back, go to microwave) gain credit as their purpose becomes visible.",
  "<b>+ 75%:</b> the prerequisite steps now outscore the premature placement.",
  "<b>Full trajectory:</b> “open microwave” reaches 0.76, while the repeated look stays low (0.23) in every context.",
];
const range = document.querySelector("[data-context-range]");
const caption = document.querySelector("[data-context-caption]");
const ticks = document.querySelectorAll(".ticks li");
const steps = [...document.querySelectorAll("[data-steps] .step")].map((li) => ({
  bar: li.querySelector(".bar i"),
  out: li.querySelector("output"),
  scores: li.dataset.scores.split(",").map(Number),
}));
function setContext(k) {
  range.value = k;
  range.setAttribute("aria-valuetext", contextNames[k]);
  caption.innerHTML = contextNotes[k];
  ticks.forEach((t, i) => t.classList.toggle("on", i === k));
  steps.forEach(({ bar, out, scores }) => {
    bar.style.setProperty("--s", scores[k]);
    out.textContent = scores[k].toFixed(2);
    const delta = scores[k] - scores[0];
    out.classList.toggle("up", k > 0 && delta >= 0.1);
    out.classList.toggle("down", k > 0 && delta <= -0.1);
  });
}
range.addEventListener("input", () => {
  stopPlay();
  setContext(Number(range.value));
});
const playButton = document.querySelector("[data-context-play]");
let playTimer = null;
function stopPlay() {
  clearInterval(playTimer);
  playTimer = null;
  playButton.firstElementChild.textContent = "▶";
  playButton.setAttribute("aria-label", "Play context reveal");
}
playButton.addEventListener("click", () => {
  if (playTimer) return stopPlay();
  let k = Number(range.value) >= 5 ? 0 : Number(range.value);
  setContext(k);
  playButton.firstElementChild.textContent = "❚❚";
  playButton.setAttribute("aria-label", "Pause context reveal");
  playTimer = setInterval(() => {
    k += 1;
    setContext(k);
    if (k >= 5) stopPlay();
  }, 1500);
});
setContext(0);

/* Judge: question toggle */
const qNotes = [
  "Within-trajectory gap, milestone − no-op: <b>0.05</b>. Value scores track the outcome, so every action looks about the same.",
  "Within-trajectory gap, milestone − no-op: <b>0.36</b>. Progress asks about each action, so milestones stand out and no-ops sink.",
];
const qBars = [...document.querySelectorAll(".qbar")];
segmented(document.querySelector("[data-question-toggle]"), (i) => {
  qBars.forEach((row) => {
    const bar = row.querySelector("i");
    const s = Number(bar.dataset.q.split(",")[i]);
    bar.style.setProperty("--s", s);
    row.querySelector("output").textContent = s.toFixed(3);
  });
  document.querySelector("[data-gap]").innerHTML = qNotes[i];
});

/* Teacher: gain baseline toggle */
const mirrorNotes = [
  "Against the best of the other three Context × Question settings, Hindsight + Progress wins on every task JEV fails as an Actor, and loses on the two it handles well.",
  "Against outcome-only GRPO, Hindsight + Progress improves every task type; the largest gain is on Cool (+23.0), a task JEV completes only 16.0% of the time as an Actor.",
];
const mirrorRows = [...document.querySelectorAll(".mirror-row")];
segmented(document.querySelector("[data-mirror-toggle]"), (i) => {
  mirrorRows.forEach((row) => {
    const g = Number(row.dataset.gain.split(",")[i]);
    const cell = row.querySelector(".m-gain");
    cell.classList.toggle("neg", g < 0);
    cell.querySelector("i").style.setProperty("--g", g);
    cell.querySelector("b").textContent = (g < 0 ? "−" : "+") + Math.abs(g).toFixed(1);
  });
  document.querySelector("[data-mirror-note]").textContent = mirrorNotes[i];
});

/* Takeaway cards: pointer glow */
document.querySelectorAll(".tk").forEach((card) =>
  card.addEventListener("pointermove", (event) => {
    const rect = card.getBoundingClientRect();
    card.style.setProperty("--mx", `${event.clientX - rect.left}px`);
    card.style.setProperty("--my", `${event.clientY - rect.top}px`);
  }),
);

/* Overview figure: one decision model, queried three ways */
(() => {
  const fig = document.querySelector("[data-ov]");
  if (!fig) return;
  const stage = fig.querySelector("[data-ov-stage]");
  const svg = fig.querySelector("[data-ov-wires]");
  const core = fig.querySelector("[data-ov-core]");
  const slots = Object.fromEntries([...fig.querySelectorAll("[data-slot]")].map((el) => [el.dataset.slot, el]));
  const tabs = [...fig.querySelectorAll("[data-ov-role]")];
  const playButton = fig.querySelector("[data-ov-play]");
  const roles = ["actor", "judge", "teacher"];
  const cards = Object.fromEntries(roles.map((r) => [r, [...fig.querySelectorAll(`[data-ov-card="${r}"]`)]]));
  const anchor = (key) => fig.querySelector(`[data-ov-a="${key}"]`);
  const judgeIn = cards.judge[0];
  const cells = [...fig.querySelectorAll("[data-cell]")];
  const scores = [...cards.teacher[1].querySelectorAll("output[data-v]")];
  const NS = "http://www.w3.org/2000/svg";

  fig.querySelectorAll(".kids").forEach((list) =>
    [...list.children].forEach((child, i) => child.style.setProperty("--i", i)),
  );

  /* ---------- Wires: an SVG overlay routed between live element boxes ---------- */
  let wires = {};
  const shown = (el) => el.getClientRects().length > 0;
  function box(el) {
    const r = el.getBoundingClientRect();
    const s = stage.getBoundingClientRect();
    const l = r.left - s.left;
    const t = r.top - s.top;
    return { l, t, r: l + r.width, b: t + r.height, cx: l + r.width / 2, cy: t + r.height / 2 };
  }
  function routes() {
    const out = {};
    const c = box(core);
    const gut = parseFloat(getComputedStyle(fig).getPropertyValue("--gut")) || 30;
    roles.forEach((role) => {
      const [input, output] = cards[role];
      if (!shown(input)) return;
      const i = box(input);
      const o = box(output);
      out[`${role}-in`] = [role, [[i.cx, i.b], [i.cx, c.t]]];
      out[`${role}-out`] = [role, [[o.cx, c.b], [o.cx, o.t]]];
    });
    if (shown(cards.actor[0])) {
      const obs = box(anchor("obs"));
      const res = box(anchor("actor-res"));
      const x = box(cards.actor[0]).l - gut * 0.55;
      out["actor-loop"] = ["actor", [[res.l, res.cy], [x, res.cy], [x, obs.cy], [obs.l, obs.cy]]];
    }
    if (shown(cards.teacher[0])) {
      const right = box(cards.teacher[0]).r;
      const roll = box(anchor("rollouts"));
      const merge = box(anchor("merge"));
      const grpo = box(anchor("grpo"));
      const policy = box(anchor("policy"));
      const x1 = right + gut * 0.3;
      const x2 = right + gut * 0.62;
      out["teacher-adv"] = ["teacher", [[roll.r, roll.cy], [x1, roll.cy], [x1, merge.cy], [merge.r, merge.cy]]];
      out["teacher-loop"] = ["teacher", [[grpo.r, grpo.cy], [x2, grpo.cy], [x2, policy.cy], [policy.r, policy.cy]]];
    }
    return out;
  }
  function pathData(pts, radius = 12) {
    let d = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length - 1; i++) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      const [x2, y2] = pts[i + 1];
      const l1 = Math.hypot(x1 - x0, y1 - y0) || 1;
      const l2 = Math.hypot(x2 - x1, y2 - y1) || 1;
      const k = Math.min(radius, l1 / 2, l2 / 2);
      d += ` L${x1 + ((x0 - x1) / l1) * k},${y1 + ((y0 - y1) / l1) * k}`;
      d += ` Q${x1},${y1} ${x1 + ((x2 - x1) / l2) * k},${y1 + ((y2 - y1) / l2) * k}`;
    }
    const [lx, ly] = pts[pts.length - 1];
    return `${d} L${lx},${ly}`;
  }
  const make = (tag, attrs) => {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
  };
  function drawWires() {
    const keep = Object.fromEntries(Object.entries(wires).map(([id, w]) => [id, w.g.getAttribute("class")]));
    svg.replaceChildren();
    const defs = make("defs", {});
    roles.forEach((role) => {
      const marker = make("marker", {
        id: `ov-arrow-${role}`,
        markerWidth: 9,
        markerHeight: 9,
        refX: 8,
        refY: 4.5,
        orient: "auto",
        markerUnits: "userSpaceOnUse",
      });
      marker.append(make("path", { d: "M0,0.5 L9,4.5 L0,8.5 z", style: `fill: var(--${role})` }));
      defs.append(marker);
    });
    svg.append(defs);
    wires = {};
    Object.entries(routes()).forEach(([id, [role, pts]]) => {
      const d = pathData(pts);
      const g = make("g", { style: `--wc: var(--${role})`, "data-role": role });
      const base = make("path", { class: "base", d, "marker-end": `url(#ov-arrow-${role})` });
      const flow = make("path", { class: "flow", d });
      const dot = make("circle", { class: "dot", r: 4.5, cx: pts[0][0], cy: pts[0][1] });
      g.append(base, flow, dot);
      svg.append(g);
      const len = flow.getTotalLength();
      flow.style.strokeDasharray = `${len} ${len}`;
      flow.style.strokeDashoffset = keep[id]?.includes("done") ? 0 : len;
      if (keep[id]) g.setAttribute("class", keep[id]);
      wires[id] = { g, flow, dot, len, role };
    });
    markWires();
  }
  function markWires() {
    Object.values(wires).forEach((w) => w.g.classList.toggle("on", w.role === roles[roleIndex]));
  }
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  function flow(id, ms) {
    const run = token;
    const start = performance.now();
    return new Promise((resolve) => {
      function frame(now) {
        const w = wires[id];
        if (run !== token || !w) return resolve();
        const t = Math.min(1, (now - start) / ms);
        const at = w.len * ease(t);
        const p = w.flow.getPointAtLength(at);
        w.flow.style.strokeDashoffset = w.len - at;
        w.dot.setAttribute("cx", p.x);
        w.dot.setAttribute("cy", p.y);
        w.g.classList.add("live");
        if (t < 1) return requestAnimationFrame(frame);
        w.g.classList.remove("live");
        w.g.classList.add("done");
        resolve();
      }
      requestAnimationFrame(frame);
    });
  }
  function clearWires(role) {
    Object.values(wires).forEach((w) => {
      if (role && w.role !== role) return;
      w.g.classList.remove("live", "done");
      w.flow.style.strokeDashoffset = w.len;
    });
  }

  /* ---------- Small effects ---------- */
  const lit = (role, n) =>
    cards[role].forEach((card) => card.querySelectorAll(`[data-step="${n}"]`).forEach((el) => el.classList.add("lit")));
  function bump(el) {
    el.classList.remove("bump");
    void el.offsetWidth;
    el.classList.add("bump");
  }
  const slotText = {
    actor: ["What should the next action be?", "goal + history + observation", "probability per action"],
    teacher: ["Did this action help?", "Hindsight: the whole rollout", "a score for every action"],
  };
  function setSlots([q, c, o]) {
    [
      [slots.q, q],
      [slots.c, c],
      [slots.o, o],
    ].forEach(([el, text]) => {
      el.classList.add("set");
      if (el.textContent === text) return;
      el.textContent = text;
      el.classList.remove("swap");
      void el.offsetWidth;
      el.classList.add("swap");
    });
  }
  function think(role, text) {
    core.style.setProperty("--rc", `var(--${role})`);
    setSlots(text || slotText[role]);
    core.classList.remove("busy");
    void core.offsetWidth;
    core.classList.add("busy");
  }
  const combos = [
    ["prefix", "value"],
    ["prefix", "progress"],
    ["hindsight", "value"],
    ["hindsight", "progress"],
  ];
  function judgeText([ctx, q]) {
    return [
      q === "value" ? "Will the task succeed?" : "Did step 3 help?",
      ctx === "prefix" ? "Prefix: steps 1–3" : "Hindsight: steps 1–5",
      "score in [0, 1]",
    ];
  }
  function setContext([ctx, q]) {
    judgeIn.dataset.ctx = ctx;
    judgeIn.dataset.q = q;
  }
  function hit(k) {
    cells.forEach((cell, i) => cell.classList.toggle("cur", i === k));
    cells[k].classList.add("hit");
  }
  function countScores() {
    const run = token;
    scores.forEach((el, i) => {
      const target = Number(el.dataset.v);
      const start = performance.now() + i * 70;
      function frame(now) {
        if (run !== token) return;
        const t = Math.max(0, Math.min(1, (now - start) / 700));
        el.textContent = (target * ease(t)).toFixed(2);
        if (t < 1) requestAnimationFrame(frame);
      }
      el.textContent = "0.00";
      requestAnimationFrame(frame);
    });
  }

  /* ---------- Sequences: [duration ms, action] ---------- */
  const idle = () => {};
  const sequences = {
    actor: [
      [500, () => lit("actor", 1)],
      [700, () => lit("actor", 2)],
      [600, () => lit("actor", 3)],
      [800, (ms) => flow("actor-in", ms)],
      [900, () => think("actor")],
      [700, (ms) => flow("actor-out", ms)],
      [1300, () => lit("actor", 4)],
      [900, () => lit("actor", 5)],
      [1000, () => lit("actor", 6)],
      [800, () => lit("actor", 7)],
      [1300, (ms) => flow("actor-loop", ms)],
      [1800, () => bump(anchor("obs"))],
    ],
    judge: [
      [900, () => {
        setContext(combos[0]);
        lit("judge", 1);
      }],
      ...combos.flatMap((cq, k) => [
        [450, () => setContext(cq)],
        [550, (ms) => flow("judge-in", ms)],
        [450, () => think("judge", judgeText(cq))],
        [500, (ms) => flow("judge-out", ms)],
        [k === 3 ? 900 : 650, () => hit(k)],
      ]),
      [2000, () => lit("judge", 2)],
    ],
    teacher: [
      [600, () => bump(anchor("policy"))],
      [1200, () => lit("teacher", 2)],
      [800, (ms) => flow("teacher-in", ms)],
      [900, () => think("teacher")],
      [700, (ms) => flow("teacher-out", ms)],
      [1400, () => {
        lit("teacher", 3);
        countScores();
      }],
      [900, (ms) => flow("teacher-adv", ms)],
      [700, () => lit("teacher", 4)],
      [800, () => lit("teacher", 5)],
      [1300, (ms) => flow("teacher-loop", ms)],
      [600, () => bump(anchor("policy"))],
      [1900, () => lit("teacher", 6)],
    ],
  };

  /* ---------- Player ---------- */
  let token = 0;
  let roleIndex = 0;
  let playing = !reduceMotion;
  let inView = false;
  let running = false;
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  function setActive(index, dim) {
    roleIndex = index;
    const role = roles[index];
    fig.dataset.active = role;
    fig.toggleAttribute("data-dim", dim);
    roles.forEach((r) => cards[r].forEach((card) => card.classList.toggle("on", r === role)));
    tabs.forEach((tab) => tab.setAttribute("aria-pressed", String(tab.dataset.ovRole === role)));
    markWires();
  }
  function reset(role) {
    cards[role].forEach((card) => {
      card.classList.add("ov-run");
      card.querySelectorAll(".lit").forEach((el) => el.classList.remove("lit"));
    });
    if (role === "judge") cells.forEach((cell) => cell.classList.remove("hit", "cur"));
    if (role === "teacher") scores.forEach((el) => (el.textContent = el.dataset.v));
    clearWires(role);
  }
  function settle(role) {
    cards[role].forEach((card) => card.classList.remove("ov-run"));
    cells.forEach((cell) => cell.classList.remove("cur"));
    if (role === "judge") setContext(combos[3]);
    if (role === "teacher") scores.forEach((el) => (el.textContent = el.dataset.v));
    core.classList.remove("busy");
    clearWires();
    tabs.forEach((tab) => tab.style.setProperty("--t", 0));
  }
  function stop() {
    token++;
    running = false;
    settle(roles[roleIndex]);
  }
  async function run(index) {
    const my = ++token;
    running = true;
    const prev = roles[roleIndex];
    settle(prev);
    setActive(index, true);
    const role = roles[index];
    // Layout may change (one card per role on narrow screens): reroute before animating.
    drawWires();
    reset(role);
    const steps = sequences[role];
    const total = steps.reduce((sum, [ms]) => sum + ms, 0);
    const tab = tabs[index];
    const start = performance.now();
    (function progressFrame(now) {
      if (my !== token) return;
      tab.style.setProperty("--t", Math.min(1, (now - start) / total));
      requestAnimationFrame(progressFrame);
    })(start);
    for (const [ms, action] of steps) {
      action(ms);
      await wait(ms);
      if (my !== token) return;
    }
    running = false;
    settle(role);
    if (playing && inView) run((index + 1) % roles.length);
  }
  function select(index) {
    stop();
    if (playing && inView) run(index);
    else {
      setActive(index, true);
      drawWires();
    }
  }
  function setPlaying(next) {
    playing = next;
    fig.toggleAttribute("data-paused", !next);
    playButton.setAttribute("aria-label", next ? "Pause animation" : "Play animation");
    if (next) run(roleIndex);
    else stop();
  }

  tabs.forEach((tab, i) => tab.addEventListener("click", () => select(i)));
  roles.forEach((role, i) =>
    cards[role].forEach((card) =>
      card.addEventListener("click", () => {
        if (roleIndex !== i || !running) select(i);
      }),
    ),
  );
  playButton.addEventListener("click", () => setPlaying(!playing));

  new IntersectionObserver(
    ([entry]) => {
      inView = entry.isIntersecting;
      if (inView && playing && !running) run(roleIndex);
      if (!inView && running) stop();
    },
    { threshold: 0.3 },
  ).observe(stage);

  let queued = false;
  new ResizeObserver(() => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      drawWires();
    });
  }).observe(stage);
  document.fonts?.ready.then(drawWires);

  fig.toggleAttribute("data-paused", !playing);
  setActive(0, false);
  drawWires();
})();
