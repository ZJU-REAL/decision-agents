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
