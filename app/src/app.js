"use strict";
const $ = (s) => document.querySelector(s);
let DATA = null;

function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  kids.forEach((c) => e.append(c));
  return e;
}

function fmt(v) { return Number.isInteger(v) ? String(v) : Number(v).toPrecision(4).replace(/\.?0+$/, ""); }

function render() {
  const q = $("#q").value.trim().toLowerCase();
  const fam = $("#fam").value;
  const rows = DATA.items.filter((i) =>
    (fam === "All" || i.family === fam) &&
    (!q || (i.id + " " + (i.algorithm || "") + " " + (i.dataset || "")).toLowerCase().includes(q)));
  const body = $("#tbl tbody");
  body.replaceChildren();
  rows.forEach((i) => {
    const key = i.headline ? `${i.headline.name} = ${fmt(i.headline.value)}` : "none saved";
    const btn = el("button", { type: "button", class: "rowbtn", text: "Details", "aria-label": "Details for " + i.title });
    btn.addEventListener("click", () => show(i));
    body.append(el("tr", {}, el("td", { text: i.title }), el("td", { text: i.family }), el("td", { text: i.dataset || "" }),
      el("td", { class: "num", text: key }), el("td", {}, btn)));
  });
  $("#count").textContent = rows.length + " of " + DATA.items.length + " notebooks";
}

function show(i) {
  $("#detail").hidden = false;
  $("#dh").textContent = i.title;
  $("#dmeta").textContent = [i.family, i.dataset, i.task, i.seed != null ? "seed " + i.seed : "", i.created ? "run " + i.created : "", i.path].filter(Boolean).join(" | ");
  const mt = $("#mt tbody");
  mt.replaceChildren();
  const entries = Object.entries(i.metrics || {});
  if (!entries.length) mt.append(el("tr", {}, el("td", { text: "No metrics saved. This notebook is a concept demo." })));
  entries.forEach(([k, v]) => mt.append(el("tr", {}, el("td", { text: k }), el("td", { class: "num", text: fmt(v) }))));
  const figs = $("#figs");
  figs.replaceChildren();
  (i.figures || []).forEach((f) => figs.append(el("img", { src: f, alt: "Figure " + f.split("/").pop().replace(".png", "").replace(/_/g, " "), loading: "lazy" })));
  if (!(i.figures || []).length) figs.append(el("p", { class: "muted", text: "No figures saved for this notebook." }));
  const a = $("#prev");
  if (i.preview) { a.href = i.preview; a.hidden = false; } else a.hidden = true;
  $("#detail").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  $("#dh").setAttribute("tabindex", "-1"); $("#dh").focus({ preventScroll: true });
}

fetch("data.json").then((r) => r.json()).then((d) => {
  DATA = d;
  const fams = ["All", ...new Set(d.items.map((i) => i.family))];
  $("#fam").append(...fams.map((f) => el("option", { value: f, text: f })));
  $("#q").addEventListener("input", render);
  $("#fam").addEventListener("change", render);
  $("#close").addEventListener("click", () => { $("#detail").hidden = true; });
  const st = $("#st tbody");
  d.status.forEach((s) => st.append(el("tr", {}, el("td", { text: s.area }), el("td", { text: s.target }), el("td", { text: s.done }),
    el("td", {}, el("span", { class: "tag " + s.status.split(" ")[0], text: s.status })))));
  render();
}).catch(() => { $("#count").textContent = "Could not load data.json. Serve this folder over HTTP."; });
