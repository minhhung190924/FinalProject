var cur = 0;
function lbShow(i) {
  if (!EV.length) return;
  cur = (i + EV.length) % EV.length;
  var e = EV[cur];
  document.getElementById("lbImg").src = "/evidence/" + e.id;
  document.getElementById("lbCap").textContent = (cur + 1) + "/" + EV.length + " - " + e.time + " - " +
    (e.severity === "warning" ? "Cảnh báo" : "Vi phạm") + " (" + e.type + "): " + e.message;
  document.getElementById("lb").hidden = false;
}
function lbMove(d) { lbShow(cur + d); }
function lbClose() { document.getElementById("lb").hidden = true; }
document.addEventListener("keydown", function (ev) {
  if (document.getElementById("lb").hidden) return;
  if (ev.key === "Escape") lbClose();
  else if (ev.key === "ArrowLeft") lbMove(-1);
  else if (ev.key === "ArrowRight") lbMove(1);
});
