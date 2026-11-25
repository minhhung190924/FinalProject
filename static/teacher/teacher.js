function toggleMainMenu(button) {
  const overlay = document.getElementById("mainMenuOverlay");
  if (!overlay) return;
  const willOpen = overlay.hidden;
  overlay.hidden = !willOpen;
  button.setAttribute("aria-expanded", willOpen ? "true" : "false");
  document.body.style.overflow = willOpen ? "hidden" : "";
}
function closeMainMenu() {
  const overlay = document.getElementById("mainMenuOverlay");
  if (overlay) overlay.hidden = true;
  document.querySelectorAll(".menu-toggle").forEach(function (button) {
    button.setAttribute("aria-expanded", "false");
  });
  document.body.style.overflow = "";
}
document.addEventListener("click", function (event) {
  const overlay = document.getElementById("mainMenuOverlay");
  if (overlay && event.target === overlay) closeMainMenu();
});
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") closeMainMenu();
});
