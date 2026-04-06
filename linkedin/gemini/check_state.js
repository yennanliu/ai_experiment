(function() {
  return window.automationState || { isPaused: false, shouldQuit: false, applied: 0, skipped: 0, failed: 0 };
})();
