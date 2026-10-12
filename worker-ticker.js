/**
 * worker-ticker.js — PostureGuard High-Precision Background Ticker
 * 
 * Runs inside a dedicated Web Worker thread.
 * Web Worker timers are immune to browser inactive/hidden tab throttling,
 * allowing reliable 3–4 FPS background monitoring and prompt alerts.
 */

let timerId = null;

self.onmessage = function (e) {
  const data = e.data || {};
  const action = typeof data === 'string' ? data : data.action;

  if (action === 'start') {
    const interval = data.interval || 250; // Default 250ms (~4 FPS)
    if (timerId) {
      clearInterval(timerId);
    }
    timerId = setInterval(() => {
      self.postMessage({ type: 'tick', timestamp: Date.now() });
    }, interval);
  } else if (action === 'stop') {
    if (timerId) {
      clearInterval(timerId);
      timerId = null;
    }
  }
};
