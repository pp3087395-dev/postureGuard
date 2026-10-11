/**
 * ============================================================================
 * PostureGuard — Real-Time Ergonomic AI Coach
 * College ML Competition: BUILD WITH ML, University of Lucknow
 * 
 * Tech Stack: Vanilla JavaScript, TensorFlow.js, Google Teachable Machine Pose
 * Architecture: 100% Client-Side WebGL Inference (Zero Video Uploads)
 * ============================================================================
 */

/**
 * Universal Posture Evaluator Helper
 * Matches any class label containing "good" (case-insensitive, trimmed).
 */
function isGoodPosture(label) {
  if (!label || typeof label !== 'string') return false;
  return label.trim().toLowerCase().includes('good');
}

// Background tab inference interval (~4.5 FPS)
const BG_INTERVAL_MS = 220;

// --- Global Application State ---
const state = {
  // Model & Vision
  model: null,
  webcam: null,
  classLabels: [],          // Dynamically loaded from metadata.json
  isModelLoaded: false,
  isMonitoring: false,
  isPaused: false,
  animationFrameId: null,
  backgroundIntervalId: null,
  isInBackground: false,
  mirrorFeed: true,

  // Prediction Smoothing Buffer (10-frame sliding window)
  smoothingWindowSize: 10,
  predictionHistory: [],    // Array of { [className]: probability }
  smoothedProbabilities: {},
  currentActivePosture: 'Detecting...',
  detectionThreshold: 0.80, // Minimum 80% confidence required

  // Session Telemetry & Statistics
  sessionStartTime: null,
  activeSessionDuration: 0, // In seconds
  sessionTimerInterval: null,
  postureDurations: {},     // { [className]: seconds }
  totalAlertsTriggered: 0,
  timelineHistory: [],      // Array of last 60 seconds posture colors

  // Smart Alert Engine
  alertDelaySeconds: 25,
  badPostureStartTime: null,
  isAlarmActive: false,
  alarmIntervalId: null,
  activeNotification: null,
  isAlertBannerShowing: false,
  isMuted: false,
  audioContext: null,
  notificationsEnabled: false,

  // Document Picture-in-Picture Floating Window
  pipWindow: null,

  // Simulation Mode (for demo/eval when weights are pending)
  isSimulatedMode: false,
  simulatedInterval: null,
  simulatedStep: 0,

  // FPS Counter
  lastFrameTime: performance.now(),
  frameCount: 0,
  currentFps: 0
};

/**
 * Dynamic metadata and coaching suggestions by detected posture
 * Uses isGoodPosture() rather than rigid strings.
 */
function getPostureMeta(label) {
  if (isGoodPosture(label)) {
    return {
      title: 'Optimal Alignment',
      tip: 'Great job! Your cervical spine is neutral and shoulders are relaxed. Maintain steady breathing.',
      className: 'is-good',
      badgeClass: 'badge-good'
    };
  }

  const lower = (label || '').toLowerCase();
  if (lower.includes('slouch')) {
    return {
      title: 'Lumbar Slouch Detected',
      tip: 'Roll your shoulders backwards, engage your core muscles, and sit firmly against your backrest.',
      className: 'is-slouch',
      badgeClass: 'badge-slouch'
    };
  }
  if (lower.includes('neck')) {
    return {
      title: 'Forward Head Strain ("Tech Neck")',
      tip: 'Pull your chin gently backwards (cervical retraction), align ears over shoulders, and lift chest.',
      className: 'is-neck',
      badgeClass: 'badge-neck'
    };
  }
  if (lower.includes('lean')) {
    return {
      title: 'Lateral Spine Asymmetry',
      tip: 'Distribute your body weight evenly across both sit-bones and bring your monitor directly ahead.',
      className: 'is-lean',
      badgeClass: 'badge-lean'
    };
  }
  if (lower.includes('no person')) {
    return {
      title: 'No Subject In View',
      tip: 'Please position yourself in front of the webcam to begin posture tracking.',
      className: '',
      badgeClass: 'badge-neutral'
    };
  }
  if (lower.includes('detect')) {
    return {
      title: 'Calibrating Pose...',
      tip: 'Ensure your upper torso, shoulders, and head are clearly visible within the camera frame.',
      className: '',
      badgeClass: 'badge-neutral'
    };
  }

  // Any other customized bad posture label
  return {
    title: `${label} Detected`,
    tip: 'Adjust your seating posture and maintain a neutral upright spine.',
    className: 'is-slouch',
    badgeClass: 'badge-slouch'
  };
}

/**
 * Dynamic color provider for badges, meters, and canvas timeline
 */
function getPostureColor(label) {
  if (isGoodPosture(label)) return '#10b981'; // Emerald Green
  const lower = (label || '').toLowerCase();
  if (lower.includes('slouch')) return '#f59e0b'; // Amber
  if (lower.includes('neck')) return '#f43f5e';   // Rose
  if (lower.includes('lean')) return '#a855f7';   // Violet
  if (lower.includes('no person')) return '#475569';
  if (lower.includes('detect')) return '#64748b'; // Slate
  return '#f59e0b'; // Default bad posture color
}

// --- DOM Element References ---
const DOM = {
  // Canvases
  canvas: document.getElementById('outputCanvas'),
  ctx: document.getElementById('outputCanvas').getContext('2d'),
  timelineCanvas: document.getElementById('timelineCanvas'),
  timelineCtx: document.getElementById('timelineCanvas').getContext('2d'),

  // Overlays & Placeholders
  videoPlaceholder: document.getElementById('videoPlaceholder'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  loadingTitle: document.getElementById('loadingTitle'),
  loadingSubtitle: document.getElementById('loadingSubtitle'),
  errorOverlay: document.getElementById('errorOverlay'),
  errorTitle: document.getElementById('errorTitle'),
  errorMessage: document.getElementById('errorMessage'),
  canvasOverlayBadge: document.getElementById('canvasPostureBadge'),
  badPostureTimerPill: document.getElementById('badPostureTimerPill'),
  timerPillText: document.getElementById('timerPillText'),
  alertBanner: document.getElementById('alertBanner'),
  alertBannerTitle: document.getElementById('alertBannerTitle'),
  alertBannerMessage: document.getElementById('alertBannerMessage'),
  dismissAlertBtn: document.getElementById('dismissAlertBtn'),

  // Controls
  btnToggleMonitoring: document.getElementById('btnToggleMonitoring'),
  btnPlayPauseIcon: document.getElementById('btnPlayPauseIcon'),
  btnPlayPauseText: document.getElementById('btnPlayPauseText'),
  btnStopSession: document.getElementById('btnStopSession'),
  btnResetSession: document.getElementById('btnResetSession'),
  startFromPlaceholderBtn: document.getElementById('startFromPlaceholderBtn'),
  startPostureCheckBtn: document.getElementById('startPostureCheckBtn'),
  retryInitBtn: document.getElementById('retryInitBtn'),
  demoSimulateBtn: document.getElementById('demoSimulateBtn'),
  mirrorToggle: document.getElementById('mirrorToggle'),

  // Document PiP Controls
  btnFloatingPip: document.getElementById('btnFloatingPip'),
  btnFloatingPipText: document.getElementById('btnFloatingPipText'),
  pipUnsupportedNote: document.getElementById('pipUnsupportedNote'),

  // Analysis Cards
  statusBannerBox: document.getElementById('statusBannerBox'),
  currentPostureLabel: document.getElementById('currentPostureLabel'),
  currentPostureSub: document.getElementById('currentPostureSub'),
  currentConfidence: document.getElementById('currentConfidence'),
  postureTipText: document.getElementById('postureTipText'),
  classProbabilitiesContainer: document.getElementById('classProbabilitiesContainer'),

  // Settings
  alertDelaySelect: document.getElementById('alertDelaySelect'),
  toggleAudioBtn: document.getElementById('toggleAudioBtn'),
  audioBtnText: document.getElementById('audioBtnText'),
  testAudioBtn: document.getElementById('testAudioBtn'),
  requestNotificationBtn: document.getElementById('requestNotificationBtn'),
  notificationBtnText: document.getElementById('notificationBtnText'),
  navMuteBtn: document.getElementById('navMuteBtn'),
  navSoundIcon: document.getElementById('navSoundIcon'),

  // Header & Status Chips
  sessionStatusPill: document.getElementById('sessionStatusPill'),
  headerTimer: document.getElementById('headerTimer'),
  headerGoodPercent: document.getElementById('headerGoodPercent'),
  camStatusDot: document.getElementById('camStatusDot'),
  fpsDisplay: document.getElementById('fpsDisplay'),
  debugLine: document.getElementById('debugLine'),

  // Dashboard Metrics
  statSessionTime: document.getElementById('statSessionTime'),
  statGoodRatio: document.getElementById('statGoodRatio'),
  statGoodTime: document.getElementById('statGoodTime'),
  statAlertCount: document.getElementById('statAlertCount'),
  statPrimaryConcern: document.getElementById('statPrimaryConcern'),
  statConcernDetail: document.getElementById('statConcernDetail'),

  // Distribution Segments & Legends
  distributionBar: document.getElementById('distributionBar'),

  // Summary Modal Dialog
  summaryModal: document.getElementById('summaryModal'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalDismissBtn: document.getElementById('modalDismissBtn'),
  modalNewSessionBtn: document.getElementById('modalNewSessionBtn'),
  modalScoreValue: document.getElementById('modalScoreValue'),
  modalScoreCircle: document.getElementById('modalScoreCircle'),
  modalScoreGrade: document.getElementById('modalScoreGrade'),
  modalScoreSummaryText: document.getElementById('modalScoreSummaryText'),
  modalStatDuration: document.getElementById('modalStatDuration'),
  modalStatGoodTime: document.getElementById('modalStatGoodTime'),
  modalStatAlerts: document.getElementById('modalStatAlerts'),
  modalStatThreshold: document.getElementById('modalStatThreshold'),
  modalBreakdownList: document.getElementById('modalBreakdownList'),
  modalAdviceText: document.getElementById('modalAdviceText')
};

// ============================================================================
// 1. Model Loading & Vision Initialization
// ============================================================================

/**
 * Loads the Teachable Machine Pose model dynamically from:
 * ./model/model.json & ./model/metadata.json
 * Class names are extracted dynamically from metadata.json without hardcoding.
 */
async function loadModel() {
  const modelURL = './model/model.json';
  const metadataURL = './model/metadata.json';

  showLoading('Loading Teachable Machine Pose Model...', 'Downloading model topology and PoseNet weights');

  try {
    // 1. Read class names dynamically from metadata.json
    try {
      const metaResp = await fetch(metadataURL);
      if (metaResp.ok) {
        const metaJson = await metaResp.json();
        if (Array.isArray(metaJson.labels) && metaJson.labels.length > 0) {
          state.classLabels = metaJson.labels;
          console.log('[PostureGuard] Dynamically loaded labels from metadata.json:', state.classLabels);
        }
      }
    } catch (metaErr) {
      console.warn('[PostureGuard] Direct metadata.json fetch fallback:', metaErr);
    }

    // 2. Load model using tmPose
    state.model = await tmPose.load(modelURL, metadataURL);

    // If metadata.json fetch was not populated, read from model.getClassLabels()
    if (!state.classLabels || state.classLabels.length === 0) {
      state.classLabels = state.model.getClassLabels();
    }
    console.log('[PostureGuard] Model successfully loaded. Dynamic classes:', state.classLabels);

    // Initialize telemetry tracking slots for all dynamically discovered classes
    state.classLabels.forEach(label => {
      state.postureDurations[label] = 0;
      state.smoothedProbabilities[label] = 0;
    });

    // Render dynamic UI meters and distribution bar for every class in the model
    renderClassMetersUI();
    renderDistributionUI();

    state.isModelLoaded = true;
    hideLoading();
    return true;
  } catch (err) {
    console.warn('[PostureGuard] Could not load model from ./model:', err);
    hideLoading();
    showError(
      'Model Files Not Found in /model',
      `Could not locate <code>model.json</code> and <code>metadata.json</code> inside the <code>./model/</code> folder.
       Please export your model from Google Teachable Machine as TensorFlow.js and place the 3 files (<code>model.json</code>, <code>metadata.json</code>, <code>weights.bin</code>) in the <code>/model</code> folder.`
    );
    return false;
  }
}

/**
 * Initializes the user's webcam using tmPose.Webcam
 */
async function startWebcam() {
  showLoading('Starting Camera...', 'Requesting webcam permissions');

  const width = 400;
  const height = 400;
  const flip = state.mirrorFeed; // Mirror horizontally for intuitive user preview

  try {
    state.webcam = new tmPose.Webcam(width, height, flip);
    await state.webcam.setup(); // Request camera access
    await state.webcam.play();

    DOM.videoPlaceholder.style.display = 'none';
    DOM.camStatusDot.classList.add('active');
    hideLoading();
    return true;
  } catch (cameraErr) {
    console.error('[PostureGuard] Camera access error:', cameraErr);
    hideLoading();
    showError(
      'Camera Permission Denied or Device Unavailable',
      'PostureGuard requires webcam access to analyze your posture locally. Please check your browser permissions, allow camera access, and ensure no other application is using your webcam.'
    );
    return false;
  }
}

// ============================================================================
// 2. Real-Time Inference & Pose Estimation Loop (Foreground & Background)
// ============================================================================

/**
 * Executes a single inference step.
 * When isBackground is false: draws frame and skeleton to canvas.
 * When isBackground is true (tab hidden): skips canvas drawing to save GPU/battery
 * but continues PoseNet estimation, smoothing, bad posture alerts & PiP updates.
 */
async function stepInference(isBackground = false) {
  if (!state.isMonitoring || state.isPaused || !state.model || !state.webcam) return;

  try {
    // 1. Update webcam frame
    state.webcam.update();

    // 2. Pose estimation & classification
    const { pose, posenetOutput } = await state.model.estimatePose(state.webcam.canvas);
    const predictions = await state.model.predict(posenetOutput);

    // 3. Render to canvas only when visible
    if (!isBackground) {
      DOM.ctx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
      DOM.ctx.drawImage(state.webcam.canvas, 0, 0, DOM.canvas.width, DOM.canvas.height);

      if (pose) {
        const minPartConfidence = 0.5;
        if (typeof tmPose.drawKeypoints === 'function') {
          tmPose.drawKeypoints(pose.keypoints, minPartConfidence, DOM.ctx);
        }
        if (typeof tmPose.drawSkeleton === 'function') {
          tmPose.drawSkeleton(pose.keypoints, minPartConfidence, DOM.ctx);
        }
      }
    }

    // 4. Process predictions with 10-frame smoothing buffer
    processPredictions(predictions, Boolean(pose));

    // 5. Update Floating Mini Window (Document PiP) if open
    if (state.pipWindow) {
      updatePipUI();
    }

    // 6. Update FPS indicator
    updateFPS();
  } catch (err) {
    console.error('[PostureGuard] Inference step error:', err);
  }
}

/**
 * Manages dual-mode execution loop:
 * - When document.hidden is false: uses requestAnimationFrame for full 30 FPS visual smoothness.
 * - When document.hidden is true: switches to setInterval at ~4-5 FPS (220ms) so background monitoring
 *   never pauses even when the user switches tabs or windows.
 */
function startInferenceLoop() {
  stopInferenceLoop();
  if (!state.isMonitoring || state.isPaused) return;

  if (document.hidden) {
    // Tab is hidden: run background loop at ~4-5 FPS
    state.isInBackground = true;
    console.log('[PostureGuard] Tab hidden: running background monitoring at ~4-5 FPS');
    state.backgroundIntervalId = setInterval(async () => {
      if (state.isMonitoring && !state.isPaused) {
        await stepInference(true);
      }
    }, BG_INTERVAL_MS);
  } else {
    // Tab is visible: run requestAnimationFrame loop
    state.isInBackground = false;
    const loop = async () => {
      if (!state.isMonitoring || state.isPaused || document.hidden) return;
      await stepInference(false);
      if (!document.hidden && state.isMonitoring && !state.isPaused) {
        state.animationFrameId = requestAnimationFrame(loop);
      }
    };
    state.animationFrameId = requestAnimationFrame(loop);
  }
}

/**
 * Stops both requestAnimationFrame and setInterval loops
 */
function stopInferenceLoop() {
  if (state.animationFrameId) {
    cancelAnimationFrame(state.animationFrameId);
    state.animationFrameId = null;
  }
  if (state.backgroundIntervalId) {
    clearInterval(state.backgroundIntervalId);
    state.backgroundIntervalId = null;
  }
}

/**
 * Handles tab visibility changes to transition seamlessly between loops
 */
function handleVisibilityChange() {
  if (!state.isMonitoring || state.isPaused) return;
  console.log('[PostureGuard] Visibility changed: document.hidden =', document.hidden);
  startInferenceLoop();
}

/**
 * Computes rolling average over last N frames to suppress prediction jitter.
 * Applies detection threshold (>= 80%).
 */
function processPredictions(predictions, hasPose) {
  if (!hasPose || !predictions) {
    setDetectedPosture('No Person Detected', 0);
    return;
  }

  // 1. Push raw frame prediction to smoothing queue
  const frameMap = {};
  predictions.forEach(p => {
    frameMap[p.className] = p.probability;
  });

  state.predictionHistory.push(frameMap);
  if (state.predictionHistory.length > state.smoothingWindowSize) {
    state.predictionHistory.shift();
  }

  // 2. Compute smoothed probabilities across current window
  const windowCount = state.predictionHistory.length;
  let highestClass = null;
  let highestProb = -1;

  state.classLabels.forEach(label => {
    let sum = 0;
    state.predictionHistory.forEach(frame => {
      sum += (frame[label] || 0);
    });
    const avg = sum / windowCount;
    state.smoothedProbabilities[label] = avg;

    // Update dynamic meter UI
    updateSingleMeter(label, avg);

    if (avg > highestProb) {
      highestProb = avg;
      highestClass = label;
    }
  });

  // 3. Thresholding: Only declare class if >= 0.80, otherwise "Detecting..."
  if (highestProb >= state.detectionThreshold && highestClass) {
    setDetectedPosture(highestClass, highestProb);
  } else {
    setDetectedPosture('Detecting...', highestProb);
  }
}

/**
 * Updates UI status badges, tips, and manages bad posture alert countdown
 * Evaluates posture using isGoodPosture().
 */
function setDetectedPosture(postureName, probability) {
  state.currentActivePosture = postureName;

  const meta = getPostureMeta(postureName);

  // Update Status Banner in Analysis Card
  DOM.currentPostureLabel.textContent = postureName;
  DOM.currentPostureSub.textContent = meta.title;
  DOM.postureTipText.textContent = meta.tip;

  DOM.statusBannerBox.className = 'status-banner ' + meta.className;

  // Update Confidence Tag
  if (probability > 0) {
    DOM.currentConfidence.textContent = `Confidence: ${(probability * 100).toFixed(1)}%`;
  } else {
    DOM.currentConfidence.textContent = 'Confidence: --%';
  }

  // Update Canvas Overlay Badge
  DOM.canvasOverlayBadge.textContent = postureName;
  DOM.canvasOverlayBadge.className = 'overlay-posture-badge ' + meta.badgeClass;

  // Evaluate Bad Posture Alert Logic
  handlePostureAlertTimer(postureName);
}

// ============================================================================
// 3. Smart Alert Engine & Notifications
// ============================================================================

/**
 * Updates small debug telemetry line:
 * label, isGood (true/false), badTimerSeconds, alarmActive (true/false), secondsUntilAlarm
 */
function updateDebugLine(label, isGood, badTimerSeconds) {
  const alarmActive = Boolean(state.isAlarmActive);
  const secondsUntilAlarm = alarmActive ? 0 : Math.max(0, state.alertDelaySeconds - (badTimerSeconds || 0));
  const text = `label: "${label}" | isGood: ${isGood} | badTimerSeconds: ${badTimerSeconds}s | alarmActive: ${alarmActive} | secondsUntilAlarm: ${secondsUntilAlarm}s`;
  if (DOM.debugLine) {
    DOM.debugLine.textContent = text;
  }
}

/**
 * Monitors continuous bad posture duration and triggers continuous alarm.
 * Rules:
 * 1. Uses isGoodPosture(label) for all good/bad decisions.
 * 2. Starts the bad-posture timer ONLY when label is not good AND not "Detecting...".
 * 3. When timer reaches delay, starts a continuous alarm (beeps every ~1s, banner, Notification).
 * 4. When posture becomes good, immediately resets timer to 0, stops any alarm, and hides banner.
 */
function handlePostureAlertTimer(posture) {
  const isGood = isGoodPosture(posture);
  const isDetecting = (
    posture === 'Detecting...' ||
    posture === 'No Person Detected' ||
    posture.toLowerCase().includes('detect')
  );
  const isBad = !isGood && !isDetecting;

  let badTimerSeconds = 0;

  if (isBad) {
    // Start continuous timer if not already ticking
    if (!state.badPostureStartTime) {
      state.badPostureStartTime = Date.now();
    }

    const elapsedMs = Date.now() - state.badPostureStartTime;
    badTimerSeconds = Math.floor(elapsedMs / 1000);
    const thresholdSec = state.alertDelaySeconds;

    // Trigger continuous alarm if threshold reached and alarm not already running
    if (badTimerSeconds >= thresholdSec && !state.isAlarmActive) {
      startContinuousAlarm(posture, badTimerSeconds);
    }

    // The "Alert Active" badge must be visible ONLY while the alert banner is actually showing
    if (state.isAlertBannerShowing) {
      DOM.badPostureTimerPill.style.display = 'flex';
      DOM.timerPillText.textContent = `Alert Active (${posture})`;
    } else {
      DOM.badPostureTimerPill.style.display = 'none';
    }
  } else {
    // Posture is good or detecting:
    // Reset timer to 0, stop any active alarm, hide alert banner and remove badge
    state.badPostureStartTime = null;
    badTimerSeconds = 0;
    stopAlarm();
    hideAlertBanner();
  }

  // Update small debug line
  updateDebugLine(posture, isGood, badTimerSeconds);
}

/**
 * Starts continuous alarm: repeats Web Audio beep every ~1 second (via setInterval),
 * displays prominent alert banner with Dismiss button, and sends desktop Notification.
 * Counts as one alert in the session stats (once per alarm, not per beep).
 */
function startContinuousAlarm(badPostureName, secondsElapsed) {
  // Ensure only ONE alarm interval exists at a time
  stopAlarm();

  state.isAlarmActive = true;
  state.totalAlertsTriggered++;
  if (DOM.statAlertCount) {
    DOM.statAlertCount.textContent = state.totalAlertsTriggered;
  }

  // 1. Play first beep immediately, then repeat every ~1 second via setInterval
  playAlertBeep();
  state.alarmIntervalId = setInterval(() => {
    playAlertBeep();
  }, 1000);

  // 2. Display on-screen alert banner with Dismiss button
  showAlertBanner(
    `Sustained ${badPostureName} Detected!`,
    `You have maintained ${badPostureName.toLowerCase()} continuously for ${secondsElapsed} seconds. Please adjust your back and neck alignment.`
  );

  // 3. Send browser Notification with requireInteraction: true
  sendPostureNotification(badPostureName, secondsElapsed);
}

// Backwards-compatible alias for any external or internal caller
function triggerPostureAlert(badPostureName, secondsElapsed) {
  startContinuousAlarm(badPostureName, secondsElapsed);
}

/**
 * Clears the continuous alarm interval and cleans up active notification.
 * Ensures only ONE alarm interval exists at a time and prevents overlap.
 */
function stopAlarm() {
  if (state.alarmIntervalId) {
    clearInterval(state.alarmIntervalId);
    state.alarmIntervalId = null;
  }
  state.isAlarmActive = false;

  if (state.activeNotification) {
    try {
      const notif = state.activeNotification;
      state.activeNotification = null;
      notif.onclick = null;
      notif.onclose = null;
      notif.close();
    } catch (e) {
      console.warn('[PostureGuard] Error closing notification:', e);
    }
  }
}

/**
 * Handles alert dismissal when user clicks "Dismiss" (or closes/clicks Notification).
 * Stops beeping, hides banner, and if posture is still bad, resets bad-posture
 * timer to 0 to start counting from zero again for another full delay cycle.
 */
function dismissAlarm() {
  stopAlarm();
  hideAlertBanner();

  const currentPosture = state.currentActivePosture;
  const isGood = isGoodPosture(currentPosture);
  const isDetecting = (
    currentPosture === 'Detecting...' ||
    currentPosture === 'No Person Detected' ||
    currentPosture.toLowerCase().includes('detect')
  );
  const isStillBad = !isGood && !isDetecting;

  if (isStillBad) {
    // Reset timer to 0 and start counting again from zero
    state.badPostureStartTime = Date.now();
    updateDebugLine(currentPosture, false, 0);
  } else {
    state.badPostureStartTime = null;
    updateDebugLine(currentPosture, isGood, 0);
  }

  updatePipUI();
}

/**
 * Sends desktop Notification with requireInteraction: true.
 * Clicking or closing the notification dismisses the alarm.
 */
function sendPostureNotification(badPostureName, secondsElapsed) {
  if ('Notification' in window && Notification.permission === 'granted') {
    const meta = getPostureMeta(badPostureName);
    const tipText = meta.tip || 'Sit upright and align your spine.';
    try {
      if (state.activeNotification) {
        state.activeNotification.close();
        state.activeNotification = null;
      }

      const notif = new Notification('PostureGuard', {
        body: `${badPostureName} detected (${secondsElapsed}s)! ${tipText}`,
        icon: 'favicon.ico',
        requireInteraction: true, // Stays until user dismisses it
        tag: 'postureguard-alert'
      });

      notif.onclick = () => {
        dismissAlarm();
        try { window.focus(); } catch (e) {}
      };

      notif.onclose = () => {
        if (state.isAlarmActive) {
          dismissAlarm();
        }
      };

      state.activeNotification = notif;
    } catch (notifErr) {
      console.warn('[PostureGuard] Notification error:', notifErr);
    }
  }
}

/**
 * Resumes AudioContext on user interaction so audio works even in background tabs
 */
function resumeAudioContext() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;
    if (!state.audioContext) {
      state.audioContext = new AudioContextClass();
    }
    if (state.audioContext.state === 'suspended') {
      state.audioContext.resume();
    }
  } catch (audioErr) {
    console.warn('[PostureGuard] AudioContext resume error:', audioErr);
  }
}

/**
 * Synthesizes a clean two-tone chime using Web Audio API (Web Audio oscillator)
 * Silenced if state.isMuted is true.
 */
function playAlertBeep() {
  if (state.isMuted) return;

  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return;

    if (!state.audioContext) {
      state.audioContext = new AudioContextClass();
    }
    if (state.audioContext.state === 'suspended') {
      state.audioContext.resume();
    }

    const ctx = state.audioContext;
    const now = ctx.currentTime;

    // Tone 1: Gentle alert fundamental (523.25 Hz - C5)
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(523.25, now);
    osc1.frequency.exponentialRampToValueAtTime(783.99, now + 0.15); // Ramp up to G5

    gain1.gain.setValueAtTime(0.25, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.35);

    // Tone 2: Harmonious chime completion (659.25 Hz - E5)
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(659.25, now + 0.18);
    osc2.frequency.setValueAtTime(1046.50, now + 0.28); // C6 peak

    gain2.gain.setValueAtTime(0.001, now);
    gain2.gain.setValueAtTime(0.22, now + 0.18);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.55);

    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.18);
    osc2.stop(now + 0.55);

  } catch (audioErr) {
    console.warn('[PostureGuard] Web Audio API failed:', audioErr);
  }
}

function showAlertBanner(title, message) {
  state.isAlertBannerShowing = true;
  if (DOM.alertBannerTitle) DOM.alertBannerTitle.textContent = title;
  if (DOM.alertBannerMessage) DOM.alertBannerMessage.textContent = message;
  if (DOM.alertBanner) DOM.alertBanner.style.display = 'flex';
}

function hideAlertBanner() {
  state.isAlertBannerShowing = false;
  if (DOM.alertBanner) DOM.alertBanner.style.display = 'none';
  if (DOM.badPostureTimerPill) DOM.badPostureTimerPill.style.display = 'none';
}

// ============================================================================
// 4. Document Picture-in-Picture Floating Mini Window
// ============================================================================

/**
 * Checks support for the Document Picture-in-Picture API.
 * If unsupported, hides button and shows desktop note.
 */
function checkPipSupport() {
  const isSupported = ('documentPictureInPicture' in window);
  if (DOM.btnFloatingPip) {
    DOM.btnFloatingPip.style.display = isSupported ? 'inline-flex' : 'none';
  }
  if (DOM.pipUnsupportedNote) {
    DOM.pipUnsupportedNote.style.display = isSupported ? 'none' : 'inline-block';
  }
}

/**
 * Toggles always-on-top Document Picture-in-Picture window
 */
async function togglePipWindow() {
  if (!('documentPictureInPicture' in window)) return;

  if (state.pipWindow && !state.pipWindow.closed) {
    state.pipWindow.close();
    state.pipWindow = null;
    updatePipButtonText();
    return;
  }

  try {
    const pipWindow = await window.documentPictureInPicture.requestWindow({
      width: 330,
      height: 220
    });
    state.pipWindow = pipWindow;

    // Inject styles and markup into PiP document
    pipWindow.document.title = 'PostureGuard — Mini Coach';
    pipWindow.document.head.innerHTML = `
      <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: #0a1120; color: #f8fafc; padding: 14px; display: flex; flex-direction: column; justify-content: space-between; height: 100vh; user-select: none; }
        .pip-header { display: flex; align-items: center; justify-content: space-between; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #38bdf8; font-weight: 700; }
        .pip-status-box { background: rgba(18, 30, 56, 0.85); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 12px; text-align: center; margin: 8px 0; }
        .pip-badge { display: inline-block; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 14px; margin-bottom: 6px; }
        .badge-good { background: rgba(6, 78, 59, 0.9); color: #a7f3d0; border: 1px solid #10b981; }
        .badge-bad { background: rgba(120, 53, 15, 0.9); color: #fde68a; border: 1px solid #f59e0b; }
        .badge-neutral { background: rgba(30, 41, 59, 0.9); color: #cbd5e1; border: 1px solid #64748b; }
        .pip-timer { font-family: monospace; font-size: 12px; color: #94a3b8; }
        .pip-timer.active-alert { color: #f87171; font-weight: bold; }
        .pip-footer { display: flex; gap: 8px; justify-content: center; }
        .btn-pip-stop { background: #ef4444; color: white; border: none; padding: 6px 18px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
        .btn-pip-stop:hover { background: #dc2626; }
      </style>
    `;

    pipWindow.document.body.innerHTML = `
      <div class="pip-header">
        <span>PostureGuard Mini</span>
        <span id="pipIndicator">🟢 Active</span>
      </div>
      <div class="pip-status-box">
        <div id="pipBadge" class="pip-badge badge-neutral">${state.currentActivePosture}</div>
        <div id="pipTimer" class="pip-timer">Timer: 0s</div>
      </div>
      <div class="pip-footer">
        <button id="pipStopBtn" class="btn-pip-stop">Stop</button>
      </div>
    `;

    pipWindow.document.getElementById('pipStopBtn').addEventListener('click', () => {
      pauseMonitoring();
      if (DOM.btnStopSession) {
        showSessionSummary();
      }
      pipWindow.close();
    });

    pipWindow.addEventListener('pagehide', () => {
      state.pipWindow = null;
      updatePipButtonText();
    });

    updatePipUI();
    updatePipButtonText();
  } catch (pipErr) {
    console.warn('[PostureGuard] Failed to open Document PiP window:', pipErr);
  }
}

/**
 * Updates content inside Document Picture-in-Picture window
 */
function updatePipUI() {
  if (!state.pipWindow || state.pipWindow.closed) return;
  const doc = state.pipWindow.document;
  const badgeEl = doc.getElementById('pipBadge');
  const timerEl = doc.getElementById('pipTimer');
  const indicatorEl = doc.getElementById('pipIndicator');

  if (badgeEl) {
    badgeEl.textContent = state.currentActivePosture;
    if (isGoodPosture(state.currentActivePosture)) {
      badgeEl.className = 'pip-badge badge-good';
    } else if (state.currentActivePosture.includes('Detect') || state.currentActivePosture.includes('No Person')) {
      badgeEl.className = 'pip-badge badge-neutral';
    } else {
      badgeEl.className = 'pip-badge badge-bad';
    }
  }

  if (timerEl) {
    if (state.badPostureStartTime) {
      const elapsedSec = Math.floor((Date.now() - state.badPostureStartTime) / 1000);
      if (state.isAlertBannerShowing) {
        timerEl.textContent = `ALERT ACTIVE (${elapsedSec}s)`;
        timerEl.className = 'pip-timer active-alert';
      } else {
        const remaining = Math.max(0, state.alertDelaySeconds - elapsedSec);
        timerEl.textContent = `Alert in ${remaining}s (${elapsedSec}s)`;
        timerEl.className = 'pip-timer';
      }
    } else {
      timerEl.textContent = 'Timer: 0s';
      timerEl.className = 'pip-timer';
    }
  }

  if (indicatorEl) {
    indicatorEl.textContent = state.isPaused ? '⏸️ Paused' : '🟢 Active';
  }
}

function updatePipButtonText() {
  if (!DOM.btnFloatingPipText) return;
  if (state.pipWindow && !state.pipWindow.closed) {
    DOM.btnFloatingPipText.textContent = 'Close Mini Window';
  } else {
    DOM.btnFloatingPipText.textContent = 'Floating Mini Window';
  }
}

// ============================================================================
// 5. Session Telemetry & Dashboard Tracking
// ============================================================================

/**
 * Starts interval timer that updates active session duration,
 * posture distribution, and rolling timeline canvas every second.
 */
function startSessionTracking() {
  state.sessionStartTime = Date.now();

  state.sessionTimerInterval = setInterval(() => {
    if (!state.isMonitoring || state.isPaused) return;

    state.activeSessionDuration++;

    // Accumulate time for currently active posture
    const current = state.currentActivePosture;
    if (state.postureDurations[current] !== undefined) {
      state.postureDurations[current]++;
    } else {
      state.postureDurations[current] = 1;
    }

    // Add current color to rolling 60-second timeline history using isGoodPosture
    const postureColor = getPostureColor(current);
    state.timelineHistory.push(postureColor);
    if (state.timelineHistory.length > 60) {
      state.timelineHistory.shift();
    }

    // Update telemetry UI
    updateDashboardUI();
    drawTimelineCanvas();

  }, 1000);
}

/**
 * Updates all live dashboard stats, score ratios, and multi-segment bar
 * Evaluates good posture time dynamically using isGoodPosture().
 */
function updateDashboardUI() {
  const total = state.activeSessionDuration;
  const timeFormatted = formatTime(total);

  // Header & Metric 1
  DOM.headerTimer.textContent = timeFormatted;
  DOM.statSessionTime.textContent = timeFormatted;

  // Good Posture calculations using isGoodPosture()
  let goodTime = 0;
  let worstClass = 'None (Optimal)';
  let maxBadTime = 0;

  Object.entries(state.postureDurations).forEach(([label, sec]) => {
    if (isGoodPosture(label)) {
      goodTime += sec;
    } else if (label !== 'Detecting...' && label !== 'No Person Detected') {
      if (sec > maxBadTime) {
        maxBadTime = sec;
        worstClass = label;
      }
    }
  });

  const ratio = total > 0 ? Math.round((goodTime / total) * 100) : 100;

  DOM.headerGoodPercent.textContent = `${ratio}%`;
  DOM.statGoodRatio.textContent = `${ratio}%`;
  DOM.statGoodTime.textContent = `${formatTime(goodTime)} upright posture`;

  if (maxBadTime > 0) {
    DOM.statPrimaryConcern.textContent = worstClass;
    DOM.statConcernDetail.textContent = `${formatTime(maxBadTime)} detected in session`;
  } else {
    DOM.statPrimaryConcern.textContent = 'None (Optimal)';
    DOM.statConcernDetail.textContent = 'Maintaining healthy spine alignment';
  }

  // Update dynamic multi-segment distribution bar and legends
  updateDistributionUI(total);
}

/**
 * Dynamically renders distribution bar segments and legends
 * based on the exact classes loaded from metadata.json
 */
function renderDistributionUI() {
  const barContainer = DOM.distributionBar;
  const legendContainer = document.querySelector('.legend-row');

  if (barContainer && state.classLabels.length > 0) {
    barContainer.innerHTML = '';
    state.classLabels.forEach(label => {
      const seg = document.createElement('div');
      seg.className = 'bar-segment';
      seg.id = `bar-seg-${sanitizeId(label)}`;
      seg.style.backgroundColor = getPostureColor(label);
      seg.style.width = isGoodPosture(label) ? '100%' : '0%';
      barContainer.appendChild(seg);
    });
  }

  if (legendContainer && state.classLabels.length > 0) {
    legendContainer.innerHTML = '';
    state.classLabels.forEach(label => {
      const item = document.createElement('div');
      item.className = 'legend-item';
      const color = getPostureColor(label);
      item.innerHTML = `
        <span class="legend-color" style="background-color: ${color}"></span>
        <span>${label}: <b id="leg-val-${sanitizeId(label)}" class="font-mono">${isGoodPosture(label) ? '100%' : '0%'}</b></span>
      `;
      legendContainer.appendChild(item);
    });
  }
}

function updateDistributionUI(total) {
  if (total <= 0) return;

  state.classLabels.forEach(label => {
    const sec = state.postureDurations[label] || 0;
    const pct = ((sec / total) * 100).toFixed(1);
    const safeId = sanitizeId(label);

    const seg = document.getElementById(`bar-seg-${safeId}`);
    if (seg) seg.style.width = `${pct}%`;

    const legVal = document.getElementById(`leg-val-${safeId}`);
    if (legVal) legVal.textContent = `${pct}%`;
  });
}

/**
 * Draws rolling 60-second posture history strip on a plain canvas
 * (No external chart library used)
 */
function drawTimelineCanvas() {
  const canvas = DOM.timelineCanvas;
  const ctx = DOM.timelineCtx;
  const width = canvas.width = canvas.parentElement.clientWidth || 360;
  const height = canvas.height = 52;

  ctx.clearRect(0, 0, width, height);

  const maxSlots = 60;
  const history = state.timelineHistory;
  const slotWidth = width / maxSlots;

  // Background grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i < maxSlots; i += 10) {
    ctx.beginPath();
    ctx.moveTo(i * slotWidth, 0);
    ctx.lineTo(i * slotWidth, height);
    ctx.stroke();
  }

  // Draw 60 posture color blocks
  const startIndex = maxSlots - history.length;
  history.forEach((color, idx) => {
    const x = (startIndex + idx) * slotWidth;
    ctx.fillStyle = color;
    ctx.fillRect(x + 0.5, 4, Math.max(1, slotWidth - 1), height - 8);
  });
}

// ============================================================================
// 6. Session Stop & Summary Report Modal
// ============================================================================

/**
 * Stops active monitoring, computes final Posture Score (out of 100),
 * and presents the Session Summary modal dialog.
 * Evaluates good posture duration using isGoodPosture().
 */
function showSessionSummary() {
  pauseMonitoring();

  const total = state.activeSessionDuration;
  let goodTime = 0;
  Object.entries(state.postureDurations).forEach(([label, sec]) => {
    if (isGoodPosture(label)) goodTime += sec;
  });

  const score = total > 0 ? Math.round((goodTime / total) * 100) : 100;

  // Populate Score
  DOM.modalScoreValue.textContent = score;
  DOM.modalStatDuration.textContent = formatTime(total);
  DOM.modalStatGoodTime.textContent = formatTime(goodTime);
  DOM.modalStatAlerts.textContent = state.totalAlertsTriggered;
  DOM.modalStatThreshold.textContent = `${state.alertDelaySeconds}s`;

  // Animate SVG Score Circle (circumference = 2 * PI * 50 ≈ 314)
  const circleOffset = 314 - (314 * score / 100);
  DOM.modalScoreCircle.style.strokeDashoffset = circleOffset;

  // Rating grade and theme color
  let grade = 'Ergonomic Champion';
  let gradeColor = '#10b981';

  if (score >= 90) {
    grade = 'Ergonomic Champion (A+)';
    gradeColor = '#10b981';
  } else if (score >= 75) {
    grade = 'Good Spine Discipline (B)';
    gradeColor = '#38bdf8';
  } else if (score >= 60) {
    grade = 'Moderate Postural Fatigue (C)';
    gradeColor = '#f59e0b';
  } else {
    grade = 'High Spinal Strain Risk (D)';
    gradeColor = '#f43f5e';
  }

  DOM.modalScoreGrade.textContent = grade;
  DOM.modalScoreGrade.style.color = gradeColor;
  DOM.modalScoreCircle.style.stroke = gradeColor;

  // Build Breakdown List dynamically for all classes
  DOM.modalBreakdownList.innerHTML = '';
  Object.keys(state.postureDurations).forEach(label => {
    const sec = state.postureDurations[label] || 0;
    const pct = total > 0 ? ((sec / total) * 100).toFixed(1) : '0.0';
    const color = getPostureColor(label);

    const row = document.createElement('div');
    row.className = 'modal-breakdown-row';
    row.innerHTML = `
      <span style="display:flex; align-items:center; gap:0.5rem;">
        <span style="width:8px; height:8px; border-radius:50%; background-color:${color};"></span>
        <strong>${label}</strong>
      </span>
      <span class="font-mono text-secondary">${formatTime(sec)} (${pct}%)</span>
    `;
    DOM.modalBreakdownList.appendChild(row);
  });

  // Targeted Ergonomic Advice based on student's most prevalent bad posture
  DOM.modalAdviceText.textContent = generatePersonalizedAdvice(score);

  // Show HTML5 Dialog
  if (typeof DOM.summaryModal.showModal === 'function') {
    DOM.summaryModal.showModal();
  } else {
    DOM.summaryModal.setAttribute('open', '');
  }
}

/**
 * Generates tailored ergonomic feedback based on session's data
 */
function generatePersonalizedAdvice(score) {
  if (score >= 85) {
    return 'Outstanding spinal hygiene! You kept neutral cervical alignment throughout your study session. Continue scheduling a 20-second break every 30 minutes to stand and hydrate.';
  }

  // Find dominant bad posture dynamically using isGoodPosture
  let maxBadClass = null;
  let maxBadSec = 0;
  Object.entries(state.postureDurations).forEach(([label, sec]) => {
    if (!isGoodPosture(label) && label !== 'Detecting...' && label !== 'No Person Detected') {
      if (sec > maxBadSec) {
        maxBadSec = sec;
        maxBadClass = label;
      }
    }
  });

  if (maxBadClass) {
    const lower = maxBadClass.toLowerCase();
    if (lower.includes('neck')) {
      return `Cervical Neck Strain Detected (${maxBadClass}): Your primary fatigue pattern is forward head tilt. We recommend elevating your laptop/monitor so the top third of your display is level with your eyes.`;
    }
    if (lower.includes('slouch')) {
      return `Lumbar Slouching Detected (${maxBadClass}): You tend to round your lower spine during prolonged sitting. Try placing a small cushion or lumbar roll behind your lower back, and adjust your seat depth so your feet rest flat.`;
    }
    if (lower.includes('lean')) {
      return `Lateral Tilt Detected (${maxBadClass}): You frequently lean toward one side. Ensure your keyboard and mouse are directly centered in front of you, and avoid resting on one elbow.`;
    }
    return `Postural Strain Detected (${maxBadClass}): Regular micro-breaks will help reset postural fatigue. Take 30 seconds every 25 minutes to roll your shoulders backwards and sit upright.`;
  }

  return 'Regular micro-breaks will help reset postural fatigue. Take 30 seconds every 25 minutes to roll your shoulders backwards and perform chin tucks.';
}

// ============================================================================
// 7. Interactive Controls & UI Handlers
// ============================================================================

/**
 * Starts or resumes posture monitoring session
 */
async function startMonitoring() {
  // Resume AudioContext on Start button click so beeps work even when the tab is hidden
  resumeAudioContext();

  // Ask for Notification permission when user clicks Start
  if ('Notification' in window && Notification.permission === 'default') {
    try {
      const perm = await Notification.requestPermission();
      state.notificationsEnabled = (perm === 'granted');
      if (perm === 'granted') {
        DOM.notificationBtnText.textContent = 'Desktop Alerts: Enabled';
        DOM.requestNotificationBtn.classList.add('btn-primary');
        DOM.requestNotificationBtn.classList.remove('btn-outline');
      }
    } catch (notifErr) {
      console.warn('[PostureGuard] Notification request error:', notifErr);
    }
  } else if ('Notification' in window) {
    state.notificationsEnabled = (Notification.permission === 'granted');
  }

  if (!state.isModelLoaded) {
    const loaded = await loadModel();
    if (!loaded) return;
  }

  if (!state.webcam) {
    const camStarted = await startWebcam();
    if (!camStarted) return;
  }

  state.isMonitoring = true;
  state.isPaused = false;

  // Update UI status badges
  DOM.sessionStatusPill.textContent = 'Active Monitoring';
  DOM.sessionStatusPill.className = 'status-indicator status-running';
  DOM.btnPlayPauseText.textContent = 'Pause Monitoring';
  DOM.btnPlayPauseIcon.innerHTML = `
    <rect x="6" y="4" width="4" height="16"></rect>
    <rect x="14" y="4" width="4" height="16"></rect>
  `;
  DOM.btnStopSession.disabled = false;

  // Start telemetry interval if not active
  if (!state.sessionTimerInterval) {
    startSessionTracking();
  }

  // Launch foreground or background inference loop depending on visibility
  startInferenceLoop();
  updatePipUI();
}

/**
 * Pauses posture monitoring
 */
function pauseMonitoring() {
  state.isPaused = true;
  stopInferenceLoop();
  stopAlarm();
  hideAlertBanner();

  DOM.sessionStatusPill.textContent = 'Paused';
  DOM.sessionStatusPill.className = 'status-indicator status-paused';
  DOM.btnPlayPauseText.textContent = 'Resume Monitoring';
  DOM.btnPlayPauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
  updatePipUI();
}

/**
 * Toggles between Start / Pause
 */
function toggleMonitoring() {
  if (state.isSimulatedMode) {
    toggleSimulation();
    return;
  }

  if (!state.isMonitoring || state.isPaused) {
    startMonitoring();
  } else {
    pauseMonitoring();
  }
}

/**
 * Resets entire session data and counters
 */
function resetSession() {
  stopAlarm();
  pauseMonitoring();

  // Reset counters
  state.activeSessionDuration = 0;
  state.totalAlertsTriggered = 0;
  state.timelineHistory = [];
  state.badPostureStartTime = null;
  state.isAlarmActive = false;

  state.classLabels.forEach(label => {
    state.postureDurations[label] = 0;
    state.smoothedProbabilities[label] = 0;
    updateSingleMeter(label, 0);
  });

  if (state.sessionTimerInterval) {
    clearInterval(state.sessionTimerInterval);
    state.sessionTimerInterval = null;
  }

  // Reset UI elements
  DOM.headerTimer.textContent = '00:00';
  DOM.statSessionTime.textContent = '00:00';
  DOM.headerGoodPercent.textContent = '100%';
  DOM.statGoodRatio.textContent = '100%';
  DOM.statGoodTime.textContent = '00:00 upright posture';
  DOM.statAlertCount.textContent = '0';
  DOM.statPrimaryConcern.textContent = 'None (Optimal)';
  DOM.statConcernDetail.textContent = 'Maintaining spine health';

  renderDistributionUI();

  DOM.sessionStatusPill.textContent = 'Ready';
  DOM.sessionStatusPill.className = 'status-indicator status-idle';
  DOM.btnPlayPauseText.textContent = 'Start Monitoring';
  DOM.btnPlayPauseIcon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
  DOM.btnStopSession.disabled = true;

  drawTimelineCanvas();
  hideAlertBanner();
  DOM.badPostureTimerPill.style.display = 'none';

  setDetectedPosture('Detecting...', 0);
  updateDebugLine('Detecting...', false, 0);
  updatePipUI();
}

// ============================================================================
// 8. Dynamic Class Probability Meters
// ============================================================================

/**
 * Builds UI meters dynamically for every class discovered in metadata.json
 */
function renderClassMetersUI() {
  DOM.classProbabilitiesContainer.innerHTML = '';

  state.classLabels.forEach(label => {
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.id = `prob-item-${sanitizeId(label)}`;

    let fillClass = 'fill-default';
    let dotColor = '#38bdf8';
    if (isGoodPosture(label)) {
      fillClass = 'fill-good';
      dotColor = '#10b981';
    } else {
      const lower = label.toLowerCase();
      if (lower.includes('slouch')) { fillClass = 'fill-slouch'; dotColor = '#f59e0b'; }
      else if (lower.includes('neck')) { fillClass = 'fill-neck'; dotColor = '#f43f5e'; }
      else if (lower.includes('lean')) { fillClass = 'fill-lean'; dotColor = '#a855f7'; }
    }

    item.innerHTML = `
      <div class="prob-meta">
        <span class="prob-label">
          <span class="prob-color-dot" style="background-color: ${dotColor}"></span>
          ${label}
        </span>
        <span class="prob-pct font-mono" id="prob-pct-${sanitizeId(label)}">0%</span>
      </div>
      <div class="prob-meter-track">
        <div class="prob-meter-fill ${fillClass}" id="prob-fill-${sanitizeId(label)}" style="width: 0%;"></div>
      </div>
    `;
    DOM.classProbabilitiesContainer.appendChild(item);
  });
}

function updateSingleMeter(className, probability) {
  const safeId = sanitizeId(className);
  const fillEl = document.getElementById(`prob-fill-${safeId}`);
  const pctEl = document.getElementById(`prob-pct-${safeId}`);

  if (fillEl && pctEl) {
    const pct = Math.min(100, Math.max(0, Math.round(probability * 100)));
    fillEl.style.width = `${pct}%`;
    pctEl.textContent = `${pct}%`;
  }
}

function sanitizeId(str) {
  return str.replace(/[^a-zA-Z0-9]/g, '-').toLowerCase();
}

// ============================================================================
// 9. Simulation Mode (For Viva / Demo / Fallback Evaluation)
// ============================================================================

/**
 * Runs simulated session demonstrating all features, dynamic classes,
 * Web Audio chime, alert banner, and PiP without physical webcam/model files.
 */
function startSimulationMode() {
  resumeAudioContext();
  hideError();
  DOM.videoPlaceholder.style.display = 'none';
  state.isSimulatedMode = true;
  state.isMonitoring = true;
  state.isPaused = false;

  // Synthesize classes
  state.classLabels = ['Good posture', 'Slouching'];
  state.classLabels.forEach(label => {
    state.postureDurations[label] = 0;
    state.smoothedProbabilities[label] = 0;
  });
  renderClassMetersUI();
  renderDistributionUI();

  DOM.sessionStatusPill.textContent = 'Simulated Demo';
  DOM.sessionStatusPill.className = 'status-indicator status-running';
  DOM.btnPlayPauseText.textContent = 'Pause Demo';
  DOM.btnStopSession.disabled = false;
  DOM.camStatusDot.classList.add('active');

  startSessionTracking();

  // Synthetic animation sequence
  const postures = [
    { name: 'Good posture', duration: 8 },
    { name: 'Slouching', duration: 10 },
    { name: 'Good posture', duration: 10 }
  ];

  let currentSeqIdx = 0;
  let seqTimer = 0;

  state.simulatedInterval = setInterval(() => {
    if (state.isPaused) return;

    seqTimer++;
    const currentItem = postures[currentSeqIdx];

    // Simulate probabilities
    state.classLabels.forEach(label => {
      const isTarget = (label === currentItem.name);
      const prob = isTarget ? 0.92 : 0.08;
      updateSingleMeter(label, prob);
    });

    setDetectedPosture(currentItem.name, 0.92);
    drawSimulatedCanvas(currentItem.name);
    updatePipUI();

    if (seqTimer >= currentItem.duration) {
      seqTimer = 0;
      currentSeqIdx = (currentSeqIdx + 1) % postures.length;
    }
  }, 1000);
}

function drawSimulatedCanvas(postureName) {
  const canvas = DOM.canvas;
  const ctx = DOM.ctx;
  ctx.fillStyle = '#090f1d';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Grid background
  ctx.strokeStyle = 'rgba(56, 189, 248, 0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < canvas.width; x += 40) {
    ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height);
  }
  for (let y = 0; y < canvas.height; y += 40) {
    ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
  }
  ctx.stroke();

  // Draw simulated stick figure skeleton
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const color = getPostureColor(postureName);

  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';

  let headX = cx;
  let headY = cy - 70;
  let spineTiltX = 0;

  const lower = postureName.toLowerCase();
  if (lower.includes('neck')) {
    headX = cx + 45;
    headY = cy - 45;
  } else if (lower.includes('slouch')) {
    headY = cy - 35;
    spineTiltX = -25;
  } else if (lower.includes('lean')) {
    headX = cx - 40;
    spineTiltX = -35;
  }

  // Head
  ctx.beginPath();
  ctx.arc(headX, headY, 24, 0, Math.PI * 2);
  ctx.stroke();

  // Neck to spine
  ctx.beginPath();
  ctx.moveTo(headX, headY + 24);
  ctx.lineTo(cx + spineTiltX, cy + 20);
  ctx.lineTo(cx, cy + 110);
  ctx.stroke();

  // Shoulders
  ctx.beginPath();
  ctx.moveTo(headX - 55, headY + 45);
  ctx.lineTo(headX + 55, headY + 45);
  ctx.stroke();

  // Arms
  ctx.beginPath();
  ctx.moveTo(headX - 55, headY + 45);
  ctx.lineTo(cx - 75, cy + 100);
  ctx.moveTo(headX + 55, headY + 45);
  ctx.lineTo(cx + 75, cy + 100);
  ctx.stroke();

  // Simulation watermark
  ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.font = '12px JetBrains Mono';
  ctx.fillText('SIMULATION DEMO MODE', 18, canvas.height - 20);
}

function toggleSimulation() {
  if (state.isPaused) {
    state.isPaused = false;
    DOM.sessionStatusPill.textContent = 'Simulated Demo';
    DOM.sessionStatusPill.className = 'status-indicator status-running';
    DOM.btnPlayPauseText.textContent = 'Pause Demo';
  } else {
    state.isPaused = true;
    DOM.sessionStatusPill.textContent = 'Demo Paused';
    DOM.sessionStatusPill.className = 'status-indicator status-paused';
    DOM.btnPlayPauseText.textContent = 'Resume Demo';
  }
}

// ============================================================================
// 10. Utilities, Helpers & Event Listeners
// ============================================================================

function formatTime(totalSeconds) {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function updateFPS() {
  state.frameCount++;
  const now = performance.now();
  if (now - state.lastFrameTime >= 1000) {
    state.currentFps = state.frameCount;
    DOM.fpsDisplay.textContent = `${state.currentFps} FPS`;
    state.frameCount = 0;
    state.lastFrameTime = now;
  }
}

function showLoading(title, subtitle) {
  DOM.loadingTitle.textContent = title;
  DOM.loadingSubtitle.textContent = subtitle;
  DOM.loadingOverlay.style.display = 'flex';
  DOM.errorOverlay.style.display = 'none';
}

function hideLoading() {
  DOM.loadingOverlay.style.display = 'none';
}

function showError(title, htmlMessage) {
  DOM.errorTitle.textContent = title;
  DOM.errorMessage.innerHTML = htmlMessage;
  DOM.errorOverlay.style.display = 'flex';
  DOM.loadingOverlay.style.display = 'none';
}

function hideError() {
  DOM.errorOverlay.style.display = 'none';
}

// Sound mute toggle
function toggleMute() {
  state.isMuted = !state.isMuted;
  const label = state.isMuted ? 'Sound: Muted' : 'Sound: On';
  DOM.audioBtnText.textContent = label;

  if (state.isMuted) {
    DOM.navSoundIcon.innerHTML = `
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
      <line x1="23" y1="9" x2="17" y2="15"></line>
      <line x1="17" y1="9" x2="23" y2="15"></line>
    `;
    DOM.navMuteBtn.title = 'Unmute Sound';
  } else {
    DOM.navSoundIcon.innerHTML = `
      <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
      <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
    `;
    DOM.navMuteBtn.title = 'Mute Sound';
  }
}

// Request Desktop Notifications
async function requestNotificationPermission() {
  if (!('Notification' in window)) {
    alert('Browser notifications are not supported in your browser.');
    return;
  }

  const permission = await Notification.requestPermission();
  if (permission === 'granted') {
    state.notificationsEnabled = true;
    DOM.notificationBtnText.textContent = 'Desktop Alerts: Enabled';
    DOM.requestNotificationBtn.classList.add('btn-primary');
    DOM.requestNotificationBtn.classList.remove('btn-outline');
  } else {
    DOM.notificationBtnText.textContent = 'Permission Denied';
  }
}

// --- Event Binding ---
function attachEventListeners() {
  // Main monitoring buttons
  DOM.btnToggleMonitoring.addEventListener('click', toggleMonitoring);
  DOM.btnStopSession.addEventListener('click', showSessionSummary);
  DOM.btnResetSession.addEventListener('click', resetSession);
  DOM.startFromPlaceholderBtn.addEventListener('click', startMonitoring);
  DOM.startPostureCheckBtn.addEventListener('click', () => {
    resumeAudioContext();
    const section = document.getElementById('monitor-section');
    section.scrollIntoView({ behavior: 'smooth' });
    if (!state.isMonitoring) {
      setTimeout(startMonitoring, 400);
    }
  });

  // Floating Mini Window (Document Picture-in-Picture)
  if (DOM.btnFloatingPip) {
    DOM.btnFloatingPip.addEventListener('click', togglePipWindow);
  }

  // Page visibility change handler for background loop switching
  document.addEventListener('visibilitychange', handleVisibilityChange);

  // Retry & simulation buttons
  DOM.retryInitBtn.addEventListener('click', startMonitoring);
  DOM.demoSimulateBtn.addEventListener('click', startSimulationMode);

  // Mirror toggle
  DOM.mirrorToggle.addEventListener('change', (e) => {
    state.mirrorFeed = e.target.checked;
    if (state.webcam) {
      startWebcam();
    }
  });

  // Settings: Alert Delay
  DOM.alertDelaySelect.addEventListener('change', (e) => {
    state.alertDelaySeconds = parseInt(e.target.value, 10);
    console.log('[PostureGuard] Alert delay updated to:', state.alertDelaySeconds, 'seconds');
  });

  // Settings: Audio & Mute
  DOM.toggleAudioBtn.addEventListener('click', toggleMute);
  DOM.navMuteBtn.addEventListener('click', toggleMute);
  DOM.testAudioBtn.addEventListener('click', () => {
    playAlertBeep();
  });

  // Notifications
  DOM.requestNotificationBtn.addEventListener('click', requestNotificationPermission);

  // Alert banner dismiss
  DOM.dismissAlertBtn.addEventListener('click', dismissAlarm);

  // Modal dialog controls
  DOM.modalCloseBtn.addEventListener('click', () => {
    DOM.summaryModal.close();
  });
  DOM.modalDismissBtn.addEventListener('click', () => {
    DOM.summaryModal.close();
  });
  DOM.modalNewSessionBtn.addEventListener('click', () => {
    DOM.summaryModal.close();
    resetSession();
    startMonitoring();
  });

  // Initial checks and draws
  checkPipSupport();
  drawTimelineCanvas();
  updateDebugLine('Detecting...', false, 0);
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  attachEventListeners();
  console.log('[PostureGuard] Application ready with background monitoring & Document PiP support.');
});
