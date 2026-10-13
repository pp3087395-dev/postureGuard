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

// Background Web Worker tick interval (~3.8 to 4 FPS, ~260 ms)
const BG_INTERVAL_MS = 260;

// Foreground inference cap interval (~10 FPS, ~100 ms)
const FG_INTERVAL_MS = 100;

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
  backgroundIntervalId: null, // Legacy / fallback interval
  isInBackground: false,
  mirrorFeed: true,
  isInferenceBusy: false,
  consecutiveSlowInferences: 0,
  fallbackAttempted: false,

  // Background Web Worker Ticker & Anti-Throttling
  tickerWorker: null,
  wakeLock: null,
  silentAudioNode: null,

  // Dynamic Prediction Smoothing Buffer (10 in FG, 5 in BG = ~1.0-1.3s)
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
  lastAlarmBeepTime: 0,
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

  // FPS Counter & Background Telemetry
  lastFrameTime: performance.now(),
  lastForegroundFrameTime: 0,
  lastTelemetryLogTime: performance.now(),
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
  modalAdviceText: document.getElementById('modalAdviceText'),

  // Local Streak Elements
  streakCard: document.getElementById('streakCard'),
  streakFlameIcon: document.getElementById('streakFlameIcon'),
  streakCurrentVal: document.getElementById('streakCurrentVal'),
  streakBestVal: document.getElementById('streakBestVal'),
  streakFreezesVal: document.getElementById('streakFreezesVal'),
  streakFreezePill: document.getElementById('streakFreezePill'),
  streakStatusText: document.getElementById('streakStatusText'),
  streakTodayProgressVal: document.getElementById('streakTodayProgressVal'),
  streakTodayStatusBadge: document.getElementById('streakTodayStatusBadge'),
  streakTodayProgressBar: document.getElementById('streakTodayProgressBar'),
  streakGoodRatioText: document.getElementById('streakGoodRatioText'),
  streakMilestonesList: document.getElementById('streakMilestonesList'),
  btnResetStreak: document.getElementById('btnResetStreak')
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
 * Screen Wake Lock Management
 * Prevents screen sleep and browser tab background throttling during sessions
 */
async function requestWakeLock() {
  if ('wakeLock' in navigator) {
    try {
      if (!state.wakeLock) {
        state.wakeLock = await navigator.wakeLock.request('screen');
        state.wakeLock.addEventListener('release', () => {
          state.wakeLock = null;
          console.log('[PostureGuard] Screen Wake Lock released');
        });
        console.log('[PostureGuard] Screen Wake Lock successfully acquired');
      }
    } catch (err) {
      console.warn('[PostureGuard] Screen Wake Lock warning:', err);
    }
  }
}

function releaseWakeLock() {
  if (state.wakeLock) {
    state.wakeLock.release().catch(() => {});
    state.wakeLock = null;
  }
}

/**
 * Web Audio Silent Loop Keep-Alive
 * Audio-producing tabs are granted high background priority by modern browsers
 */
function startSilentAudio() {
  try {
    resumeAudioContext();
    if (state.audioContext && !state.silentAudioNode) {
      const buffer = state.audioContext.createBuffer(1, state.audioContext.sampleRate, state.audioContext.sampleRate);
      const source = state.audioContext.createBufferSource();
      source.buffer = buffer;
      source.loop = true;
      source.connect(state.audioContext.destination);
      source.start();
      state.silentAudioNode = source;
      console.log('[PostureGuard] Silent audio keep-alive track active');
    }
  } catch (err) {
    console.warn('[PostureGuard] Silent audio keep-alive error:', err);
  }
}

function stopSilentAudio() {
  if (state.silentAudioNode) {
    try {
      state.silentAudioNode.stop();
      state.silentAudioNode.disconnect();
    } catch (e) {}
    state.silentAudioNode = null;
  }
}

/**
 * Initializes the Web Worker background ticker
 */
function initTickerWorker() {
  if (state.tickerWorker) return;
  try {
    state.tickerWorker = new Worker('worker-ticker.js');
  } catch (workerErr) {
    console.warn('[PostureGuard] Direct Worker creation failed, using Blob fallback:', workerErr);
    const workerBlobCode = `
      let timerId = null;
      self.onmessage = function(e) {
        const d = e.data || {};
        const action = typeof d === 'string' ? d : d.action;
        if (action === 'start') {
          const interval = d.interval || 260;
          if (timerId) clearInterval(timerId);
          timerId = setInterval(() => self.postMessage({ type: 'tick', timestamp: Date.now() }), interval);
        } else if (action === 'stop') {
          if (timerId) { clearInterval(timerId); timerId = null; }
        }
      };
    `;
    const blob = new Blob([workerBlobCode], { type: 'application/javascript' });
    state.tickerWorker = new Worker(URL.createObjectURL(blob));
  }

  state.tickerWorker.onmessage = async (e) => {
    if (e.data && e.data.type === 'tick') {
      if (state.isMonitoring && !state.isPaused && document.hidden) {
        if (!state.isInferenceBusy) {
          state.isInferenceBusy = true;
          try {
            await stepInference(true);
          } finally {
            state.isInferenceBusy = false;
          }
        }
      }
    }
  };
}

/**
 * Initializes the user's webcam using tmPose.Webcam with 320x240 resolution
 */
async function startWebcam() {
  showLoading('Starting Camera...', 'Requesting webcam permissions');

  const width = 320;
  const height = 240;
  const flip = state.mirrorFeed; // Mirror horizontally for intuitive user preview

  try {
    state.webcam = new tmPose.Webcam(width, height, flip);
    await state.webcam.setup(); // Request camera access
    await state.webcam.play();

    // Match canvas dimensions to webcam resolution for 1:1 crisp display
    DOM.canvas.width = width;
    DOM.canvas.height = height;

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
 * but continues PoseNet estimation, tensor cleanup, smoothing, bad posture alerts & PiP updates.
 */
async function stepInference(isBackground = false) {
  if (!state.isMonitoring || state.isPaused || !state.model || !state.webcam) return;

  const startTime = performance.now();

  try {
    // 1. Update webcam frame
    state.webcam.update();

    // 2. Pose estimation & classification with guaranteed intermediate tensor disposal
    let pose = null;
    let posenetOutput = null;
    let predictions = null;

    try {
      const estimation = await state.model.estimatePose(state.webcam.canvas);
      pose = estimation.pose;
      posenetOutput = estimation.posenetOutput;

      if (posenetOutput) {
        predictions = await state.model.predict(posenetOutput);
      }
    } finally {
      // Memory Leak Prevention: dispose intermediate posenetOutput Tensor
      if (posenetOutput && typeof posenetOutput.dispose === 'function') {
        posenetOutput.dispose();
      }
    }

    // 3. Render to canvas only when visible (skip completely in background)
    if (!isBackground && !document.hidden) {
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

    // 4. Process predictions with mode-adaptive smoothing window
    processPredictions(predictions, Boolean(pose), isBackground);

    // 5. Update Floating Mini Window (Document PiP) if open
    if (state.pipWindow && !state.pipWindow.closed) {
      updatePipUI();
    }

    // 6. Resilient background alert beeper: repeat beep every 1s while alarm is active
    if (state.isAlarmActive) {
      const now = Date.now();
      if (now - (state.lastAlarmBeepTime || 0) >= 1000) {
        playAlertBeep();
        state.lastAlarmBeepTime = now;
      }
    }

    // 7. Backend performance check: detect GPU / WebGL throttling in hidden tabs
    const stepDuration = performance.now() - startTime;
    if (isBackground && stepDuration > 280 && !state.fallbackAttempted) {
      state.consecutiveSlowInferences = (state.consecutiveSlowInferences || 0) + 1;
      if (state.consecutiveSlowInferences >= 5) {
        console.warn(`[PostureGuard] WebGL inference slowed down in background (${stepDuration.toFixed(1)}ms). Switching to CPU backend...`);
        try {
          if (window.tf && typeof tf.setBackend === 'function') {
            await tf.setBackend('cpu');
            state.fallbackAttempted = true;
            console.log('[PostureGuard] Successfully switched to CPU backend for background stability.');
          }
        } catch (backendErr) {
          console.warn('[PostureGuard] Backend fallback failed:', backendErr);
        }
      }
    } else if (!isBackground && state.fallbackAttempted) {
      // Restore WebGL when tab returns to foreground
      try {
        if (window.tf && typeof tf.setBackend === 'function') {
          await tf.setBackend('webgl');
          state.fallbackAttempted = false;
          state.consecutiveSlowInferences = 0;
          console.log('[PostureGuard] Restored WebGL backend for foreground session.');
        }
      } catch (e) {}
    }

    // 8. Update FPS indicator & periodic telemetry
    updateFPS(isBackground);
  } catch (err) {
    console.error('[PostureGuard] Inference step error:', err);
  }
}

/**
 * Manages dual-mode execution loop:
 * - When document.hidden is false: uses requestAnimationFrame capped at ~10 FPS.
 * - When document.hidden is true: uses Web Worker ticker at ~3.8 FPS (~260ms), immune to tab throttling.
 */
function startInferenceLoop() {
  stopInferenceLoop();
  if (!state.isMonitoring || state.isPaused) return;

  initTickerWorker();

  if (document.hidden) {
    // Tab is hidden: run background Web Worker loop at ~3.8 FPS (~260ms)
    state.isInBackground = true;
    console.log('[PostureGuard] Tab hidden: running Web Worker background ticker at ~3.8 FPS');
    if (state.tickerWorker) {
      state.tickerWorker.postMessage({ action: 'start', interval: BG_INTERVAL_MS });
    }
  } else {
    // Tab is visible: run requestAnimationFrame loop capped at ~10 FPS (every ~100ms)
    state.isInBackground = false;
    let lastForegroundTimestamp = 0;

    const loop = async (timestamp) => {
      if (!state.isMonitoring || state.isPaused || document.hidden) return;

      // Cap foreground inference to ~10 FPS (100 ms)
      if (timestamp - lastForegroundTimestamp >= 95) {
        lastForegroundTimestamp = timestamp;
        if (!state.isInferenceBusy) {
          state.isInferenceBusy = true;
          try {
            await stepInference(false);
          } finally {
            state.isInferenceBusy = false;
          }
        }
      }

      if (!document.hidden && state.isMonitoring && !state.isPaused) {
        state.animationFrameId = requestAnimationFrame(loop);
      }
    };

    state.animationFrameId = requestAnimationFrame(loop);
  }
}

/**
 * Stops both requestAnimationFrame and Web Worker ticker
 */
function stopInferenceLoop() {
  if (state.animationFrameId) {
    cancelAnimationFrame(state.animationFrameId);
    state.animationFrameId = null;
  }
  if (state.tickerWorker) {
    state.tickerWorker.postMessage({ action: 'stop' });
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
  console.log('[PostureGuard] Visibility changed: document.hidden =', document.hidden, 'at', new Date().toLocaleTimeString());

  if (!document.hidden) {
    requestWakeLock();
  }

  startInferenceLoop();
}

/**
 * Computes rolling average over last N frames to suppress prediction jitter.
 * Dynamically adapts window size: 10 frames in foreground (~1.0s), 5 frames in background (~1.3s).
 * Applies detection threshold (>= 80%).
 */
function processPredictions(predictions, hasPose, isBackground = false) {
  if (!hasPose || !predictions) {
    setDetectedPosture('No Person Detected', 0);
    return;
  }

  // Adaptive smoothing window: 10 frames in foreground (~10 FPS -> ~1.0s), 5 frames in background (~3.8 FPS -> ~1.3s)
  state.smoothingWindowSize = (isBackground || document.hidden) ? 5 : 10;

  // 1. Push raw frame prediction to smoothing queue
  const frameMap = {};
  predictions.forEach(p => {
    frameMap[p.className] = p.probability;
  });

  state.predictionHistory.push(frameMap);
  while (state.predictionHistory.length > state.smoothingWindowSize) {
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
 * If unsupported, hides button and shows informative desktop fallback note.
 */
function checkPipSupport() {
  const isSupported = ('documentPictureInPicture' in window);
  if (DOM.btnFloatingPip) {
    DOM.btnFloatingPip.style.display = isSupported ? 'inline-flex' : 'none';
  }
  if (DOM.pipUnsupportedNote) {
    DOM.pipUnsupportedNote.style.display = isSupported ? 'none' : 'inline-block';
    DOM.pipUnsupportedNote.textContent = isSupported
      ? ''
      : 'Mini window supported in Chrome/Edge. In Firefox/Safari, background audio & desktop notifications alert automatically.';
  }
}

/**
 * Opens always-on-top Document Picture-in-Picture window
 */
async function openPipWindow() {
  if (!('documentPictureInPicture' in window)) return;
  if (state.pipWindow && !state.pipWindow.closed) return;

  try {
    const pipWindow = await window.documentPictureInPicture.requestWindow({
      width: 340,
      height: 230
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
        .pip-timer.active-alert { color: #f87171; font-weight: bold; animation: pipPulse 1s infinite alternate; }
        @keyframes pipPulse { from { opacity: 0.7; } to { opacity: 1; } }
        .pip-footer { display: flex; gap: 8px; justify-content: center; }
        .btn-pip-pause { background: #2563eb; color: white; border: none; padding: 7px 16px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
        .btn-pip-pause:hover { background: #1d4ed8; }
        .btn-pip-stop { background: #ef4444; color: white; border: none; padding: 7px 16px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer; transition: background 0.15s; }
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
        <button id="pipPauseBtn" class="btn-pip-pause">${state.isPaused ? 'Resume' : 'Pause'}</button>
        <button id="pipStopBtn" class="btn-pip-stop">Stop</button>
      </div>
    `;

    pipWindow.document.getElementById('pipPauseBtn').addEventListener('click', () => {
      if (state.isPaused) {
        startMonitoring();
      } else {
        pauseMonitoring();
      }
      updatePipUI();
    });

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

  await openPipWindow();
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
  const pauseBtn = doc.getElementById('pipPauseBtn');

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
    indicatorEl.textContent = state.isPaused ? '⏸️ Paused' : (document.hidden ? '👁️ Background' : '🟢 Active');
  }

  if (pauseBtn) {
    pauseBtn.textContent = state.isPaused ? 'Resume' : 'Pause';
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
// Local Posture Streak Feature
// ============================================================================

// Easy-to-change constants (Rules: >= 30 min monitored AND >= 60% good posture per calendar day)
const STREAK_GOAL_MINUTES = 30; // Minimum monitored minutes per calendar day
const STREAK_MIN_GOOD_RATIO = 0.60; // Minimum 60% of monitored time was good posture
const STREAK_GOAL_SECONDS = STREAK_GOAL_MINUTES * 60; // 1800 seconds
const STREAK_STORAGE_KEY = 'postureguard_streak_data_v1';

const STREAK_MILESTONES = [
  { days: 3, id: 'badge-3', title: 'Bronze Spine', label: '3 Days', icon: '🥉', desc: '3-day consistency' },
  { days: 7, id: 'badge-7', title: 'Silver Alignment', label: '7 Days', icon: '🥈', desc: '1 week habit' },
  { days: 14, id: 'badge-14', title: 'Gold Discipline', label: '14 Days', icon: '🥇', desc: '2 continuous weeks' },
  { days: 30, id: 'badge-30', title: 'Ergonomic Master', label: '30 Days', icon: '👑', desc: '1 full month' }
];

let streakState = {
  lastActiveDate: null,       // YYYY-MM-DD of last qualified day
  currentStreak: 0,
  bestStreak: 0,
  freezesLeft: 1,             // 1 free missed day per week
  lastFreezeRefillWeek: null, // YYYY-Www
  lastFreezeDate: null,       // YYYY-MM-DD
  lastReminderDate: null,     // YYYY-MM-DD
  dailyTotals: {},            // { [dateStr]: { monitoredSeconds: 0, goodSeconds: 0, qualified: false } }
  activeDay: null,            // current local calendar date string
  saveCounter: 0
};

/**
 * Returns YYYY-MM-DD string in user's local timezone
 */
function getLocalDateString(d = new Date()) {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

/**
 * Returns calendar date shifted by offsetDays (e.g. -1 for yesterday)
 */
function getDaysOffset(dateStr, offsetDays) {
  if (!dateStr) return '';
  const [y, m, d] = dateStr.split('-').map(Number);
  const targetDate = new Date(y, m - 1, d + offsetDays);
  return getLocalDateString(targetDate);
}

/**
 * Returns ISO week identifier (e.g. "2026-W42") in user's local timezone
 */
function getWeekIdentifier(date = new Date()) {
  const d = new Date(date.getFullYear(), date.getMonth(), date.getDate());
  const dayNum = d.getDay() || 7;
  d.setDate(d.getDate() + 4 - dayNum);
  const yearStart = new Date(d.getFullYear(), 0, 1);
  const weekNo = Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
  return `${d.getFullYear()}-W${String(weekNo).padStart(2, '0')}`;
}

/**
 * Computes calendar day difference between two YYYY-MM-DD date strings.
 * Using UTC midnight dates ensures immunity to DST shifts (23h / 25h).
 */
function getDaysBetween(dateStrA, dateStrB) {
  if (!dateStrA || !dateStrB) return 0;
  const [y1, m1, d1] = dateStrA.split('-').map(Number);
  const [y2, m2, d2] = dateStrB.split('-').map(Number);
  const utcA = Date.UTC(y1, m1 - 1, d1);
  const utcB = Date.UTC(y2, m2 - 1, d2);
  const diffDays = Math.round((utcB - utcA) / (86400 * 1000));
  return Math.max(0, diffDays);
}

/**
 * Loads streak data from localStorage wrapped in try/catch
 */
function loadStreakData() {
  const todayStr = getLocalDateString();
  const currentWeek = getWeekIdentifier();
  streakState.activeDay = todayStr;

  try {
    const raw = localStorage.getItem(STREAK_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') {
        streakState.lastActiveDate = parsed.lastActiveDate || null;
        streakState.currentStreak = Number.isInteger(parsed.currentStreak) ? Math.max(0, parsed.currentStreak) : 0;
        streakState.bestStreak = Number.isInteger(parsed.bestStreak) ? Math.max(0, parsed.bestStreak) : 0;
        streakState.freezesLeft = Number.isInteger(parsed.freezesLeft) ? Math.max(0, parsed.freezesLeft) : 1;
        streakState.lastFreezeRefillWeek = parsed.lastFreezeRefillWeek || null;
        streakState.lastFreezeDate = parsed.lastFreezeDate || null;
        streakState.lastReminderDate = parsed.lastReminderDate || null;
        streakState.dailyTotals = (parsed.dailyTotals && typeof parsed.dailyTotals === 'object') ? parsed.dailyTotals : {};
      }
    }
  } catch (err) {
    console.warn('[PostureGuard Streak] Could not read streak from localStorage:', err);
  }

  // Refill freeze: 1 free missed day per week
  if (streakState.lastFreezeRefillWeek !== currentWeek) {
    streakState.freezesLeft = 1;
    streakState.lastFreezeRefillWeek = currentWeek;
  }

  // Evaluate streak status against today
  evaluateStreakLiveness(todayStr);

  // Initialize today's record if missing
  if (!streakState.dailyTotals[todayStr]) {
    streakState.dailyTotals[todayStr] = { monitoredSeconds: 0, goodSeconds: 0, qualified: false };
  }

  saveStreakData();
}

/**
 * Evaluates whether streak is still alive, frozen, or broken on load/rollover
 */
function evaluateStreakLiveness(todayStr) {
  if (!streakState.lastActiveDate || streakState.currentStreak <= 0) return;

  const gap = getDaysBetween(streakState.lastActiveDate, todayStr);
  if (gap <= 1) {
    // 0: Qualified today. 1: Qualified yesterday (or yesterday was protected by freeze). Streak is alive!
    return;
  } else if (gap === 2) {
    // Exactly 1 day was missed (yesterday was skipped).
    if (streakState.freezesLeft > 0) {
      // 1 freeze automatically protects the streak
      streakState.freezesLeft--;
      const yesterdayStr = getDaysOffset(todayStr, -1);
      streakState.lastActiveDate = yesterdayStr;
      streakState.lastFreezeDate = yesterdayStr;
      if (!streakState.dailyTotals[yesterdayStr]) {
        streakState.dailyTotals[yesterdayStr] = { monitoredSeconds: 0, goodSeconds: 0, qualified: false, frozen: true };
      } else {
        streakState.dailyTotals[yesterdayStr].frozen = true;
      }
      console.log('[PostureGuard Streak] 1 freeze automatically applied to keep streak alive.');
    } else {
      console.log('[PostureGuard Streak] Streak broken due to missed day with 0 freezes left.');
      streakState.currentStreak = 0;
    }
  } else {
    // 2 or more days were missed. Even with 1 freeze, streak cannot be bridged.
    console.log('[PostureGuard Streak] Streak broken due to multiple missed days.');
    streakState.currentStreak = 0;
  }
}

/**
 * Saves streak data to localStorage wrapped in try/catch
 */
function saveStreakData() {
  try {
    const dates = Object.keys(streakState.dailyTotals).sort();
    if (dates.length > 60) {
      const toDelete = dates.slice(0, dates.length - 60);
      toDelete.forEach(d => delete streakState.dailyTotals[d]);
    }

    const payload = {
      lastActiveDate: streakState.lastActiveDate,
      currentStreak: streakState.currentStreak,
      bestStreak: streakState.bestStreak,
      freezesLeft: streakState.freezesLeft,
      lastFreezeRefillWeek: streakState.lastFreezeRefillWeek,
      lastFreezeDate: streakState.lastFreezeDate,
      lastReminderDate: streakState.lastReminderDate,
      dailyTotals: streakState.dailyTotals
    };

    localStorage.setItem(STREAK_STORAGE_KEY, JSON.stringify(payload));
  } catch (err) {
    console.warn('[PostureGuard Streak] Could not save streak to localStorage:', err);
  }
}

/**
 * Records one second of active monitoring toward today's streak goal.
 * Seamlessly handles midnight rollover and sessions spanning two calendar days.
 */
function recordStreakSecond(isGood) {
  const now = new Date();
  const todayStr = getLocalDateString(now);

  // Handle midnight rollover edge case: session spanning into next calendar day
  if (streakState.activeDay && streakState.activeDay !== todayStr) {
    handleMidnightRollover(streakState.activeDay, todayStr);
    streakState.activeDay = todayStr;
  }

  if (!streakState.dailyTotals[todayStr]) {
    streakState.dailyTotals[todayStr] = { monitoredSeconds: 0, goodSeconds: 0, qualified: false };
  }

  const todayData = streakState.dailyTotals[todayStr];
  todayData.monitoredSeconds++;
  if (isGood) {
    todayData.goodSeconds++;
  }

  // Check if today qualifies
  if (!todayData.qualified) {
    const isDurationMet = todayData.monitoredSeconds >= STREAK_GOAL_SECONDS;
    const currentRatio = todayData.monitoredSeconds > 0 ? (todayData.goodSeconds / todayData.monitoredSeconds) : 0;
    const isRatioMet = currentRatio >= STREAK_MIN_GOOD_RATIO;

    if (isDurationMet && isRatioMet) {
      todayData.qualified = true;
      applyDayQualification(todayStr);
    }
  }

  // Check if evening reminder notification should fire (after 8 PM if allowed and not qualified)
  checkStreakReminderNotification(now, todayStr);

  // Periodically save to localStorage (every 5 seconds) to minimize disk writes
  streakState.saveCounter = (streakState.saveCounter + 1) % 5;
  if (streakState.saveCounter === 0) {
    saveStreakData();
  }

  // Update streak UI
  updateStreakUI();
}

/**
 * Handles seamless day transition across midnight during an ongoing session
 */
function handleMidnightRollover(prevDayStr, newDayStr) {
  console.log(`[PostureGuard Streak] Midnight rollover detected: ${prevDayStr} -> ${newDayStr}`);

  const newWeek = getWeekIdentifier();
  if (streakState.lastFreezeRefillWeek !== newWeek) {
    streakState.freezesLeft = 1;
    streakState.lastFreezeRefillWeek = newWeek;
  }

  const prevData = streakState.dailyTotals[prevDayStr];
  if (streakState.currentStreak > 0 && (!prevData || !prevData.qualified)) {
    if (streakState.freezesLeft > 0) {
      streakState.freezesLeft--;
      streakState.lastActiveDate = prevDayStr;
      streakState.lastFreezeDate = prevDayStr;
      if (!streakState.dailyTotals[prevDayStr]) {
        streakState.dailyTotals[prevDayStr] = { monitoredSeconds: 0, goodSeconds: 0, qualified: false, frozen: true };
      } else {
        streakState.dailyTotals[prevDayStr].frozen = true;
      }
      console.log('[PostureGuard Streak] Freeze applied to protect streak across midnight rollover.');
    } else {
      streakState.currentStreak = 0;
      console.log('[PostureGuard Streak] Streak broken at midnight (previous day incomplete with 0 freezes).');
    }
  }

  if (!streakState.dailyTotals[newDayStr]) {
    streakState.dailyTotals[newDayStr] = { monitoredSeconds: 0, goodSeconds: 0, qualified: false };
  }

  saveStreakData();
  updateStreakUI();
}

/**
 * Applies streak increment when a calendar day's goal is met
 */
function applyDayQualification(todayStr) {
  console.log(`[PostureGuard Streak] Daily goal achieved for ${todayStr}!`);

  if (!streakState.lastActiveDate) {
    streakState.currentStreak = 1;
  } else {
    const gap = getDaysBetween(streakState.lastActiveDate, todayStr);
    if (gap === 0) {
      // Already qualified today
    } else if (gap === 1) {
      streakState.currentStreak++;
    } else {
      streakState.currentStreak = 1;
    }
  }

  streakState.lastActiveDate = todayStr;
  if (streakState.currentStreak > streakState.bestStreak) {
    streakState.bestStreak = streakState.currentStreak;
  }

  saveStreakData();
  updateStreakUI();
}

/**
 * Sends notification when streak is about to end (after 8 PM, if notifications allowed)
 */
function checkStreakReminderNotification(now, todayStr) {
  if (!('Notification' in window) || Notification.permission !== 'granted') return;
  if (streakState.currentStreak <= 0) return;

  const todayData = streakState.dailyTotals[todayStr];
  if (todayData && todayData.qualified) return;

  if (now.getHours() >= 20) {
    if (streakState.lastReminderDate !== todayStr) {
      try {
        new Notification('PostureGuard — Streak at Risk! 🔥', {
          body: `Your ${streakState.currentStreak}-day posture streak will end at midnight! Complete today's 30-min goal to keep it alive.`,
          icon: 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="%23f59e0b"><path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/></svg>'
        });
        streakState.lastReminderDate = todayStr;
        saveStreakData();
      } catch (err) {
        console.warn('[PostureGuard Streak] Reminder notification error:', err);
      }
    }
  }
}

/**
 * Resets all streak data upon user confirmation
 */
function resetStreakData() {
  try {
    localStorage.removeItem(STREAK_STORAGE_KEY);
  } catch (err) {
    console.warn('[PostureGuard Streak] Could not clear localStorage:', err);
  }

  const todayStr = getLocalDateString();
  streakState = {
    lastActiveDate: null,
    currentStreak: 0,
    bestStreak: 0,
    freezesLeft: 1,
    lastFreezeRefillWeek: getWeekIdentifier(),
    lastFreezeDate: null,
    lastReminderDate: null,
    dailyTotals: {
      [todayStr]: { monitoredSeconds: 0, goodSeconds: 0, qualified: false }
    },
    activeDay: todayStr,
    saveCounter: 0
  };

  saveStreakData();
  updateStreakUI();
  console.log('[PostureGuard Streak] Streak data successfully reset.');
}

/**
 * Updates streak card UI elements
 */
function updateStreakUI() {
  if (!DOM.streakCard) return;

  const todayStr = getLocalDateString();
  const todayData = streakState.dailyTotals[todayStr] || { monitoredSeconds: 0, goodSeconds: 0, qualified: false };

  // 1. Current and Best Streak
  if (DOM.streakCurrentVal) DOM.streakCurrentVal.textContent = streakState.currentStreak;
  if (DOM.streakBestVal) DOM.streakBestVal.textContent = `${streakState.bestStreak} ${streakState.bestStreak === 1 ? 'day' : 'days'}`;

  // Flame Icon Activity Glow
  if (DOM.streakFlameIcon) {
    if (streakState.currentStreak > 0 || todayData.qualified) {
      DOM.streakFlameIcon.classList.add('streak-active');
    } else {
      DOM.streakFlameIcon.classList.remove('streak-active');
    }
  }

  // 2. Freezes left
  if (DOM.streakFreezesVal) {
    if (streakState.freezesLeft > 0) {
      DOM.streakFreezesVal.textContent = `${streakState.freezesLeft} Freeze Available`;
      if (DOM.streakFreezePill) DOM.streakFreezePill.classList.remove('freeze-used');
    } else {
      DOM.streakFreezesVal.textContent = '0 Freezes (Refills Weekly)';
      if (DOM.streakFreezePill) DOM.streakFreezePill.classList.add('freeze-used');
    }
  }

  // 3. Today's progress
  const monitoredSec = todayData.monitoredSeconds;
  const goodSec = todayData.goodSeconds;
  const monitoredMin = (monitoredSec / 60).toFixed(1);
  const goalMin = STREAK_GOAL_MINUTES;

  const progressPct = Math.min(100, Math.round((monitoredSec / STREAK_GOAL_SECONDS) * 100));
  const goodRatio = monitoredSec > 0 ? Math.round((goodSec / monitoredSec) * 100) : 0;

  if (DOM.streakTodayProgressVal) {
    DOM.streakTodayProgressVal.textContent = `${monitoredMin} / ${goalMin} min (${progressPct}%)`;
  }

  if (DOM.streakTodayProgressBar) {
    DOM.streakTodayProgressBar.style.width = `${progressPct}%`;
    if (todayData.qualified) {
      DOM.streakTodayProgressBar.classList.add('bar-qualified');
    } else {
      DOM.streakTodayProgressBar.classList.remove('bar-qualified');
    }
  }

  if (DOM.streakGoodRatioText) {
    DOM.streakGoodRatioText.textContent = `Good Posture: ${goodRatio}% (${Math.round(STREAK_MIN_GOOD_RATIO * 100)}% needed)`;
  }

  if (DOM.streakTodayStatusBadge) {
    if (todayData.qualified) {
      DOM.streakTodayStatusBadge.textContent = 'Goal Achieved! 🔥';
      DOM.streakTodayStatusBadge.className = 'streak-goal-badge goal-achieved';
    } else if (monitoredSec >= STREAK_GOAL_SECONDS && goodRatio < Math.round(STREAK_MIN_GOOD_RATIO * 100)) {
      DOM.streakTodayStatusBadge.textContent = 'Needs 60% Good Posture';
      DOM.streakTodayStatusBadge.className = 'streak-goal-badge goal-needs-ratio';
    } else if (monitoredSec > 0) {
      DOM.streakTodayStatusBadge.textContent = 'In Progress';
      DOM.streakTodayStatusBadge.className = 'streak-goal-badge goal-active';
    } else {
      DOM.streakTodayStatusBadge.textContent = 'Pending';
      DOM.streakTodayStatusBadge.className = 'streak-goal-badge goal-pending';
    }
  }

  if (DOM.streakStatusText) {
    if (todayData.qualified) {
      DOM.streakStatusText.textContent = 'Streak secured for today! Next goal opens tomorrow.';
    } else if (streakState.currentStreak > 0) {
      DOM.streakStatusText.textContent = `Complete ${goalMin} min (${Math.round(STREAK_MIN_GOOD_RATIO * 100)}% good) to keep your streak alive!`;
    } else {
      DOM.streakStatusText.textContent = `Goal: ${goalMin} min monitored & \u2265${Math.round(STREAK_MIN_GOOD_RATIO * 100)}% good posture`;
    }
  }

  // 4. Milestone Badges Rendering
  renderStreakMilestonesUI();
}

/**
 * Renders milestone cards at 3, 7, 14, and 30 days
 */
function renderStreakMilestonesUI() {
  if (!DOM.streakMilestonesList) return;

  DOM.streakMilestonesList.innerHTML = '';
  const effectiveStreak = Math.max(streakState.currentStreak, streakState.bestStreak);

  STREAK_MILESTONES.forEach(m => {
    const isEarned = effectiveStreak >= m.days;
    const card = document.createElement('div');
    card.className = `milestone-badge-card ${isEarned ? 'is-earned' : 'is-locked'}`;

    const progressText = isEarned
      ? 'Achieved 🔥'
      : `${Math.min(effectiveStreak, m.days)} / ${m.days}d`;

    card.innerHTML = `
      <div class="milestone-icon" aria-hidden="true">${m.icon}</div>
      <div class="milestone-name">${m.title}</div>
      <div class="milestone-days">${m.label}</div>
      <span class="milestone-status-chip ${isEarned ? 'chip-unlocked' : 'chip-locked'}">
        ${progressText}
      </span>
    `;

    DOM.streakMilestonesList.appendChild(card);
  });
}

/**
 * Initializes streak system on startup
 */
function initStreakSystem() {
  loadStreakData();
  updateStreakUI();

  if (DOM.btnResetStreak) {
    DOM.btnResetStreak.addEventListener('click', () => {
      if (confirm('Reset your posture streak data and daily goal progress? This cannot be undone.')) {
        resetStreakData();
      }
    });
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

    // Record second for local streak engine
    recordStreakSecond(isGoodPosture(current));

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

  // Ensure streak state is saved and streak dashboard UI is fully refreshed
  saveStreakData();
  updateStreakUI();

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
  // 1. Resume AudioContext & start silent Web Audio keep-alive track
  resumeAudioContext();
  startSilentAudio();

  // 2. Request Screen Wake Lock to avoid tab suspension
  requestWakeLock();

  // 3. Request desktop Notification permission upfront when user clicks Start
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

  // 4. Offer/Auto-open Document Picture-in-Picture floating mini widget
  if ('documentPictureInPicture' in window && (!state.pipWindow || state.pipWindow.closed)) {
    try {
      await openPipWindow();
    } catch (pipErr) {
      console.warn('[PostureGuard] Auto PiP launch note:', pipErr);
    }
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
  releaseWakeLock();
  stopSilentAudio();
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

function updateFPS(isBackground = false) {
  state.frameCount++;
  const now = performance.now();
  if (now - state.lastFrameTime >= 1000) {
    state.currentFps = state.frameCount;
    DOM.fpsDisplay.textContent = `${state.currentFps} FPS`;
    state.frameCount = 0;
    state.lastFrameTime = now;
  }

  // Periodic telemetry logging every ~10s in console for background validation & test observability
  if (now - state.lastTelemetryLogTime >= 10000) {
    state.lastTelemetryLogTime = now;
    const numTensors = (window.tf && typeof tf.memory === 'function') ? tf.memory().numTensors : 'N/A';
    console.log(
      `[PostureGuard Telemetry] Mode: ${isBackground || document.hidden ? 'BACKGROUND (Worker)' : 'FOREGROUND (rAF)'} | ` +
      `FPS: ~${state.currentFps} | Active Tensors: ${numTensors} | ` +
      `Posture: "${state.currentActivePosture}" | Time: ${new Date().toLocaleTimeString()}`
    );
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

  // Initialize local streak habit engine
  initStreakSystem();
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  attachEventListeners();
  console.log('[PostureGuard] Application ready with background monitoring & Document PiP support.');
});
