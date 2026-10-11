# PostureGuard — 2-Minute Demo Video & Viva Presentation Script

> **Competition:** *BUILD WITH ML*, University of Lucknow  
> **Project:** PostureGuard — Real-Time Sitting-Posture Coach  
> **Total Target Duration:** 2 Minutes (120 seconds)  
> **Presenter Tone:** Confident, articulate, engineering-focused, and student-centric.

---

## Video Timeline Breakdown

| Time Window | Visual Action (What to Show on Screen) | Spoken Narration (What to Say) |
| :--- | :--- | :--- |
| **0:00 – 0:20** *(20s)*<br>**The Hook & Problem** | • Start on the PostureGuard Landing Hero.<br>• Highlight the *BUILD WITH ML* competition badge.<br>• Cursor hovers over the 100% Client-Side Privacy banner. | *"Respected judges and faculty, welcome. Students and developers sit for over 9 hours a day, leading to severe cervical and spinal strain. But existing solutions require expensive wearables or stream private bedroom camera feeds to the cloud. Today, we present **PostureGuard**—a real-time, privacy-preserving sitting-posture coach built entirely with client-side machine learning."* |
| **0:20 – 0:45** *(25s)*<br>**Zero-Cloud Architecture** | • Click **"Start Posture Check"**.<br>• Screen smoothly scrolls to the Live Monitor.<br>• Camera initializes, rendering the mirrored feed and PoseNet skeleton overlay. | *"PostureGuard runs 100% locally in your browser. Using TensorFlow.js and Google Teachable Machine Pose, it tracks 17 upper-body skeletal landmarks at 30 frames per second inside the browser’s WebGL context. Not a single video frame or biometric coordinate ever touches an external server."* |
| **0:45 – 1:15** *(30s)*<br>**Live Posture Classification** | • Sit upright: badge shows **Good Posture** (Emerald Green, 95% confidence).<br>• Lean neck forward toward monitor: watch badge shift to **Neck Forward** (Rose Red).<br>• Point out the dynamic 10-frame smoothed probability bars.<br>• Read the dynamic Ergonomic Tip text. | *"Notice how the model dynamically classifies posture across 4 states: Good Posture, Slouching, Neck Forward, and Leaning Sideways. We implemented a 10-frame moving-average filter and an 80% confidence threshold to eliminate jitter. As I lean forward into a typical study slouch, the coach tip instantly instructs: 'Pull your chin back and align your ears over your shoulders.'"* |
| **1:15 – 1:35** *(20s)*<br>**Smart Alert Engine** | • Stay in a bad posture for the set delay threshold.<br>• Show the countdown pill: *"Warning in 5s... 4s..."*<br>• Alert fires: synthesized chime plays, screen banner pulses red. | *"PostureGuard doesn't nag you for momentary adjustments. But if bad posture persists continuously for 25 seconds, the smart alert engine triggers. Listen to this—a gentle two-tone chime synthesized live using the Web Audio API without requiring any external audio files, accompanied by an on-screen alert banner."* |
| **1:35 – 1:50** *(15s)*<br>**Telemetry & Summary Card** | • Click **"Stop & Summary"**.<br>• Session Summary modal opens.<br>• Highlight Posture Score (e.g., 88/100), time breakdown, and customized advice. | *"When your study block ends, clicking 'Stop & Summary' delivers your personal Ergonomic Scorecard out of 100, a complete time distribution breakdown, and tailored suggestions to prevent chronic musculoskeletal fatigue."* |
| **1:50 – 2:00** *(10s)*<br>**Conclusion & Deployment** | • Show the live dashboard distribution bar and footer links.<br>• Conclude with viva-ready statement. | *"Built with pure vanilla HTML, CSS, and JavaScript, PostureGuard requires zero build tools and deploys instantaneously to Vercel as a static site. Thank you for your time, and we look forward to your questions!"* |

---

## Viva Q&A Quick Reference Guide

Be prepared to answer these common viva questions during your defense:

### Q1: Why use Teachable Machine Pose instead of raw PoseNet?
> **Answer:** *"Teachable Machine allows custom transfer learning on top of PoseNet's MobileNet backbone. While raw PoseNet outputs raw 2D landmark coordinates, our Teachable Machine model trains a specialized classification head on our custom dataset of student sitting postures (Good Posture, Slouching, Neck Forward, Leaning Sideways)."*

### Q2: How did you solve frame flicker and false positives?
> **Answer:** *"We implemented a 10-frame sliding window moving average filter. We average class probabilities across the last 10 frames and enforce a strict 80% threshold before switching the active state. Furthermore, alerts only trigger after continuous bad posture exceeding our configured delay (e.g. 25 seconds)."*

### Q3: Why is this better for student privacy?
> **Answer:** *"All inference runs locally on the client's GPU via TensorFlow.js WebGL backend. The webcam stream never leaves `navigator.mediaDevices`. There is zero server backend, zero network payload for images, and zero data storage."*

### Q4: Why is there no backend or build step?
> **Answer:** *"By keeping the architecture strictly vanilla HTML, CSS, and JavaScript with CDN dependencies, the entire app is lightweight (< 100 KB initial HTML/CSS/JS payload), requires zero node build steps, and can be hosted instantly on Vercel or any static edge CDN."*
