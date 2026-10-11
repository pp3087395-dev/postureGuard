# PostureGuard — Model Directory

Place your Google Teachable Machine exported pose model files here.

## Required Files:
1. `model.json` — Model architecture and graph definitions.
2. `metadata.json` — Class names and model metadata.
3. `weights.bin` — Quantized neural network weight parameters.

---

## How to Export from Google Teachable Machine:

1. Open your project on [Teachable Machine Pose](https://teachablemachine.withgoogle.com/).
2. Train your model with the 4 posture classes:
   - `Good Posture`
   - `Slouching`
   - `Neck Forward`
   - `Leaning Sideways`
3. Click **"Export Model"**.
4. In the export dialog:
   - Select the **TensorFlow.js** tab.
   - Choose **Download** (Download my model).
5. Extract the downloaded `.zip` archive.
6. Copy the three files (`model.json`, `metadata.json`, and `weights.bin`) into this `/model` directory:
   ```
   /model/
     ├── model.json
     ├── metadata.json
     ├── weights.bin
     └── README.md
   ```
7. Once placed, reload PostureGuard in your browser. The application loads `./model/model.json` and `./model/metadata.json` automatically!
