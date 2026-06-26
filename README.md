# 🚗 RL Car — Obstacle-Avoidance AI Car with Q-Learning

Train a reinforcement learning agent in a desktop simulator, then export the
learned policy straight to an **Arduino Uno** and watch a real robot car
avoid obstacles using the brain it taught itself.

![Simulator GUI](assets/simulator-screenshot.png)

> The simulator above shows a car learning to navigate a circular track with
> a center obstacle. After ~140 episodes of training, the agent already
> completes full laps consistently (see the **Laps Completed** chart).

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [GUI Walkthrough](#gui-walkthrough)
- [Getting Started](#getting-started)
- [Training Workflow](#training-workflow)
- [Reading the Training Graphs](#reading-the-training-graphs)
- [Exporting to Arduino](#exporting-to-arduino)
- [Hardware & Wiring](#hardware--wiring)
- [Flashing & Testing the Real Car](#flashing--testing-the-real-car)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project teaches a small car to avoid obstacles **without writing a
single `if-else` steering rule by hand**. Instead, a Q-learning agent learns
the steering policy through trial and error inside a 2D simulator, and the
final policy table is exported as ready-to-flash Arduino C++ code.

**Pipeline:**

```
Simulate & train  →  Watch the policy converge  →  Export Arduino code  →  Flash real car  →  Test on a real obstacle course
```

No deep learning framework, no GPU, no internet connection required — the
entire "brain" is a lookup table small enough to fit in an Arduino Uno's
flash memory.

---

## How It Works

The car has **3 ultrasonic sensors** (left −45°, center 0°, right +45°) and
**5 possible actions** (forward, turn left, turn right, sharp left, sharp
right). Each sensor reading is discretized into 5 distance bins (0 = very
close, 4 = far), giving:

```
5 (left bins) × 5 (center bins) × 5 (right bins) = 125 possible states
125 states × 5 actions = 625 Q-values total
```

During training the agent:

1. **Observes** the current state (its 3 sensor readings).
2. **Chooses an action** — either randomly (exploration) or the
   highest-scoring action from its Q-table (exploitation), balanced by an
   epsilon (ε) value that decays over time.
3. **Receives a reward**: large negative reward on collision, positive
   reward for passing checkpoints/completing a lap, small shaping rewards
   for moving forward and staying centered.
4. **Updates its Q-table** using the Q-learning update rule, slightly
   nudging the value toward what it just experienced.

Repeated over hundreds of episodes, the 625-number table converges into a
working obstacle-avoidance policy — which is exactly what gets exported to
the Arduino.

---

## GUI Walkthrough

The simulator is a single-window desktop app split into three regions:

### 1. Simulation Panel (top-left)

A live 2D view of the track: a square loop, a configurable center obstacle,
checkpoint markers (①–④), and the car with its three sensor rays drawn in
green/yellow. Live readouts show:

- **센서 (좌/중/우)** — current Left / Center / Right sensor distances (cm)
- **상태** — the discretized state tuple, e.g. `(1, 2, 3)`
- **스텝 / 완주** — current step count and laps completed this episode

### 2. Environment & Training Settings (left sidebar)

| Section | Setting | Description |
|---|---|---|
| **환경 설정** (Environment) | 장애물 크기 (Obstacle size) | Size of the center obstacle in cm |
| | 도로 폭 (Road width) | Width of the driving lane in cm |
| **학습 파라미터** (Hyperparameters) | 학습 횟수 (Episodes) | Total training episodes to run |
| | 학습률 α (Learning rate) | How much each update shifts the Q-value |
| | 할인율 γ (Discount factor) | How much future reward matters vs. immediate reward |
| | ε 감소율 (Epsilon decay) | How fast exploration shifts to exploitation |
| | 시각화 (Visualization) | Toggle live rendering on/off (off = faster training) |

Action buttons below let you **start / pause / resume / stop** training, run
a **policy test** with the current Q-table, **reset** the environment, **save /
load** a trained Q-table, and — most importantly — **export Arduino code**.

### 3. Graphs & Tables (right panel)

Three tabs:

- **학습 그래프 (Learning Graphs)** — 4 live charts described below
- **정책 테이블 (Policy Table)** — the raw 125×5 Q-table, human-readable
- **학습 로그 (Training Log)** — per-episode text log (reward, steps, result)

---

## Getting Started

### Requirements

- Python 3.9+
- PyQt5
- numpy

```bash
pip install PyQt5 numpy
```

### Run the simulator

```bash
python rl_car_simulator.py
```

> Replace `rl_car_simulator.py` with your actual entry-point filename.

---

## Training Workflow

1. **Set the environment** — pick an obstacle size and road width. Smaller
   road widths and larger obstacles make the task harder and may need more
   episodes to converge.
2. **Set hyperparameters** — the defaults shown in the screenshot
   (`episodes=300`, `α=0.15`, `γ=0.95`, `ε decay=0.990`) are a solid
   starting point for a 80×80cm track.
3. **Click 학습 시작 (Start Training)**. Turn visualization **off** if you
   want to train faster; turn it **on** to watch the car's behavior evolve
   in real time.
4. **Pause / resume** anytime — training state is preserved.
5. Once the progress bar reaches 100%, **run 테스트 실행 (Test Run)** to
   evaluate the final policy with ε = 0 (no randomness).
6. Happy with the result? **저장 (Save)** the Q-table for later, then move
   on to exporting Arduino code.

---

## Reading the Training Graphs

| Chart | What it shows | What "good" looks like |
|---|---|---|
| **Episode Reward** | Raw reward per episode (thin line) plus a moving average (thick line) | Starts near 0 / negative, then climbs and gets less noisy over time |
| **Laps Completed** | Bars = laps that episode, line = cumulative laps | Bars should appear more frequently and grow taller as training progresses |
| **Epsilon (Exploration)** | The ε value decaying from 1.0 toward ~0 | A smooth downward curve — the agent explores less and exploits more |
| **Cumulative Reward** | Running total of all rewards earned so far | A "V" shape is normal: it dips early (lots of collisions) then recovers and climbs once the policy improves |

If your **Cumulative Reward** curve never turns upward, or **Laps
Completed** stays at zero past 100+ episodes, see
[Troubleshooting](#troubleshooting).

---

## Exporting to Arduino

Once training is complete (or even mid-training, if the policy already
looks decent in the test run), click:

```
아두이노 코드 내보내기  →  Export Arduino Code
```

This converts the 625-value Q-table into a compact `.ino` sketch. Internally,
it picks the **best action per state** and stores it as a flat
`PROGMEM` lookup table:

```cpp
const uint8_t POLICY[125] PROGMEM = {
  0, 2, 4, 1, 3, /* ... 125 values total ... */
};

int getAction(int left, int center, int right) {
  int idx = left * 25 + center * 5 + right;
  return pgm_read_byte(&POLICY[idx]);
}
```

At runtime, the real car does **no learning** — it just reads its 3 sensors,
discretizes them into bins, looks up the best action, and drives. This is
why the policy fits comfortably on an Arduino Uno.

---

## Hardware & Wiring

| # | Part | Qty | Role |
|---|---|---|---|
| 1 | Arduino Uno | 1 | Runs the exported policy code |
| 2 | HC-SR04 Ultrasonic Sensor | 3 | Left / Center / Right distance sensing |
| 3 | L298N Motor Driver | 1 | Speed & direction control for both motors |
| 4 | DC Gear Motor + Wheel | 2 | Differential drive |
| 5 | Caster ball | 1 | 3-point balance support |
| 6 | Robot chassis | 1 | Mounting frame |
| 7 | Battery pack (2× 18650 or 9V) | 1 | Power supply |
| 8 | Breadboard + jumper wires | — | Sensor wiring |
| 9 | Power switch | 1 | On/off control |

### Pin Mapping

| Component | Signal | Arduino Pin |
|---|---|---|
| Left sensor | TRIG / ECHO | D2 / D3 |
| Center sensor | TRIG / ECHO | D4 / D5 |
| Right sensor | TRIG / ECHO | D6 / D7 |
| L298N (left motor) | ENA / IN1 / IN2 | D9 / D8 / D10 |
| L298N (right motor) | ENB / IN3 / IN4 | D11 / D12 / D13 |

> ⚠️ **Sensor orientation matters.** If left/right sensors are swapped, the
> car will steer *away* from open space instead of toward it. Double-check
> wiring against this table before powering on.

> ⚠️ **Common ground required.** Arduino `GND` and L298N `GND` must be tied
> together, or motor signals will behave erratically.

---

## Flashing & Testing the Real Car

1. Open the exported `.ino` file in **Arduino IDE**.
2. **Tools → Board** → Arduino Uno, **Tools → Port** → select your USB port.
3. Click **Upload**. Wait for `Done uploading`.
4. Open **Serial Monitor** (`Ctrl+Shift+M`) at **9600 baud** to verify sensor
   readings update correctly as you move your hand near each sensor:

   ```
   D:32/41/28 S:3,4,2 A:0
   ```
   `D` = raw distances (cm) · `S` = discretized state · `A` = chosen action

5. Build a test track matching your simulator settings (e.g. obstacle 20cm +
   road width 30cm → an 80×80cm square loop with checkpoints at each corner).
6. Run the car. **Success criteria:**
   - Completes at least one full lap without collision
   - Smooth turns at corners (no abrupt stop-and-spin)
   - Roughly even left/right wall clearance

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| Car spins in one direction only | Left/right sensor wiring swapped | Re-check TRIG/ECHO pin mapping |
| Drives straight into walls | Sensor mounting angle off | Re-confirm −45° / 0° / +45° angles |
| Doesn't move at all | Motor wiring / power issue | Check battery polarity & L298N connections |
| One wheel doesn't turn | Broken wire / wrong pin | Re-check ENA/ENB pin mapping |
| Spins in place repeatedly | Old policy still flashed | Re-export & re-upload the latest `.ino` |
| Erratic stop/start behavior | Low battery voltage | Charge or replace battery |
| No data in Serial Monitor | Baud rate mismatch | Set Serial Monitor to 9600 baud |
| Training never improves (Laps stay at 0) | Track too hard for given hyperparameters | Lower obstacle size / increase road width, or train for more episodes |

---

## Project Structure

```
.
├── rl_car_simulator.py       # Main PyQt5 simulator application
├── assets/
│   └── simulator-screenshot.png
├── exports/                  # Arduino .ino files exported from the GUI
├── saved_models/             # Saved Q-tables (.json / .npy)
└── README.md
```

> Adjust this section to match your actual repository layout.

---

## Acknowledgments

Built as part of a hands-on AI + robotics curriculum at
**Pumpkin Idea Factory Makerspace (호박공장 메이커스페이스)**, a student-led
maker education space in Busan, South Korea.

---

## License

Specify your license here (e.g. MIT, Apache-2.0) or remove this section if
not yet decided.<img width="1915" height="1012" alt="simulator-screenshot" src="https://github.com/user-attachments/assets/b5521e83-aa98-4b93-8255-d9b7ac7757f2" />
