# Technical Spec Sheet
## Webcam-Based Finger-Tapping Measurement for Home Hand-Rehabilitation Research


## 1. Problem and Scope

Finger tapping — repeatedly bringing the thumb and index finger together and apart as fully and quickly as instructed — is a standard way to observe hand motor performance. Slowing, amplitude decrement, and growing irregularity across a tapping sequence are characteristic signs of bradykinesia in conditions such as Parkinson's disease and of motor impairment after stroke [1][2][4][5]. In routine practice the movement is rated by eye on a coarse ordinal scale (for example MDS-UPDRS item 3.4, scored 0–4), which is subjective, rater-dependent, and poorly suited to frequent or unsupervised measurement at home [6]. Prior work shows that instrumented and camera-based methods — smartphones, conventional webcams, and contactless vision systems — can convert this movement into reproducible numbers, which motivates a low-cost, webcam-only approach usable in the home [1][3][7].

This project uses a standard RGB webcam to report tap count, average frequency, inter-tap intervals (ITI), ITI coefficient of variation, normalized movement amplitude, and early-versus-late within-session performance changes.

The system measures **visible 2-D motion only**. It will not diagnose disease, estimate an MDS-UPDRS score, identify fatigue, or support clinical decisions. Early-versus-late results are descriptive **within-session performance changes**, not clinical findings.

### 1.1 Success criteria

The prototype should:

1. process 640×480 webcam video at at least 15 FPS on the demonstration computer;
2. obtain a usable normalized-distance signal while one hand is clearly visible;
3. detect complete tap cycles without counting gaps as taps;
4. agree with manually annotated events within documented tolerances;
5. flag low-quality recordings; and
6. provide a simple live view and exportable results.

### 1.2 Scope strategy

The default runtime pipeline is intentionally small:

`video → MediaPipe landmarks → normalized distance → causal smoothing → tap state machine → metrics/export`

Camera calibration, image enhancement, HSV segmentation, morphology, edges, optical flow, CamShift, and Kalman filtering are **experiments or optional extensions**. They enter the runtime pipeline only if evaluation demonstrates a meaningful benefit.

---

## 2. Course Alignment

| Course concept | Use in this project |
|---|---|
| Lab 1: image/video I/O | Timestamped webcam/video input, resize, and colour conversion |
| Labs 2–3: enhancement/filtering | Small offline ablation; one causal temporal filter at runtime |
| Lab 4: segmentation/morphology | HSV hand-mask experiment compared with full-frame detection |
| Lab 5: features/geometry | Landmark distance and hand-size normalization |
| Lab 8: camera calibration | Optional center-versus-edge distortion experiment |
| Lab 9: tracking | Landmark tracking evaluation; one fallback tested only if needed |
| Evaluation | Event accuracy, count/timing error, runtime, and robustness |

This demonstrates the required techniques without forcing every laboratory method into the production path.

---

## 3. Technology

| Component | Choice | Purpose |
|---|---|---|
| Language | Python 3.10+ | Implementation |
| Video/classical CV | OpenCV [9] | Capture, playback, resize, overlays, optional experiments |
| Hand landmarks | One selected MediaPipe API [8] | Single-hand landmark detection/tracking |
| Numerics | NumPy | Geometry and time-series calculations |
| Signal processing | SciPy or NumPy | Causal smoothing and event processing |
| Logging | Standard CSV/JSON or pandas | Frame and session output |
| Plots | Matplotlib | Signal plots |
| UI | Tkinter or PyQt5; choose one | Minimal desktop interface |

Select either MediaPipe Solutions or Tasks and document only the confidence fields its API actually returns.

**Required landmarks:** thumb tip 4, index tip 8, wrist 0, and middle MCP 9. Landmarks 5 and 17 may be evaluated as an alternate reference.

---

## 4. Requirements

| ID | Requirement |
|---|---|
| FR-1 | Read webcam or prerecorded video and assign monotonic timestamps. |
| FR-2 | Detect one hand and extract the required landmarks. |
| FR-3 | Compute thumb–index distance normalized by a hand reference length. |
| FR-4 | Flag frames with missing landmarks, implausible jumps, or unstable reference length. |
| FR-5 | Smooth valid samples with one causal filter. |
| FR-6 | Detect complete `OPEN → CLOSED → OPEN` cycles with hysteresis and duration constraints. |
| FR-7 | Compute count, frequency, amplitude, ITIs, ITI CV, and early/late changes. |
| FR-8 | Show video, signal, count, FPS, and recording quality in one interface. |
| FR-9 | Export frame data, detected events, configuration, and session summary. |
| FR-10 | Evaluate event accuracy, timing/count error, runtime, and robustness. |
| FR-11 | Compare selected course methods through controlled ablations. |

### 4.1 Non-functional requirements

- **Performance:** at least 15 FPS at 640×480; target 30 FPS.
- **Reproducibility:** save timestamps, parameters, API choice, and thresholds.
- **Auditability:** retain validity flags and event times.
- **Safety:** predicted or interpolated samples must not independently create taps.
- **Privacy:** use subject codes, obtain consent, and avoid retaining video when derived data is sufficient.

---

## 5. Core Pipeline

The runtime path is the five-stage flow in §1.2. Each stage below gives what it does and the reasoning that constrains it; §5.6 traces one illustrative session end to end.

### 5.1 Acquisition and landmarks

Use `cv2.VideoCapture` for webcam or file input [9]. Timestamp live frames with `time.perf_counter()` and **do not assume constant FPS** — every timing metric (frequency, ITI, ITI CV) is computed from real inter-frame times, so a nominal "30 FPS" assumption would bias them whenever the camera jitters. Resize to 640×480 for consistent geometry and speed, and convert BGR to RGB before MediaPipe, since OpenCV decodes in BGR while MediaPipe expects RGB; mixing the two corrupts detection without raising an error (Lab 1).

Configure MediaPipe for a single hand [8]. When no hand is detected, store an **invalid sample** (a flagged gap) rather than invented coordinates — this flag is what later lets the detector suspend counting instead of fabricating taps (FR-4, safety requirement).

### 5.2 Distance and normalization

For valid landmarks, the tap signal is the thumb-tip↔index-tip separation:

\[
d_{raw}(t)=\lVert p_4(t)-p_8(t)\rVert_2
\]

Raw pixel distance shrinks as the hand moves away from the camera, so it is normalized by a hand reference length that does **not** change during a tap. The default is wrist-to-middle-MCP:

\[
d_{ref}(t)=\lVert p_0(t)-p_9(t)\rVert_2,\qquad
d_{norm}(t)=\frac{d_{raw}(t)}{d_{ref}(t)}
\]

`d_norm` is large when the fingers are apart (open) and near zero at contact (close), and is approximately stable for a fixed pose regardless of apparent hand size. Smooth `d_ref` lightly and invalidate implausible changes (a sudden jump signals a bad landmark frame). This is **approximate scale normalization**, not full geometric invariance: perspective, depth, foreshortening, and out-of-plane pose remain limitations (§12), which is also why calibration/undistortion is held as an optional experiment (§9.2) rather than a runtime requirement.

### 5.3 Temporal smoothing

Human tapping sits at roughly 1–6 Hz; landmark jitter is higher-frequency, so one light low-pass stage cleans the signal without flattening real taps. Select the filter after an offline comparison (§9.1) and use a **causal** filter at runtime — an exponential moving average, or a short trailing moving average — so the live display and the detector never depend on future frames. Centered Savitzky–Golay [12] and zero-phase `filtfilt` may be used for **offline** analysis/figures but must not be described or used as causal live filters, since they look ahead in time.

### 5.4 Tap detection

A single threshold on `d_norm` chatters when the signal hovers near it, over-counting. A **two-state hysteresis** detector with two thresholds avoids this:

- transition `OPEN → CLOSED` when `d_norm` falls below a lower threshold `T_low`;
- while CLOSED, record the candidate **close** at the local minimum;
- transition `CLOSED → OPEN` when `d_norm` rises above a higher threshold `T_high` (`T_high > T_low`);
- **count a tap only after the full `OPEN → CLOSED → OPEN` cycle completes**; and
- enforce a minimum closed time and a minimum cycle duration (a refractory period) so brief dips and physically impossible rates cannot double-count.

Derive `T_low`/`T_high` from a short open/close calibration or from robust session percentiles of `d_norm` rather than fixed absolutes, so the detector adapts to a participant whose full "open" or contact depth differs. **Suspend detection during invalid gaps**; mark any cycle that spans a gap as uncertain and exclude it from the confident count.

### 5.5 Metrics

- **Count:** accepted complete cycles.
- **Frequency:** accepted taps divided by **valid** assessment duration (gap time removed).
- **ITI:** time between accepted close events.
- **Timing variability:** `CV_ITI = std(ITI) / mean(ITI)`; lower is more regular.
- **Amplitude:** local open distance minus close distance per cycle.
- **Performance change:** predefined early-versus-late differences in frequency and median amplitude.

Do not use `1 − CV`, which can be negative. If a bounded display score is desired, use `1 / (1 + CV_ITI)` and label it as derived.

### 5.6 Illustrative pipeline walkthrough

> **Illustrative only.** The numbers below are hand-constructed to show data flow and detector behaviour. They are **not** measured results and must never be reported as performance (§8.1).

Setup: 30 FPS nominal, one hand at ~50 cm, ~2 Hz tapping for 10 s. Reference length `d_ref ≈ 180 px`, stable. Hysteresis thresholds on `d_norm`: `T_low = 0.35`, `T_high = 0.60`; minimum closed time 60 ms; minimum cycle 150 ms.

**First tap cycle, frame by frame:**

| t (s) | d_raw (px) | d_norm | state action |
|---|---|---|---|
| 0.00 | 212 | 1.18 | OPEN |
| 0.07 | 148 | 0.82 | OPEN (above T_high) |
| 0.12 | 94 | 0.52 | OPEN (in hysteresis band, > T_low) |
| 0.17 | 40 | 0.22 | < T_low → **CLOSED** |
| 0.19 | 27 | 0.15 | local minimum → candidate close @ 0.19 |
| 0.24 | 59 | 0.33 | CLOSED (below T_high) |
| 0.29 | 115 | 0.64 | > T_high → **OPEN**, cycle complete → **tap #1** |

Tap #1 is counted at close time 0.19 s; its amplitude = local open (1.18) − close (0.15) = **1.03**. Note the frame at 0.12 (`d_norm` 0.52) sits between the thresholds and correctly triggers nothing — a single 0.5 threshold would have fired here and again on the way back up.

**Normalization check:** in a separate "far" recording the same pose gives `d_raw ≈ 150 px` at open, but `d_ref` also shrinks to ~128 px, so `d_norm ≈ 1.17` — essentially unchanged. Distance invariance holds to first order; frame-edge perspective effects do not, which is the §9.2 experiment's target.

**Invalid gap:** at t ≈ 3.30–3.37 the hand briefly rotates and MediaPipe returns nothing for 2 frames. Those samples are flagged invalid, the detector **suspends**, and the partially-formed cycle around the gap is marked uncertain and excluded. No phantom tap is produced, and 0.07 s is removed from the valid assessment duration.

**Session roll-up (illustrative):** 19 accepted taps over 9.93 s valid duration → frequency ≈ **1.91 Hz**; ITI mean ≈ 0.523 s, std ≈ 0.049 s → `CV_ITI ≈ 0.094`, derived regularity `1/(1+CV) ≈ 0.914`; amplitude early-window ≈ 1.04 vs late-window ≈ 0.88 → **≈ 15% within-session amplitude decrement**. Whether that decrement is real or noise is exactly what repeatability analysis (§8.2), not this walkthrough, decides.

---

## 6. Minimal User Interface

Use one main window with:

- subject code, hand, duration, and condition tag;
- Start, Stop, and Export;
- live video with landmarks and thumb–index line;
- scrolling normalized-distance signal with detected taps;
- count, current frequency, FPS, and quality indicator; and
- compact end-of-session summary.

Camera-calibration screens, session history, live filter selection, and edge toggles are not required. Course diagnostics can be produced by separate experiment scripts.

---

## 7. Data and Protocol

### 7.1 Recording protocol

- Record one hand for approximately 10 seconds.
- Use a plain background and keep the entire hand visible.
- Keep the palm approximately facing the camera.
- Capture a baseline at roughly 50 cm under even lighting.
- Preserve original recordings separately from outputs.

### 7.2 Evaluation scenarios

Use a baseline plus selected one-factor stress cases:

1. baseline;
2. near and far distance;
3. dim and side lighting;
4. off-centre hand;
5. rotated or tilted hand;
6. slow and fast tapping; and
7. brief hand loss or occlusion.

Record repeated trials per scenario. This replaces the original 144-combination full factorial matrix.

### 7.3 Ground truth

Annotate one row per contact event:

`event_id, frame_idx, timestamp_sec, annotator_id`

Repeat annotation or use a second annotator for a subset. A ±100 ms matching tolerance is acceptable only when justified by frame rate and annotation uncertainty; optionally report sensitivity at ±50, ±100, and ±150 ms. Bounding boxes are required only if an ROI tracker is evaluated.

### 7.4 Outputs

| Artifact | Format | Contents |
|---|---|---|
| Per-frame log | CSV | frame, time, distances, validity, available confidence |
| Detected events | CSV | time, cycle amplitude, uncertainty flag |
| Ground truth | CSV | manually annotated contact events |
| Session summary | JSON | metrics, quality, condition, configuration |
| Figures | PNG | full signal/events, ITI, and amplitude series |

---

## 8. Evaluation

### 8.1 Primary metrics

- event precision, recall, and F1 using one-to-one time matching;
- absolute and percentage count error;
- close-event timing MAE;
- frequency MAE;
- valid-frame coverage and longest gap; and
- achieved FPS.

Report results per scenario and in aggregate. Illustrative simulations (such as §5.6) must not be presented as measured performance.

### 8.2 Amplitude evaluation

Without an independent physical instrument, normalized amplitude has no absolute ground truth. Evaluate repeatability across baseline trials, sensitivity to distance/pose, and within-session trends. Use Bland–Altman agreement analysis [13] only when there is a meaningful second measurement method or a defined repeatability question.

### 8.3 Essential diagnostics

Retain FPS, valid-landmark percentage, rolling `d_ref` stability, gap count, and longest gap. Assign `reliable`, `borderline`, or `discard` using documented thresholds tuned on development—not final test—recordings.

---

## 9. Course Ablations and Optional Extensions

### 9.1 Small ablations

Compare:

1. normalized versus unnormalized distance;
2. raw versus causally smoothed signal;
3. full-frame MediaPipe versus HSV-derived ROI, if cropping is correctly implemented; and
4. unenhanced versus one low-light enhancement method.

Measure event F1, valid-frame coverage, and FPS. Prefer the simpler method unless added complexity produces a meaningful improvement.

### 9.2 Optional calibration experiment

Capture 10–20 checkerboard views and calculate reprojection error [11]. Compare centered and edge-position measurements with and without undistortion. Make per-frame undistortion mandatory only if it materially reduces error on the actual webcam.

### 9.3 Optional tracking extension

If landmark dropout is a major measured failure, evaluate **one** fallback, preferably Lucas–Kanade optical flow with a forward–backward check [10]. It may bridge the display, but predicted samples must not create events. Add CamShift or Kalman filtering only if a separate experiment demonstrates further benefit.

---

## 10. Testing

### 10.1 Unit tests

- known coordinates produce the expected distance;
- uniform scaling leaves `d_norm` unchanged;
- invalid or near-zero `d_ref` is rejected;
- complete synthetic cycles are counted once;
- shallow crossings and incomplete cycles are rejected;
- no event is counted across an invalid gap; and
- known ITIs produce correct mean, CV, and early/late changes.

### 10.2 Integration tests

- an annotated clip meets chosen event/count tolerances;
- a hand-loss clip creates invalid samples and no phantom taps;
- live and offline modes share the same event logic; and
- frame, event, and summary exports are mutually consistent.

---

## 11. Implementation Plan

| Phase | Deliverable |
|---|---|
| 1 | Timestamped capture and live MediaPipe overlay |
| 2 | Validated normalized-distance signal and quality checks |
| 3 | Causal filter and hysteresis detector |
| 4 | Metrics and CSV/JSON export |
| 5 | Annotation format and evaluation harness |
| 6 | Minimal one-window UI |
| 7 | Baseline and stress-scenario results |
| 8 | Course ablations and, only if justified, one optional extension |
| 9 | Failure analysis, screenshots, and final report |

Evaluation begins before UI polish so complexity is added only in response to measured failures.

---

## 12. Limitations

- A monocular 2D distance cannot fully correct three-dimensional pose.
- Detection may degrade with occlusion, blur, lighting, visibility, and hardware.
- Normalized amplitude is relative, not a physical distance.
- Performance-change metrics do not establish fatigue or disease.
- Results apply only to tested participants, devices, and conditions.

---

## 13. References

**Domain and prior quantification work**

[1] C. Y. Lee, S. J. Kang, S.-K. Hong, H.-I. Ma, U. Lee, and Y. J. Kim, "A validation study of a smartphone-based finger tapping application for quantitative assessment of bradykinesia in Parkinson's disease," *PLOS ONE*, vol. 11, no. 7, e0158852, 2016. doi:10.1371/journal.pone.0158852

[2] S. Williams et al., "The discerning eye of computer vision: Can it measure Parkinson's finger tap bradykinesia?," *Journal of the Neurological Sciences*, vol. 416, 117003, 2020. doi:10.1016/j.jns.2020.117003

[3] M. H. G. Monje et al., "Remote evaluation of Parkinson's disease using a conventional webcam and artificial intelligence," *Parkinsonism & Related Disorders*, vol. 88, pp. 40–46, 2021. doi:10.1016/j.parkreldis.2021.06.012

[4] A. J. Noyce, A. Nagy, S. Acharya, S. Hadavi, et al., "Technologies assessing limb bradykinesia in Parkinson's disease," *Journal of Parkinson's Disease*, vol. 7, no. 2, pp. 315–326, 2017. doi:10.3233/JPD-161000

[5] L. di Biase et al., "Quantitative analysis of bradykinesia and rigidity in Parkinson's disease," *Frontiers in Neurology*, vol. 9, 121, 2018. doi:10.3389/fneur.2018.00121

[6] Movement Disorder Society, "MDS-sponsored revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS)," 2019.

[7] P. Mazzoni, M. Shabbir, et al., "Towards objective quantification of hand tremors and bradykinesia using contactless sensors: A systematic review," *Sensors*, vol. 21, no. 21, 7284, 2021.

**Methods and tools**

[8] F. Zhang, V. Bazarevsky, A. Vakunov, A. Tkachenka, C. Sung, C.-L. Chang, and M. Grundmann, "MediaPipe Hands: On-device real-time hand tracking," *Proc. IEEE/CVF CVPR Workshops*, 2020. arXiv:2006.10214

[9] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.

[10] B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," *Proc. 7th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 674–679, 1981.

[11] Z. Zhang, "A flexible new technique for camera calibration," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no. 11, pp. 1330–1334, 2000. doi:10.1109/34.888718

[12] A. Savitzky and M. J. E. Golay, "Smoothing and differentiation of data by simplified least squares procedures," *Analytical Chemistry*, vol. 36, no. 8, pp. 1627–1639, 1964. doi:10.1021/ac60214a047

[13] J. M. Bland and D. G. Altman, "Statistical methods for assessing agreement between two methods of clinical measurement," *The Lancet*, vol. 327, no. 8476, pp. 307–310, 1986. doi:10.1016/S0140-6736(86)90837-8

**General course texts:** R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 3rd ed., Pearson, 2012; R. Szeliski, *Computer Vision: Algorithms and Applications*, Springer, 2011.

---

**Prototype reminder:** This is a computer-vision measurement and evaluation prototype for coursework and research, not a diagnostic system or medical device.
