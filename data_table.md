# Dataset Inventory

---

## FineBadminton (FB) — Labeled Tactical Strategy Dataset

**Role in pipeline:** Phase B (few-shot training + evaluation). All shots have strategy annotations.

**Skeleton extraction:**
- **GDINO v2** = one `.npy` per rally, shape `(2, T_rally, 34)` — GroundingDINO-guided YOLOv8-Pose. All 40 rallies extracted.

**Shuttle tracking:**
- **Hit-frame** = TrackNet detection at hit frame only, shape `(N_shots, 3)` [x, y, vis] per shot.

**Labels used (296 out of 414 shots have a usable strategy):**

| Label | Strategy | Count |
|-------|----------|------:|
| intercept | Intercept | 108 |
| create_depth | Create depth | 61 |
| defensive | Defensive | 59 |
| passive | Passive | 50 |
| move_to_net | Move to net | 18 |

> Excluded: shots with only `deception`, `hesitation`, `seamlessly`, or `a high net early shot` labels (not skeleton-classifiable). Multi-strategy shots use first valid strategy.

| fb_id | Players | Cat | Rallies | Shots (total) | Shots (labeled) | Frames | Skeletons (GDINO v2) |
|-------|---------|-----|--------:|-------------:|---------------:|-------:|----------------------|
| **fb0011** | Kento MOMOTA vs Viktor AXELSEN | MS | 5 | 83 | 74 | ~2 038 | 5/5 rallies |
| **fb0012** | Chen Yu Fei vs Wang Zhi Yi | WS | 3 | 41 | 31 | ~1 141 | 3/3 rallies |
| **fb0013** | Shi Yu Qi vs Su Li Yang | MS | 3 | 25 | 19 | ~594 | 3/3 rallies |
| **fb0014** | Leong Jun Hao vs Li Shi Feng | MS | 5 | 32 | 22 | ~766 | 5/5 rallies |
| **fb0016** | Lin Hang Yang vs Shi Yu Qi | MS | 3 | 12 | 8 | ~318 | 3/3 rallies |
| **fb0018** | Rasmus Gemke vs Tien Chen Chou | MS | 3 | 50 | 39 | ~1 233 | 3/3 rallies |
| **fb0019** | An Se Young vs Wang Zhi Yi | WS | 3 | 33 | 25 | ~814 | 3/3 rallies |
| **fb0022** | Akane Yamaguchi vs Carolina Marin | WS | 5 | 64 | 55 | ~1 663 | 5/5 rallies |
| **fb0025** | Chen Yu Fei vs Gregoria Mariska Tunjung | WS | 3 | 25 | 20 | ~710 | 3/3 rallies |
| **fb0028** | An Se Young vs Tai Tzu Ying | WS | 3 | 21 | 15 | ~607 | 3/3 rallies |
| **fb0030** | Ng Ka Long Angus vs Viktor Axelsen | MS | 4 | 28 | 21 | ~736 | 4/4 rallies |
| **Total** | 11 matches · 6 MS / 5 WS | | **40** | **414** | **296** | **~10 620** | **40/40** |

### Train / Val / Test Split

Evaluation uses **5-fold stratified cross-validation**, stratified by **strategy label** (not by match).

| Split | Shots per fold |
|-------|---------------:|
| Train | ~177 |
| Val   | ~60  |
| Test  | ~59  |

> **Important:** all 11 matches appear in every fold's train, val, and test partitions — the split is shot-level, not match-level. This means train and test sets share the same players and playing styles, which is a known limitation for the current dataset size (296 shots across 11 matches is too small for a clean match-level hold-out). A match-level leave-one-out (11-fold, ~27 test shots/fold) would give a cleaner but noisier estimate.

---

## ShuttleSet (SS) — Unlabeled Biomechanics Dataset

**Role in pipeline:** Phase A (SSL pre-training only). No strategy labels.

**Frame extraction:**
- **Full** = consecutive frames for every frame in the rally (streaming pipeline, gap=1)
- **Sparse** = ~10-frame-interval samples around hit moments only (old PNG pipeline, converted to JPEG)

**Skeleton extraction:**
- **GDINO/rally** = one `.npy` per rally, shape `(2, T, 34)`, covers every extracted frame — via GroundingDINO-guided YOLOv8-Pose
- **YOLOv8/shot** = one `.npy` per shot, shape `(2, 16, 34)`, 16-frame window centred on hit — via standard YOLOv8-Pose top-2-confidence (chair umpire contamination risk)
- **None** = not yet extracted

**Shuttle tracking:**
- **Hit-frame** = one row per shot `(frame_num, x, y, visible)`, shape `(N_shots, 4)` — TrackNet detection at the hit frame only
- **None** = not yet run

---

# ShuttleSet Match Inventory

**Frame extraction:**
- **Full** = consecutive frames for every frame in the rally (streaming pipeline, gap=1)
- **Sparse** = ~10-frame-interval samples around hit moments only (old PNG pipeline, converted to JPEG)

**Skeleton extraction:**
- **GDINO/rally** = one `.npy` per rally, shape `(2, T, 34)`, covers every extracted frame — via GroundingDINO-guided YOLOv8-Pose
- **YOLOv8/shot** = one `.npy` per shot, shape `(2, 16, 34)`, 16-frame window centred on hit — via standard YOLOv8-Pose top-2-confidence (chair umpire contamination risk)
- **None** = not yet extracted

**Shuttle tracking:**
- **Hit-frame** = one row per shot `(frame_num, x, y, visible)`, shape `(N_shots, 4)` — TrackNet detection at the hit frame only
- **None** = not yet run

---

| Label | Short name | Match folder | Cat | Shots | Frames | Frame extraction | Skeleton | Shuttle tracking |
|-------|-----------|--------------|-----|------:|-------:|-----------------|----------|-----------------|
| **ss01** | Momota vs Chou — Fuzhou 2019 F | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals` | MS | 1644 | 40 401 | **Full** (consecutive) | **GDINO/rally** (39 rallies) | Hit-frame (1143/1644 vis) |
| **ss02** | Ginting vs Axelsen — INA Masters 2020 SF | `Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals` | MS | 506 | 11 509 | **Full** (consecutive) | None | Hit-frame (282/506 vis) |
| ss03 | An Se Young vs Intanon — THA Open 2021 QF | `An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals` | WS | 663 | 1 972 | Sparse | **YOLOv8/shot** (460 files) | Hit-frame (612/663 vis) |
| ss04 | Antonsen vs Christie — INA Masters 2020 QF | `Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals` | MS | 1018 | 2 891 | Sparse | None | Hit-frame (626/1018 vis) |
| ss05 | Ginting vs Antonsen — INA Masters 2020 F | `Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final` | MS | 824 | 2 394 | Sparse | None | Hit-frame (457/824 vis) |
| ss06 | Ginting vs Gemke — THA Open 2021 QF | `Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals` | MS | 1127 | 3 305 | Sparse | None | Hit-frame (1059/1127 vis) |
| ss07 | Chen Long vs Chou — Denmark Open 2019 QF | `CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal` | MS | 818 | 2 361 | Sparse | None | Hit-frame (700/818 vis) |
| ss08 | Chen Long vs Chou — World Tour Finals GS | `CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage` | MS | 860 | 2 472 | Sparse | None | Hit-frame (489/860 vis) |
| ss09 | Chou vs Antonsen — Fuzhou Open 2019 SF | `CHOU_Tien_Chen_Anders_ANTONSEN_Fuzhou_Open_2019_Semi-finals` | MS | 704 | 2 030 | Sparse | None | None |
| ss10 | Chou vs Christie — Indonesia Open 2019 QF | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Indonesia_Open_2019_Quarter-finals` | MS | 1219 | 3 519 | Sparse | None | None |
| ss11 | Chou vs Christie — Sudirman Cup 2019 QF | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Sudirman_Cup_2019_Quarter-finals` | MS | 529 | 1 540 | Sparse | None | None |
| ss12 | Chou vs Ng Ka Long — Sudirman Cup 2019 GS | `CHOU_Tien_Chen_NG_Ka_Long_Angus_Sudirman_Cup_2019_Group_Stage` | MS | 692 | 2 010 | Sparse | None | None |
| ss13 | Marin vs Supanida — THA Open 2021 QF | `Carolina_Marin_Supanida_Katethong_YONEX_Thailand_Open_2021_QuarterFinals` | WS | 629 | 1 864 | Sparse | None | None |
| ss14 | Momota vs Chou — Denmark Open 2018 F | `Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals` | MS | 1311 | 3 796 | Sparse | None | None |
| ss15 | Momota vs Chou — Fuzhou Open 2018 F | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals` | MS | 1166 | 3 355 | Sparse | None | None |
| ss16 | Momota vs Chou — Korea Open 2019 F | `Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final` | MS | 928 | 2 685 | Sparse | None | None |
| ss17 | Momota vs Chou — Malaysia Open 2018 QF | `Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals` | MS | 972 | 2 790 | Sparse | None | None |
| ss18 | Momota vs Axelsen — Malaysia Masters 2020 F | `Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals` | MS | 756 | 2 219 | Sparse | None | None |
| ss19 | Blichfeldt vs Busanan — THA Open 2021 QF | `Mia_Blichfeldt_Busanan_Ongbamrungphan_YONEX_Thailand_Open_2021_QuarterFinals` | WS | 581 | 1 710 | Sparse | None | None |
| ss20 | Ng Ka Long vs Christie — MAS Masters 2020 QF | `NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals` | MS | 951 | 2 723 | Sparse | None | None |
| ss21 | Ng Ka Long vs Lee Cheuk Yiu — THA Open 2021 QF | `Ng_Ka_Long_Angus_Lee_Cheuk_Yiu_YONEX_Thailand_Open_2021_QuarterFinals` | MS | 829 | 2 413 | Sparse | None | None |
| ss22 | Axelsen vs Shi Yu Qi — All England 2020 QF | `Viktor_AXELSEN _SHI_Yu_Qi_All_England_Open_2020_QuarterFinals` | MS | 312 | 893 | Sparse | None | None |
| ss23 | Axelsen vs Chen Long — MAS Masters 2020 QF | `Viktor_AXELSEN_CHEN_Long_Malaysia_Masters_2020_QuarterFinals` | MS | 910 | 2 640 | Sparse | None | None |
| ss24 | Axelsen vs Ng Ka Long — MAS Masters 2020 SF | `Viktor_AXELSEN_NG_Ka_Long_Angus_Malaysia_Masters_2020_SemiFinals` | MS | 569 | 1 656 | Sparse | None | None |
| ss25 | Axelsen vs Christie — THA Open 2021 QF | `Viktor_Axelsen_Jonatan_Christie_YONEX_Thailand_Open_2021_QuarterFinals` | MS | 673 | 2 000 | Sparse | None | None |

**Totals:** 25 matches · 19 MS / 3 WS · 20 879 shots · ~120 000 frames

---

## Standardisation gaps

| Issue | Affected matches | Priority |
|-------|-----------------|----------|
| Only 2 matches have full consecutive frames | ss03–ss25 | High — scrubber useless without full frames |
| Skeleton only for ss01 (GDINO) and ss03 (YOLOv8) | ss02, ss04–ss25 | High |
| Shuttle tracking only at hit frame (not per-frame dense) | all 25 | Medium — dense tracking needs re-run on full frames |
| ss03 skeleton uses YOLOv8 top-2 (chair umpire contamination ~14%) | ss03 | Medium — redo with GDINO |
| Folder name has trailing space: `AXELSEN _SHI` and `AXELSEN _Indonesia` | ss02, ss22 | Low — rename would break all paths |
| Mixed case: `Viktor_Axelsen` vs `Viktor_AXELSEN` | ss25 vs ss22–24 | Low |