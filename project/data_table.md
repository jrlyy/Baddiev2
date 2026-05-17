# Dataset Inventory

---

## ShuttleSet (SS) — Shot Type Dataset (44 matches total)

**Role in pipeline:** Supervised shot type classification. 19 train + 2 held-out = 21 active matches.

**Frame extraction:**
- **Sparse** = every 4th frame + hit frames (~10 FPS), JPEG
- **None** = never downloaded

**Skeleton extraction:**
- **GDINO** = `skeletons.npy` + `frame_nums.npy` per match, GroundingDINO-guided YOLOv8-Pose
- **None** = not yet extracted

**Shuttle tracking:**
- **Hit-frame** = one row per shot `(frame_num, x, y, visible)` — TrackNet detection at the hit frame only
- **None** = not yet run

---

### Active matches — Train (19)

All men's singles, both players right-handed, good skeleton quality.

| # | Short name | Match folder | Shots | Frames | Skeleton | Shuttle |
|---|-----------|--------------|------:|-------:|----------|---------|
| 1 | Antonsen vs Axelsen — WTF 2020 F | `Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals` | 895 | 6 446 | GDINO | None |
| 2 | Ginting vs Antonsen — INA Masters 2020 F | `Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final` | 824 | 4 726 | GDINO | None |
| 3 | Ginting vs Axelsen — INA Masters 2020 SF | `Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals` | 506 | 3 307 | GDINO | None |
| 4 | Chou vs Antonsen — Fuzhou 2019 SF | `CHOU_Tien_Chen_Anders_ANTONSEN_Fuzhou_Open_2019_Semi-finals` | 704 | 4 746 | GDINO | None |
| 5 | Chou vs Christie — INA Open 2019 QF | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Indonesia_Open_2019_Quarter-finals` | 1 219 | 7 447 | GDINO | None |
| 6 | Vittinghus vs Antonsen — THA Open 2021 SF | `Hans-Kristian_Solberg_Vittinghus_Anders_Antonsen_TOYOTA_THAILAND_OPEN_2021_SemiFinals` | 735 | 5 441 | GDINO | None |
| 7 | Vittinghus vs Lee CY — THA Open 2021 QF | `Hans-Kristian_Solberg_Vittinghus_Lee_Cheuk_Yu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | 1 273 | 9 638 | GDINO | None |
| 8 | Ng KL vs Christie — MAS Masters 2020 QF | `NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals` | 951 | 6 179 | GDINO | None |
| 9 | Ng KL vs Srikanth — WTF 2020 QF | `Ng_Ka_Long_Angus_Kidambi_Srikanth_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | 1 200 | 9 183 | GDINO | None |
| 10 | Ng KL vs Lee CY — THA Open 2021 QF | `Ng_Ka_Long_Angus_Lee_Cheuk_Yiu_YONEX_Thailand_Open_2021_QuarterFinals` | 829 | 5 688 | GDINO | None |
| 11 | Antonsen vs Sameer — THA Open 2021 QF | `Anders_Antonsen_Sameer_Verma_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | 1 204 | 9 040 | GDINO | None |
| 12 | Axelsen vs Shi YQ — All England 2020 QF | `Viktor_AXELSEN _SHI_Yu_Qi_All_England_Open_2020_QuarterFinals` | 312 | 2 142 | GDINO | None |
| 13 | Axelsen vs Chen Long — MAS Masters 2020 QF | `Viktor_AXELSEN_CHEN_Long_Malaysia_Masters_2020_QuarterFinals` | 910 | 6 194 | GDINO | None |
| 14 | Axelsen vs Ng KL — MAS Masters 2020 SF | `Viktor_AXELSEN_NG_Ka_Long_Angus_Malaysia_Masters_2020_SemiFinals` | 569 | 3 710 | GDINO | None |
| 15 | Axelsen vs Ginting — THA Open 2021 SF | `Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals` | 1 095 | 8 321 | GDINO | None |
| 16 | Axelsen vs Vittinghus — THA Open 2021 F | `Viktor_Axelsen_Hans-Kristian_Solberg_VIittinghus_TOYOTA_THAILAND_OPEN_2021_Finals` | 727 | 5 554 | GDINO | None |
| 17 | Axelsen vs Christie — THA Open 2021 QF | `Viktor_Axelsen_Jonatan_Christie_YONEX_Thailand_Open_2021_QuarterFinals` | 673 | 5 353 | GDINO | None |
| 18 | Axelsen vs Liew Daren — THA Open 2021 QF | `Viktor_Axelsen_Liew_Daren_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | 591 | 4 136 | GDINO | None |
| 19 | Axelsen vs Ng KL — THA Open 2021 F | `Viktor_Axelsen_Ng_Ka_Long_Angus_YONEX_Thailand_Open_2021_Finals` | 862 | 5 666 | GDINO | None |
| | **Train total** | | **16 079** | **112 917** | | |

### Active matches — Held-out (2)

Reserved for final evaluation. Same eligibility criteria as train.

| # | Short name | Match folder | Shots | Frames | Skeleton | Shuttle |
|---|-----------|--------------|------:|-------:|----------|---------|
| 1 | Ginting vs Lee ZJ — WTF 2020 QF | `Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | 815 | 6 269 | GDINO | None |
| 2 | Chen Long vs Chou — WTF GS | `CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage` | 860 | 5 699 | GDINO | None |
| | **Held-out total** | | **1 675** | **11 968** | | |

---

### Excluded — Left-handed player (6)

Kento Momota is left-handed. Skeleton mirroring would be needed to make poses comparable with right-handed players; not implemented.

| # | Short name | Match folder | Cat | Shots | Frames | Exclusion reason |
|---|-----------|--------------|-----|------:|-------:|-----------------|
| 1 | Momota vs Chou — Fuzhou 2019 F | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals` | MS | 1 644 | 11 399 | Momota left-handed |
| 2 | Momota vs Chou — Korea 2019 F | `Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final` | MS | 928 | 5 986 | Momota left-handed |
| 3 | Momota vs Chou — Fuzhou 2018 F | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals` | MS | 1 166 | 5 959 | Momota left-handed |
| 4 | Momota vs Chou — Denmark 2018 F | `Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals` | MS | 1 311 | 8 393 | Momota left-handed |
| 5 | Momota vs Chou — Malaysia 2018 QF | `Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals` | MS | 972 | 6 562 | Momota left-handed |
| 6 | Momota vs Axelsen — MAS Masters 2020 F | `Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals` | MS | 756 | 5 391 | Momota left-handed |

### Excluded — Women's singles (11)

Different biomechanics and body proportions add confounding variation to skeleton-based classification at current data scale.

| # | Short name | Match folder | Shots | Frames | Exclusion reason |
|---|-----------|--------------|------:|-------:|-----------------|
| 1 | An Se Young vs Intanon — THA Open 2021 QF | `An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals` | 663 | 6 734 | Women's singles |
| 2 | Marin vs Supanida — THA Open 2021 QF | `Carolina_Marin_Supanida_Katethong_YONEX_Thailand_Open_2021_QuarterFinals` | 629 | 3 441 | Women's singles |
| 3 | Marin vs Neslihan — THA Open 2021 QF | `Carolina_Marin_Neslihan_Yigit_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | 518 | 4 075 | Women's singles |
| 4 | Intanon vs Sindhu — THA Open 2021 QF | `Ratchanok_Intanon_Pusarla_V._Sindhu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | 583 | 4 688 | Women's singles |
| 5 | Marin vs An Se Young — THA Open 2021 SF | `Carolina_Marin_An_Se_Young_TOYOTA_THAILAND_OPEN_2021_SemiFinals` | 765 | 5 064 | Women's singles |
| 6 | Marin vs An Se Young — WTF 2020 QF | `Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | 814 | 6 752 | Women's singles |
| 7 | Kosetskaya vs Michelle Li — WTF 2020 QF | `Evgeniya_Kosetskaya_Michelle_Li_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | 568 | 4 951 | Women's singles |
| 8 | Blichfeldt vs Busanan — THA Open 2021 QF | `Mia_Blichfeldt_Busanan_Ongbamrungphan_YONEX_Thailand_Open_2021_QuarterFinals` | 581 | 3 672 | Women's singles |
| 9 | Sindhu vs Pornpawee — WTF 2020 QF | `Pusarla_V._Sindhu_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | 644 | 5 464 | Women's singles |
| 10 | Marin vs Pornpawee — WTF 2020 SF | `Carolina_Marin_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_SemiFinals` | 577 | 4 215 | Women's singles |
| 11 | An Se Young vs Pornpawee — THA Open 2021 QF | `An_Se_Young_Pornpawee_Chochuwong_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | 747 | 0 | Women's singles; frames never extracted |

### Excluded — Non-standard camera angle or no video (3)

Side-court or overhead broadcast angles break the top-court/bottom-court player assignment that the skeleton pipeline assumes.

| # | Short name | Match folder | Cat | Shots | Exclusion reason |
|---|-----------|--------------|-----|------:|-----------------|
| 1 | Chou vs Christie — Sudirman Cup 2019 QF | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Sudirman_Cup_2019_Quarter-finals` | MS | 529 | Sudirman Cup side-court camera |
| 2 | Chou vs Ng KL — Sudirman Cup 2019 GS | `CHOU_Tien_Chen_NG_Ka_Long_Angus_Sudirman_Cup_2019_Group_Stage` | MS | 692 | Sudirman Cup side-court camera |
| 3 | Ng KL vs Shi YQ — Thailand Masters 2020 SF | `NG_Ka_Long_Angus_SHI_Yu_Qi_Thailand_Masters_2020_SemiFinals` | MS | 680 | No YouTube URL in dataset |

### Excluded — Bad data quality (2)

| # | Short name | Match folder | Cat | Shots | Frames | Exclusion reason |
|---|-----------|--------------|-----|------:|-------:|-----------------|
| 1 | Ginting vs Gemke — THA Open 2021 QF | `Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals` | MS | 1 127 | 3 734 | All skeletons wrong (extraction failure) |
| 2 | Antonsen vs Christie — INA Masters 2020 QF | `Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals` | MS | 1 018 | 1 018 | Sparse frames & skeletons, not useful |

### Excluded — Non-standard camera angle (pending verification) (1)

| # | Short name | Match folder | Cat | Shots | Exclusion reason |
|---|-----------|--------------|-----|------:|-----------------|
| 1 | Chen Long vs Chou — Denmark Open 2019 QF | `CHEN_Long_CHOU_Tien_Chen_Denmark_Open_2019_QuarterFinal` | MS | 818 | Camera angle or download issue (needs verification) |

---

### Summary

| Category | Matches | Shots | Reason |
|----------|--------:|------:|--------|
| **Train** | 19 | 16 079 | Men's singles, right-handed, good quality |
| **Held-out** | 2 | 1 675 | Men's singles, right-handed, reserved for eval |
| Excluded: left-handed | 6 | 6 777 | Momota is left-handed |
| Excluded: women's singles | 11 | 7 089 | Different biomechanics domain |
| Excluded: camera angle / no video | 3 | 1 901 | Side-court angle or missing URL |
| Excluded: bad quality | 2 | 2 145 | Broken skeletons or sparse data |
| Excluded: unverified | 1 | 818 | Needs investigation |
| **Total** | **44** | **36 484** | |

---

## FineBadminton (FB) — Cross-Dataset Evaluation (40 rallies, 413 shots)

**Role in pipeline:** Cross-dataset generalization test. Train on ShuttleSet, evaluate on FB with mapped labels. Not used for model selection.

**Source:** FineBadminton dataset — different matches, different annotators, different label taxonomy.

**Annotations:** `Datasets/FineBadminton-dataset/dataset/transformed_combined_rounds_output_en_evals_translated.json`

**Skeletons:** `datasets_preprocessing/finebadminton_skeletons/` (40 × `.npy`, YOLOv8) and `datasets_preprocessing/finebadminton_skeletons_gdino_v2/` (GDINO-guided)

### FB Shot Type Taxonomy (hierarchical: hit_type → subtype)

| # | FB Hit Type | Count | Subtypes |
|---|------------|------:|----------|
| 1 | kill | 104 | jump smash (37), full smash (24), common smash (18), slice smash (15), stick smash (10) |
| 2 | push shot | 84 | flat lift (67), unspecified (17) |
| 3 | serve | 44 | short serve (32), high serve (8), flick serve (4) |
| 4 | net shot | 44 | spinning net (22), unspecified (22) |
| 5 | block | 43 | unspecified (42), high block (1) |
| 6 | drive | 34 | unspecified (22), high drive (9), flat drive (3) |
| 7 | clear | 31 | attacking clear (25), unspecified (6) |
| 8 | drop shot | 23 | slice drop (14), stop drop (4), reverse slice (3), blocked drop (2) |
| 9 | cross-court net shot | 20 | all unspecified |
| 10 | net lift | 16 | high lift (9), unspecified (7) |
| 11 | net kill | 8 | all unspecified |

### FB → SS Label Mapping (10-class shared taxonomy)

Evaluation uses a merged taxonomy where both datasets have unambiguous coverage. SS classes without a FB equivalent (defensive return lob, defensive return drive, driven flight, back-court drive, passive drop, rush) are excluded from evaluation.

| Merged Class | SS Source Classes | FB Source Classes | Confidence |
|---|---|---|---|
| **short service** | short service (16) | serve / short serve | High |
| **long service** | long service (17) | serve / high serve + flick serve | High |
| **smash** | smash (2) + wrist smash (3) | kill (all subtypes) | High |
| **clear** | clear (6) | clear | High |
| **drop** | drop (10) + passive drop (11) | drop shot (all subtypes) | High |
| **net shot** | net shot (0) | net shot | High |
| **cross-court net** | cross-court net shot (15) | cross-court net shot | Exact |
| **lob** | lob (4) | net lift + push shot / flat lift | Medium |
| **drive** | drive (7) + back-court drive (9) | drive (all subtypes) | Medium |
| **block** | return net (1) | block | Medium |

**Note on "push shot / flat lift":** FB's "push shot" category is 80% "flat lift" subtype, which biomechanically resembles SS's "lob" more than "push". Mapped to lob, not push.

---

## Benchmark Comparison

| Method | Year | Input Modalities | Classes | Dataset | Accuracy | Macro-F1 | Other | Reference |
|---|---|---|---|---|---|---|---|---|
| TemPose-TF (Ibh et al.) | 2023 | Skeleton + shuttle pos. + court pos. | 25 (merged) | ShuttleSet | 58.2% | 43.7% | Top-2: 75.6%; Clear: 12% F1, Drive: 14% F1 | 10.1109/CVPRW59228.2023.00549 |
| LSTM (Jain et al.) | 2024 | Skeleton keypoints | 3 (clear, serve, smash) | Own (BWF match) | 89.5% | — | Only 3 coarse classes | 10.1007/978-3-031-60935-0_28 |
| Shot Refinement | 2024 | Shuttle tracking + hit detection | 7 (merged) | — | 72.1% | — | No skeleton input | — |
| BST (Chang) | 2025 | Skeleton + shuttle traj. + court pos. | 35 (17 × top/bottom) | ShuttleSet (40 matches) | 76.9% | 70.4% | Top-2: 92.7%; transformer + TraClean | arXiv:2502.21085 |
| **Ours (Run 4 best)** | 2026 | Skeleton + shuttle traj. (cross-attn) | 18 | ShuttleSet (21 matches) | 61.9% | 59.3% | Top-3: 90.6%; ST-GCN, GDINO extraction | — |
| **Ours (Run 5, planned)** | 2026 | + bone vectors + bbox norm + attn pool | 18 | ShuttleSet (21 matches) | TBD | Target ~65% | + cross-dataset eval on FB (413 shots) | — |

---

## Standardisation gaps

| Issue | Affected matches | Priority |
|-------|-----------------|----------|
| Shuttle tracking not yet run | all 21 active | High — needed for shuttle fusion ablation |
| GDINO skeletons are match-level (not per-rally) | all 21 active | Info — `skeletons.npy` + `frame_nums.npy` per match |
| Folder name has trailing space: `AXELSEN _SHI` and `AXELSEN _Indonesia` | ss02, ss12 | Low — rename would break all paths |
| Mixed case: `Viktor_Axelsen` vs `Viktor_AXELSEN` | multiple | Low |
