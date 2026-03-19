# ShuttleSet Match Split Table

> **Config:** `datasets_preprocessing/shuttleset_split.json`  
> 8 train (SSL pre-training) · 1 test (SSL monitoring) · 2 held-out (strategy prediction + expert eval)

**Current:** 19 train · 0 test · 2 held-out · 16 unused = 37 total

## Train (SSL pre-training) (19)

| # | Split | Match | Winner | Loser | Rallies | Shots | Skeleton |
|---|-------|-------|--------|-------|---------|-------|----------|
| 1 | TRAIN | `Anders_Antonsen_Sameer_Verma_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | Anders ANTONSEN | Sameer VERMA | 42 | 1204 | GDINO |
| 2 | TRAIN | `Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals` | Anders ANTONSEN | Viktor AXELSEN | 38 | 860 | GDINO |
| 3 | TRAIN | `Anthony_Sinisuka_GINTING_Anders_ANTONSEN_Indonesia_Masters_2020_Final` | Anthony Sinisuka GINTING | Anders ANTONSEN | 38 | 824 | GDINO |
| 4 | TRAIN | `Anthony_Sinisuka_GINTING_Viktor_AXELSEN _Indonesia_Masters_2020_SemiFinals` | Anthony Sinisuka GINTING | Viktor AXELSEN | 39 | 506 | GDINO |
| 5 | TRAIN | `CHOU_Tien_Chen_Anders_ANTONSEN_Fuzhou_Open_2019_Semi-finals` | CHOU Tien Chen | Anders ANTONSEN | 33 | 704 | GDINO |
| 6 | TRAIN | `CHOU_Tien_Chen_Jonatan_CHRISTIE_Indonesia_Open_2019_Quarter-finals` | CHOU Tien Chen | Jonatan CHRISTIE | 39 | 1219 | GDINO |
| 7 | TRAIN | `Hans-Kristian_Solberg_Vittinghus_Anders_Antonsen_TOYOTA_THAILAND_OPEN_2021_SemiFinals` | Hans-Kristian Solberg VITTINGHUS | Anders ANTONSEN | 40 | 704 | GDINO |
| 8 | TRAIN | `Hans-Kristian_Solberg_Vittinghus_Lee_Cheuk_Yu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | Hans-Kristian Solberg VITTINGHUS | LEE Cheuk Yiu | 42 | 1273 | GDINO |
| 9 | TRAIN | `NG_Ka_Long_Angus_Jonatan_CHRISTIE_Malaysia_Masters_2020_QuarterFinals` | NG Ka Long Angus | Jonatan CHRISTIE | 39 | 951 | GDINO |
| 10 | TRAIN | `Ng_Ka_Long_Angus_Kidambi_Srikanth_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | NG Ka Long Angus | KIDAMBI Srikanth | 40 | 1200 | GDINO |
| 11 | TRAIN | `Ng_Ka_Long_Angus_Lee_Cheuk_Yiu_YONEX_Thailand_Open_2021_QuarterFinals` | NG Ka Long Angus | LEE Cheuk Yiu | 38 | 829 | GDINO |
| 12 | TRAIN | `Viktor_AXELSEN _SHI_Yu_Qi_All_England_Open_2020_QuarterFinals` | Viktor AXELSEN | SHI Yuqi | 36 | 312 | GDINO |
| 13 | TRAIN | `Viktor_AXELSEN_CHEN_Long_Malaysia_Masters_2020_QuarterFinals` | Viktor AXELSEN | CHEN Long | 42 | 910 | GDINO |
| 14 | TRAIN | `Viktor_AXELSEN_NG_Ka_Long_Angus_Malaysia_Masters_2020_SemiFinals` | Viktor AXELSEN | NG Ka Long Angus | 39 | 569 | GDINO |
| 15 | TRAIN | `Viktor_Axelsen_Anthony_Sinisuka_Ginting_YONEX_Thailand_Open_2021_SemiFinals` | Viktor AXELSEN | Anthony Sinisuka GINTING | 40 | 1095 | GDINO |
| 16 | TRAIN | `Viktor_Axelsen_Hans-Kristian_Solberg_VIittinghus_TOYOTA_THAILAND_OPEN_2021_Finals` | Viktor AXELSEN | Hans-Kristian Solberg VITTINGHUS | 32 | 727 | GDINO |
| 17 | TRAIN | `Viktor_Axelsen_Jonatan_Christie_YONEX_Thailand_Open_2021_QuarterFinals` | Viktor AXELSEN | Jonatan CHRISTIE | 35 | 673 | GDINO |
| 18 | TRAIN | `Viktor_Axelsen_Liew_Daren_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | Viktor AXELSEN | LIEW Daren | 37 | 537 | GDINO |
| 19 | TRAIN | `Viktor_Axelsen_Ng_Ka_Long_Angus_YONEX_Thailand_Open_2021_Finals` | Viktor AXELSEN | NG Ka Long Angus | 35 | 776 | GDINO |

## Test (SSL monitoring / checkpoint selection) (0)

*(none yet)*

## Held-out (strategy prediction + expert eval) (2)

| # | Split | Match | Winner | Loser | Rallies | Shots | Skeleton |
|---|-------|-------|--------|-------|---------|-------|----------|
| 1 | HELD_OUT | `Anthony_Sinisuka_Ginting_Lee_Zii_Jia_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | Anthony Sinisuka GINTING | LEE Zii Jia | 36 | 815 | GDINO |
| 2 | HELD_OUT | `CHEN_Long_CHOU_Tien_Chen_World_Tour_Finals_Group_Stage` | CHEN Long | CHOU Tien Chen | 40 | 860 | GDINO |

## Unused (16)

| # | Split | Match | Winner | Loser | Rallies | Shots | Skeleton |
|---|-------|-------|--------|-------|---------|-------|----------|
| 1 | UNUSED | `An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals` | An Se Young | Ratchanok INTANON | 42 | 663 | — |
| 2 | UNUSED | `Carolina_Marin_An_Se_Young_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | An Se Young | Carolina MARIN | 40 | 814 | — |
| 3 | UNUSED | `Carolina_Marin_An_Se_Young_TOYOTA_THAILAND_OPEN_2021_SemiFinals` | Carolina MARIN | An Se Young | 40 | 629 | — |
| 4 | UNUSED | `Carolina_Marin_Neslihan_Yigit_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | Carolina MARIN | Neslihan YIGIT | 36 | 518 | — |
| 5 | UNUSED | `Carolina_Marin_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_SemiFinals` | Carolina MARIN | Pornpawee CHOCHUWONG | 34 | 576 | — |
| 6 | UNUSED | `Carolina_Marin_Supanida_Katethong_YONEX_Thailand_Open_2021_QuarterFinals` | Carolina MARIN | Supanida KATETHONG | 37 | 629 | — |
| 7 | UNUSED | `Evgeniya_Kosetskaya_Michelle_Li_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | Evgeniya KOSETSKAYA | Michelle LI | 48 | 568 | — |
| 8 | UNUSED | `Kento_MOMOTA_CHOU_Tien_Chen_Denmark_Open_2018_Finals` | Kento MOMOTA | CHOU Tien Chen | 42 | 1311 | — |
| 9 | UNUSED | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2018_Finals` | Kento MOMOTA | CHOU Tien Chen | 37 | 1166 | — |
| 10 | UNUSED | `Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals` | Kento MOMOTA | CHOU Tien Chen | 39 | 1644 | — |
| 11 | UNUSED | `Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final` | Kento MOMOTA | CHOU Tien Chen | 40 | 928 | — |
| 12 | UNUSED | `Kento_MOMOTA_CHOU_Tien_Chen_Malaysia_Open_2018_QuarterFinals` | Kento MOMOTA | CHOU Tien Chen | 40 | 972 | — |
| 13 | UNUSED | `Kento_MOMOTA_Viktor_AXELSEN_Malaysia_Masters_2020_Finals` | Kento MOMOTA | Viktor AXELSEN | 46 | 756 | — |
| 14 | UNUSED | `Mia_Blichfeldt_Busanan_Ongbamrungphan_YONEX_Thailand_Open_2021_QuarterFinals` | Mia BLICHFELDT | Busanan ONGBAMRUNGPHAN | 39 | 581 | — |
| 15 | UNUSED | `Pusarla_V._Sindhu_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals` | PUSARLA V. Sindhu | Pornpawee CHOCHUWONG | 39 | 644 | — |
| 16 | UNUSED | `Ratchanok_Intanon_Pusarla_V._Sindhu_TOYOTA_THAILAND_OPEN_2021_QuarterFinals` | Ratchanok INTANON | PUSARLA V. Sindhu | 34 | 583 | — |

## Unassigned (not in split.json) (2)

| # | Split | Match | Winner | Loser | Rallies | Shots | Skeleton |
|---|-------|-------|--------|-------|---------|-------|----------|
| 1 | REMOVED | `Anders_ANTONSEN_Jonatan_CHRISTIE Indonesia_Masters_2020_QuarterFinals` | Anders ANTONSEN | Jonatan CHRISTIE | 35 | 1018 | — |
| 2 | REMOVED | `Anthony_Sinisuka_Ginting_Rasmus_Gemke_YONEX_Thailand_Open_2021_QuarterFinals` | Anthony Sinisuka GINTING | Rasmus GEMKE | 40 | 1127 | — |
