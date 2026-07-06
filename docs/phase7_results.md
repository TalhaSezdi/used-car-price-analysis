# Phase 7 Results

## 7A. Junk / non-sale listing filter -- NOT IMPLEMENTED (stopped at the gate)

**Motivation:** Phase 6C showed the price model badly under-covers the 50-150k
segment (70.2% coverage vs 90% target). Manual inspection of the worst
under-predicted expensive listings suggested some are junk (non-vehicle items,
"wanted to buy" ads, commercial equipment) rather than missing trim signal.

**Method:** `scripts/probe_junk_rules.py` -- three candidate regex rules (buyer/
wanted ads, non-vehicle items, commercial-equipment vehicles), each measured for
match count and manually precision-checked on a 15-listing random sample, plus a
check of how much of the rules' combined reach actually falls inside the 50-150k
problem segment. Read-only: no pipeline code touched.

### Findings

| Rule | Matches | Share of data | Manual precision (n=15 sample) |
|---|---|---|---|
| 1. Wanted/buyer ads | 138 | 0.070% | ~40-45% -- most "flags" are false positives: sellers mentioning their OWN next purchase ("looking to buy a newer Prius") inside a genuine sale ad, not a non-sale signal |
| 2. Non-vehicle items (model missing + no vehicle vocab) | 265 | 0.134% | Low -- mostly genuine but tersely-worded real vehicle ads, not the "$60k kitchen equipment"-class junk that motivated the rule |
| 3. Commercial equipment vehicles | 1,529 | 0.773% | ~50% -- roughly half are genuine commercial vehicles (dump trucks, box trucks), half are ordinary consumer cars (Grand Prix, Malibu) caught by boilerplate dealer SEO text mentioning "box truck" somewhere in a long multi-vehicle inventory template |
| **Any rule (union)** | **1,918** | **0.970%** | mixed, see above |

**Decisive number -- reach into the target segment:** of the 4,627 listings in the
50-150k price segment (the one 6C flagged as under-covered), only **50 rows
(1.08%)** are caught by any of the three rules combined.

### Decision: do not implement

Two independent problems, either one alone would be disqualifying:

1. **Reach is too small to matter.** Even with perfect precision, removing 50 out
   of 4,627 rows in the target segment cannot materially move that segment's
   RMSE/MAPE/coverage. The rules do not address the scale of the 6C gap.
2. **Precision is weak on top of that.** Rule 1 and Rule 2 are dominated by false
   positives in the reviewed sample; Rule 3 is contaminated by dealer-template
   boilerplate matching keywords anywhere in a long inventory description rather
   than describing the actual listed vehicle.

Per the Phase 7 plan's gate step, no rule is implemented in `src/preprocess/cleaner.py`
and `clean_data.py` is not rerun. This is a negative result, recorded honestly:
the recon that motivated 7A (a handful of manually-inspected bad listings) does
not generalize into a rule with enough reach or precision to justify a cleaning
change. `scripts/probe_junk_rules.py` is kept in the repo as a reusable check if a
future, larger, or differently-sourced dataset shows a bigger junk-listing problem.

---

## 7B. Description trim/equipment features

**Method:** `src/features/description.py::DescriptionFeatureExtractor` adds `desc_trim_luxury` (0/1, curated trim-name keyword match), `desc_equip_count` (count of equipment keywords), `desc_len_log` (log1p description length). Stateless, row-wise, alphabetic-keyword-only (no digit extraction) -- computed in `FeatureEngineer` before the raw `description` column is dropped; the processed parquet never contains raw text. Ablation A4 (`scripts/ablation_description_features.py`) trains the final LightGBM pipeline with vs without these 3 columns, same train/test split (Phase 3/6A), same encoding pipeline.

### Ablation A4: baseline vs with desc_* features

| Model                |   RMSE ($) |   MAE ($) |   MAPE (%) |   R2 |
|:---------------------|-----------:|----------:|-----------:|-----:|
| baseline (no desc_*) |    6591.16 |   3349.24 |      36.93 | 0.76 |
| with desc_* features |    6252.5  |   3142.81 |      32.4  | 0.78 |

### Gain share of desc_* features (with-desc model)

|                  |     0 |
|:-----------------|------:|
| desc_len_log     | 5.764 |
| desc_equip_count | 1.501 |
| desc_trim_luxury | 1.076 |

### Error by price segment (with-desc model)

| segment   |      MAE |   MAPE |   count |
|:----------|---------:|-------:|--------:|
| 50-150k   | 20592.7  |  28.7  |     929 |
| 20-50k    |  5213.55 |  16.89 |    9191 |
| 10-20k    |  2603.24 |  17.5  |   11327 |
| <5k       |  1534.85 |  89.26 |    7738 |
| 5-10k     |  1534.69 |  20.33 |   10378 |

**Verdict: real improvement.** RMSE drops 5.14% ($6,591 -> $6,253), MAPE improves (36.93% -> 32.40%). desc_* features carry 8.34% of total gain -- a real, if modest, signal. Recommendation: adopt as default features (separate follow-up step, requires re-rippling Phase 3/6 metrics).
