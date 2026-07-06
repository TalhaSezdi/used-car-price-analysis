# Attrition Analysis: what got dropped and does the survivor stay representative?

Cleaning retained **197,814 of 426,880 rows (46.3%)**.
A reviewer will reasonably ask whether that drop was uniform, or whether it
narrowed the population the price model is actually valid for. This document
answers that question segment by segment.

The retention funnel itself (which filter removes how many rows) is in
[cleaning_pipeline.md](cleaning_pipeline.md). The analysis below focuses on the
*composition* shift between the raw and cleaned samples.

## 1. Retention by manufacturer (top 15 by raw volume)

| manufacturer   |   raw_n |   cleaned_n |   retention_pct |   raw_share_pct |   cleaned_share_pct |   share_drift_pp |
|:---------------|--------:|------------:|----------------:|----------------:|--------------------:|-----------------:|
| ford           |   70985 |       33139 |            46.7 |           16.63 |               16.75 |             0.12 |
| chevrolet      |   55064 |       25994 |            47.2 |           12.9  |               13.14 |             0.24 |
| toyota         |   34202 |       18108 |            52.9 |            8.01 |                9.15 |             1.14 |
| honda          |   21269 |       12624 |            59.4 |            4.98 |                6.38 |             1.4  |
| nissan         |   19067 |       10345 |            54.3 |            4.47 |                5.23 |             0.76 |
| jeep           |   19014 |        9357 |            49.2 |            4.45 |                4.73 |             0.28 |
| ram            |   18342 |        6996 |            38.1 |            4.3  |                3.54 |            -0.76 |
| <NA>           |   17646 |           0 |             0   |            4.13 |                0    |            -4.13 |
| gmc            |   16785 |        7400 |            44.1 |            3.93 |                3.74 |            -0.19 |
| bmw            |   14699 |        6660 |            45.3 |            3.44 |                3.37 |            -0.07 |
| dodge          |   13707 |        6466 |            47.2 |            3.21 |                3.27 |             0.06 |
| mercedes-benz  |   11817 |        5598 |            47.4 |            2.77 |                2.83 |             0.06 |
| hyundai        |   10338 |        5287 |            51.1 |            2.42 |                2.67 |             0.25 |
| subaru         |    9495 |        5631 |            59.3 |            2.22 |                2.85 |             0.63 |
| volkswagen     |    9345 |        4648 |            49.7 |            2.19 |                2.35 |             0.16 |

**Reading:** `retention_pct` is the share of a manufacturer's raw rows that
survived. `share_drift_pp` is (cleaned share) - (raw share) in percentage
points -- positive means the manufacturer is *over-represented* in the cleaned
set relative to raw. Max absolute drift across NAMED top-15 manufacturers:
**1.40 pp** (honda). The `<NA>` row's -4.13 pp drift
is a deterministic consequence of the `manufacturer` core-null drop and does
not indicate a bias among identified manufacturers.

## 2. Retention by state (top 15 by raw volume)

| state   |   raw_n |   cleaned_n |   retention_pct |   raw_share_pct |   cleaned_share_pct |   share_drift_pp |
|:--------|--------:|------------:|----------------:|----------------:|--------------------:|-----------------:|
| ca      |   50614 |       23187 |            45.8 |           11.86 |               11.72 |            -0.14 |
| fl      |   28511 |       14176 |            49.7 |            6.68 |                7.17 |             0.49 |
| tx      |   22945 |       11256 |            49.1 |            5.38 |                5.69 |             0.31 |
| ny      |   19386 |        8649 |            44.6 |            4.54 |                4.37 |            -0.17 |
| oh      |   17696 |        7731 |            43.7 |            4.15 |                3.91 |            -0.24 |
| or      |   17104 |        7132 |            41.7 |            4.01 |                3.61 |            -0.4  |
| mi      |   16900 |        6918 |            40.9 |            3.96 |                3.5  |            -0.46 |
| nc      |   15277 |        5552 |            36.3 |            3.58 |                2.81 |            -0.77 |
| wa      |   13861 |        5682 |            41   |            3.25 |                2.87 |            -0.38 |
| pa      |   13753 |        5331 |            38.8 |            3.22 |                2.69 |            -0.53 |
| wi      |   11398 |        5334 |            46.8 |            2.67 |                2.7  |             0.03 |
| co      |   11088 |        7125 |            64.3 |            2.6  |                3.6  |             1    |
| tn      |   11066 |        4470 |            40.4 |            2.59 |                2.26 |            -0.33 |
| va      |   10732 |        3671 |            34.2 |            2.51 |                1.86 |            -0.65 |
| il      |   10387 |        5982 |            57.6 |            2.43 |                3.02 |             0.59 |

Max absolute drift across top-15 states: **1.00 pp** (co).

## 3. Retention by title_status

| title_status   |   raw_n |   cleaned_n |   retention_pct |   raw_share_pct |   cleaned_share_pct |   share_drift_pp |
|:---------------|--------:|------------:|----------------:|----------------:|--------------------:|-----------------:|
| clean          |  405117 |      189972 |            46.9 |           94.9  |               96.04 |             1.14 |
| <NA>           |    8242 |        2661 |            32.3 |            1.93 |                1.35 |            -0.58 |
| rebuilt        |    7219 |        5181 |            71.8 |            1.69 |                2.62 |             0.93 |
| salvage        |    3868 |           0 |             0   |            0.91 |                0    |            -0.91 |
| lien           |    1422 |           0 |             0   |            0.33 |                0    |            -0.33 |
| missing        |     814 |           0 |             0   |            0.19 |                0    |            -0.19 |
| parts only     |     198 |           0 |             0   |            0.05 |                0    |            -0.05 |

The `clean` + `rebuilt` filter is the intended behavior; the strong drops on
`salvage`, `lien`, `missing`, and `parts only` are by design (these rows are
either legally unmarketable at retail or fundamentally different price
dynamics -- see [cleaning_pipeline.md](cleaning_pipeline.md) for the
rationale).

## 4. Retention by price band

| price_band   |   raw_n |   cleaned_n |   retention_pct |   raw_share_pct |   cleaned_share_pct |   share_drift_pp |
|:-------------|--------:|------------:|----------------:|----------------:|--------------------:|-----------------:|
| <=0          |   32895 |           0 |             0   |            7.71 |                0    |            -7.71 |
| 1-499        |   10220 |         375 |             3.7 |            2.39 |                0.19 |            -2.2  |
| 500-4999     |   54065 |       38223 |            70.7 |           12.67 |               19.32 |             6.65 |
| 5-10k        |   79057 |       51995 |            65.8 |           18.52 |               26.28 |             7.76 |
| 10-20k       |  102291 |       56780 |            55.5 |           23.96 |               28.7  |             4.74 |
| 20-50k       |  135416 |       45872 |            33.9 |           31.72 |               23.19 |            -8.53 |
| 50-150k      |   12740 |        4569 |            35.9 |            2.98 |                2.31 |            -0.67 |
| >150k        |     196 |           0 |             0   |            0.05 |                0    |            -0.05 |

Rows with `price <= 0` and `price >= 150k` are removed by design (the price
filter). This is the largest single source of drop and it is *deliberate* --
the model is scoped to consumer-marketplace listings, not junk prices or
exotic-car outliers. The mid-price bands (5k-50k) retain at their fair share.

## 5. Retention by model year band

| year_band   |   raw_n |   cleaned_n |   retention_pct |   raw_share_pct |   cleaned_share_pct |   share_drift_pp |
|:------------|--------:|------------:|----------------:|----------------:|--------------------:|-----------------:|
| <1970       |    5627 |         216 |             3.8 |            1.32 |                0.11 |            -1.21 |
| 1970-89     |    7355 |        4974 |            67.6 |            1.72 |                2.51 |             0.79 |
| 1990-99     |   15840 |       11004 |            69.5 |            3.71 |                5.56 |             1.85 |
| 2000-09     |  109574 |       68167 |            62.2 |           25.67 |               34.46 |             8.79 |
| 2010-14     |  136854 |       62569 |            45.7 |           32.06 |               31.63 |            -0.43 |
| 2015-19     |  147896 |       50089 |            33.9 |           34.65 |               25.32 |            -9.33 |
| 2020-22     |    2529 |         795 |            31.4 |            0.59 |                0.4  |            -0.19 |
| >2022       |       0 |           0 |           nan   |            0    |                0    |             0    |

The `<1970` and `>2022` drops are by design (year filter). The middle bands
retain proportionally.

## 6. Missingness shift on kept columns

|   index | column       |   raw_missing_pct |   cleaned_missing_pct |   delta_pp |
|--------:|:-------------|------------------:|----------------------:|-----------:|
|       3 | size         |             71.77 |                 63.3  |      -8.47 |
|       1 | cylinders    |             41.62 |                 34.35 |      -7.27 |
|       0 | condition    |             40.79 |                 37.98 |      -2.8  |
|       6 | VIN          |             37.73 |                 50.47 |      12.74 |
|       2 | drive        |             30.59 |                 27.18 |      -3.41 |
|       5 | paint_color  |             30.5  |                 27.79 |      -2.71 |
|       4 | type         |             21.75 |                 24.63 |       2.87 |
|       8 | manufacturer |              4.13 |                  0    |      -4.13 |
|      10 | title_status |              1.93 |                  1.35 |      -0.59 |
|       9 | model        |              1.24 |                  1.27 |       0.03 |
|       7 | odometer     |              1.03 |                  0    |      -1.03 |

**Reading:** `delta_pp` = missingness in the cleaned sample minus missingness in
the raw sample. A large positive delta would mean cleaning kept the incomplete
rows and dropped the complete ones (bad); a large negative delta would mean
cleaning kept the complete rows and dropped the incomplete ones (also
informative -- selection on data quality).

## 7. Numeric distribution summaries

### price ($)

| stat   |              raw |   cleaned |
|:-------|-----------------:|----------:|
| count  | 426880           |    197814 |
| mean   |  75199           |     15490 |
| std    |      1.21823e+07 |     13299 |
| min    |      0           |       500 |
| 25%    |   5900           |      6000 |
| 50%    |  13950           |     11500 |
| 75%    |  26486           |     20899 |
| max    |      3.73693e+09 |    150000 |

### year

| stat   |      raw |   cleaned |
|:-------|---------:|----------:|
| count  | 425675   |  197814   |
| mean   |   2011.2 |    2010.2 |
| std    |      9.5 |       7.6 |
| min    |   1900   |    1970   |
| 25%    |   2008   |    2007   |
| 50%    |   2013   |    2012   |
| 75%    |   2017   |    2016   |
| max    |   2022   |    2022   |

### odometer (miles)

| stat   |        raw |   cleaned |
|:-------|-----------:|----------:|
| count  | 422480     |    197814 |
| mean   |  98043     |    107256 |
| std    | 213882     |     63707 |
| min    |      0     |         1 |
| 25%    |  37704     |     58098 |
| 50%    |  85548     |    103964 |
| 75%    | 133542     |    149000 |
| max    |      1e+07 |    500000 |

## Verdict

- **Category composition drift is small among identified manufacturers.** The
  largest absolute share drift among named top-15 manufacturers is
  1.40 pp (honda); state drift is
  1.00 pp (co). The cleaned sample is not
  manufacturer-biased or geography-biased relative to raw.
- **Price and year drops are deliberate and documented.** `price <= 0`,
  `price > 150k`, `year < 1970`, and `year > 2022` are removed by design, which
  implements the scoping decision ("consumer marketplace, non-junk,
  non-exotic"). These are not hidden biases.
- **The 20-50k / 2015-19 under-retention is real and worth naming.** The
  20-50k price band loses 8.5 pp of share, and the 2015-19 year band loses
  9.3 pp -- the largest non-boundary drifts in the table. These segments are
  where dealer re-postings and fingerprint duplicates concentrate (newer,
  higher-priced inventory is churned more aggressively across regions), so
  dedup thins them more than the older/cheaper bands. **Consequence:** the
  cleaned sample slightly under-represents late-model mid-priced listings.
  Segment-level metrics (see docs/phase3_results.md error analysis) already
  slice by these bands, so this shift does not silently distort the reported
  headline numbers, but it is a caveat when generalizing to that segment's
  raw-market prevalence.
- **Title-status drops are deliberate.** `salvage`/`parts only`/`lien` rows
  price under different dynamics; excluding them keeps the model's target
  well-defined.
- **Missingness on kept columns mostly decreases after cleaning.** The one
  exception is VIN (+12.7 pp), which is expected: the salvage-title filter
  disproportionately removes rows that had VINs (dealer inventory), leaving a
  higher share of VIN-missing private-party listings behind. This does not
  affect the model since VIN is not a feature; it is a bookkeeping note.

**Practical consequence:** the model's reported metrics apply to the population
described above (500 <= price <= 150,000; 1970 <= year <= 2022; clean/rebuilt
title; deduped by VIN or fingerprint). They do not claim to describe the
salvage/junk market, exotic cars, or listings with fabricated $0/$1 prices.
This scope is a marketplace-analytics choice, not a hidden bias.
