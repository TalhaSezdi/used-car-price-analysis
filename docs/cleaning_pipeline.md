# Cleaning Pipeline & Feature Engineering

Documents every rule applied to turn the raw ~426,880-row Craigslist dump into
the modeling dataset. Implemented as OOP classes in `src/preprocess/cleaner.py`
(`DataCleaner`) and `src/features/engineer.py` (`FeatureEngineer`); the entry
point is `scripts/clean_data.py`. The raw file is never modified in place --
cleaning always writes to `data/processed/cleaned.parquet`.

## Retention funnel

| Step | Rule | Rows removed | Rows remaining |
|---|---|---:|---:|
| Start | raw dataset | -- | 426,880 |
| Price filter | drop `price < 500` or `price > 150,000` | 42,290 | 384,590 |
| Year filter | drop `year < 1970` or `year > 2022` | 6,171 | 378,419 |
| Odometer filter | drop `odometer <= 0` or `odometer > 500,000` | 3,918 | 374,501 |
| Title status | keep only `clean`, `rebuilt`, or null | 5,636 | 368,865 |
| Deduplication | VIN exact + no-VIN fingerprint | 163,683 | 205,182 |
| Core null drop | drop rows null in price/year/odometer/manufacturer | 7,368 | 197,814 |

**Final: 197,814 rows (46.34% retention).** Aggressive, but every drop is a
documented rule -- what we drop matters more than how much.

## Rules and rationale

**Dropped columns (non-predictive):** `id`, `url`, `region_url`, `image_url`,
`county`. These carry no generalizable signal; `id`/`url` are unique per row and
`county` is 100% empty. `VIN` is used for deduplication first, then dropped
before modeling.

**Price filter [500, 150,000].** Raw `price` contains junk: a large mass at 0-1
(placeholder listings) and absurd highs (max is 3.7 billion -- a keyboard error).
The floor removes non-serious listings; the ceiling removes data-entry errors and
puts exotic/commercial vehicles (a different pricing regime) out of scope.

**Year filter [1970, 2022].** Data was collected around 2021, so a `year` beyond
2022 is impossible. Pre-1970 vehicles are a separate collector market with
different price dynamics.

**Odometer filter [1, 500,000].** An odometer of 0 on a used car is implausible;
above 500,000 miles is almost always a data-entry error.

**Title status.** Keep `clean`, `rebuilt`, and null; drop `salvage`, `lien`,
`missing`, `parts only`. Salvage/parts vehicles price on a different regime and
would add noise to a market-value model.

**Deduplication (largest single step -- 163,683 rows).** `id` is unique per
posting, so real duplicates are the same physical car re-posted. Removed in two
passes: (1) exact `VIN` match where VIN is present (130,857 rows); (2) for
rows without a VIN, a `(manufacturer, model, price, odometer)` fingerprint
(32,826 rows). Fingerprint dedup is restricted to no-VIN rows so distinct VINs
are never collapsed. This matters for split integrity: if the same car lands in
both train and test, reported metrics inflate (measured directly in
`docs/phase6_results.md`, Phase 6A).

**Core null drop.** Rows still missing any of `price`, `year`, `odometer`,
`manufacturer` are dropped -- these four are indispensable for a price model.
For lower-priority fields (condition, drive, paint_color, ...) missingness is
kept as its own signal (a "missing" category) rather than dropped;
`paint_color_missing` and `drive_missing` both surface in feature importance.

## Engineered features

- **`age` = `posting_year` - `year`.** Computed relative to the posting date,
  NEVER today's real-world year. Using the current year would silently corrupt
  the feature by ~5 years on 2021-era data. `posting_year` is extracted from
  `posting_date`.
- **`mileage_per_year` = `odometer` / `age`** (age clipped to >= 1 to avoid
  division by zero). Separates a low-mileage old car from a high-mileage new one.
- **`log_odometer` = `log1p(odometer)`.** Tames the right skew of raw mileage.
- **`log_price` = `log1p(price)`.** The model target (skew 2.06 -> -0.43).
  Metrics are inverted with `expm1` and reported in dollars -- never in log space.
  See Ablation A1 in `docs/phase3_results.md` for why log over raw.
- **Description-derived features** (`desc_trim_luxury`, `desc_equip_count`,
  `desc_len_log`): leakage-free keyword signal extracted from the listing text.
  Alphabetic matching only -- no digits are ever read from the description, so the
  price/MSRP that listings often contain cannot leak in. See
  `docs/phase7_results.md` (Phase 7B, Ablation A4).

## Encoding (applied inside the model pipeline, fit on train only)

- **High cardinality (`model`, thousands of values):** out-of-fold KFold target
  encoding -- a row is never encoded with its own label. See Ablation A3.
- **Low cardinality (`condition`, `fuel`, `transmission`, `drive`, `type`,
  `title_status`, `cylinders`, `paint_color`, `state`):** one-hot, with a
  missing-indicator where appropriate.

All imputers/encoders are fit on the training split only and applied to the test
split, so no test information leaks into feature construction.
