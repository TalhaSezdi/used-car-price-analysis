# Phase 2 Insights — EDA

Dataset: 197,814 cleaned listings.

## Headline findings (each maps to a figure in reports/figures/)

### 1. Price is right-skewed -> log target is correct
- Raw price skew: 2.06; log1p(price) skew: -0.43.
- Business meaning: modeling raw price would let a handful of expensive listings dominate the loss. Training on log price treats a $2k error on a $5k car and a $20k car proportionally.

### 2. Depreciation is front-loaded
- Market median price: age 1 = $35,592, age 5 = $18,980, age 10 = $9,990.
- A car loses roughly 47% of value by year 5 and 72% by year 10.
- Business meaning: the first few years carry the most pricing risk; accurate age handling matters most there.

### 3. Mileage drives price, but non-linearly
- Correlation price~odometer: -0.51 (strongest single numeric driver).
- The trend line drops steeply then flattens after ~150k miles.
- Business meaning: below ~150k miles each mile costs real money; above it, mileage barely moves price.

### 4. mileage_per_year is a weak linear signal
- Correlation price~mileage_per_year: 0.07 (near zero).
- Business meaning: raw mileage and age each carry more signal than their ratio; keep the ratio only if a tree model finds interactions, don't rely on it linearly.

### 5. Strong regional price spread
- Highest-median state: ak ($21,999); lowest: ri ($7,225).
- Spread: 3.04x between the priciest and cheapest state medians.
- Business meaning: location is a real pricing feature (arbitrage opportunity for a marketplace), not noise.

### 6. Brand tiers are clear
- Cheapest volume brand median: honda ($8,200); across all brands, luxury marques sit well above mass-market ones.
- Business meaning: `manufacturer` (and `model`) are high-value categorical features; worth target encoding.

## Experimental / behavioral findings

### 7. Value heaping (figure 08)
- 15% of prices end in 995 and 10% end in 000 -> psychological pricing anchors.
- 30% of odometer readings are an exact round thousand -> sellers round their mileage.
- Business meaning: odometer is not a precise continuous measurement; treat round-number heaping as noise, and expect list prices to cluster at .995/.999 rather than vary smoothly.

### 8. Confound warning: several 'premiums' are mostly age (figure 09)
- Raw VIN premium: 1.98x, but within the 11-15yr bucket it collapses to 1.29x.
- Raw missing-condition premium: 1.44x, but within the 11-15yr bucket it is only 1.09x -- it almost entirely reflects newer cars omitting the field.
- Business meaning: univariate category 'premiums' (VIN, missing condition, and likely 4wd/diesel/color) are heavily confounded with age and body type. This is the core argument for a multivariate model over single-feature rules of thumb.

### 9. Age x odometer interact (figure 10)
- Median price over the 2D age x odometer grid falls along both axes jointly, not additively; a low-mileage old car and a high-mileage new car are priced very differently.
- Business meaning: tree models (which capture interactions natively) should beat a purely additive linear model here; mileage_per_year was a weak proxy for exactly this interaction.

### 10. Structured missingness + extreme model cardinality (figure 11)
- Missing fields co-occur (drive/type/paint_color/cylinders/size, phi up to ~0.5): missingness is NOT random.
- `model` has 20,638 unique values and 82% appear in fewer than 5 listings.
- Business meaning (Phase 3): impute with a 'missing' indicator rather than silent fill; and do NOT naive target-encode 20k models -- use smoothing / collapse the rare long tail to avoid overfitting.

## Quant caveats (methodological honesty)

- **Perfect collinearity:** corr(age, year) = -1.00 by construction. Use only ONE of them in the linear baseline; keeping both breaks coefficient interpretation.
- **State spread is confounded:** the highest-median state (AK) is 58% 4wd vs 32% market-wide. The 3x 'regional' spread partly reflects a heavier truck/4wd mix, not pure geography -- same confound lesson as finding 8, applied to location.
- **Single-month snapshot:** all listings span just 30 days (Apr-May 2021). No seasonal/temporal modeling is possible or claimed; this also justifies a random (non-temporal) train/test split in Phase 3.