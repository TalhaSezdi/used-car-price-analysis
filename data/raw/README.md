# data/raw

Bu klasor bosdur; ham veri (~1.4 GB) repoya dahil edilmez.

Projeyi calistirmak icin Craigslist ikinci el arac veri setini indirip
`vehicles.csv` dosyasini bu klasore koyun:

    data/raw/vehicles.csv

Kaynak: Kaggle "Used Cars Dataset" (Austin Reese) -- ~426,880 ilan, ABD, 2021.

Ardindan pipeline sirasiyla calisir (bkz. kok README.md):

    python scripts/clean_data.py   # data/raw/vehicles.csv -> data/processed/cleaned.parquet
