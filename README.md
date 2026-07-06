# İkinci El Araç Piyasa Analizi ve Fiyat Tahmini

Craigslist'ten toplanmış ~427 bin ikinci el araç ilanı üzerinde uçtan uca bir analiz projesi:
veri temizleme, iş odaklı EDA, fiyat tahmin modeli (regresyon) ve şüpheli ilan tespiti (anomali).
Projenin ayırt edici tarafı metodoloji: **her karar noktasında alternatif de denendi ve sonuç
sayıyla belgelendi** — "neden X, Y değil?" sorusunun cevabı iddia değil, ablation.

## Nasıl çalıştırılır

Veri: ham `vehicles.csv` (~1.4 GB) repoya dahil degildir. Kaggle "Used Cars
Dataset" (Craigslist, ~426,880 ilan) indirilip `data/raw/vehicles.csv` olarak
konulmalidir (bkz. [data/raw/README.md](data/raw/README.md)).

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt

# Sirasiyla (her adim bir sonrakinin girdisini uretir):
python scripts/clean_data.py          # data/raw -> data/processed/cleaned.parquet (desc_* dahil)
python scripts/run_eda.py             # 11 figur + docs/phase2_insights.md
python scripts/train.py               # 3 model + 3 ablation -> docs/phase3_results.md
python scripts/detect_anomalies.py    # anomali skorlama -> reports/suspicious_listings.csv
python scripts/predict_intervals.py   # conformal tahmin araligi -> docs/phase6_results.md
```

Tekrarlanabilirlik: Python 3.12; tüm rastgele işlemler `random_state=42`; bağımlılıklar,
raporlanan metrikleri üreten ortamın birebir sürümleriyle [requirements.txt](requirements.txt)
içinde pinli.

## Nihai model karşılaştırması

Test seti: 39,563 ilan (%20, fiyat desiline göre stratified). Metrikler dolar ölçeğinde
(`expm1` ile log'dan geri çevrilmiş).

| Model | RMSE ($) | MAE ($) | MAPE (%) | R2 |
|---|---|---|---|---|
| Linear Regression | 8,046 | 4,346 | 55.6 | 0.64 |
| Random Forest | 6,955 | 3,621 | 42.9 | 0.73 |
| **LightGBM** | **6,261** | **3,149** | **32.3** | **0.78** |

Gain bazlı feature importance beklentiyle tutarlı: `age` %39 > `model` %16 > `odometer` %10 >
`desc_len_log` %6 (Phase 7B, aşağıya bakın). Detay ve hata analizi (fiyat segmenti / marka /
yaş bazında): [docs/phase3_results.md](docs/phase3_results.md)

## Tahmin aralıkları (conformal prediction)

Nokta tahmine ek olarak, split-conformal quantile regression (CQR) ile her ilan için
%90 güven aralığı üretildi (`scripts/predict_intervals.py`). Genel (marjinal) kapsama
**%89.8**; segment-koşullu (Mondrian, tahmin-bazlı 5 bin) kalibrasyonla bin başına
kapsama da **%89-91** — model kendi verdiği garantiyi tutuyor.

Dürüst kısıt: **gerçek** fiyata göre dilimlenince uçlar hedefin altında kalıyor
(50-150k segmenti ~%70). Bunun kalibrasyonla düzeltilemeyeceği probe ile kanıtlandı
(`scripts/probe_mondrian_conditional_coverage.py`): kaçırılan pahalı ilanlar, modelin
~$30k tahmin ettiği ama gerçekte ~$65k olan nadir trim'ler — sorun aralığın genişliği
değil, nokta tahminin kendisi. Outcome'a koşullu kapsama zaten teorik olarak garanti
edilemez (Foygel Barber et al. 2021); ulaşılabilir ve teslim edilen garanti
feature-koşullu olan. Detay: [docs/phase6_results.md](docs/phase6_results.md).

## Öne çıkan 5 bulgu

1. **Değer kaybı öne yüklü:** medyan fiyat 5. yılda %47, 10. yılda %72 düşüyor. Fiyatlama
   riskinin en yüksek olduğu yer ilk yıllar — [figür 02](reports/figures/02_depreciation.png).
2. **Kilometre en güçlü tekil sayısal sürücü (corr -0.51), ama doğrusal değil:** ~150 bin milden
   sonra etkisi düzleşiyor. Yaş ile kilometre etkileşimli (toplamsal değil) — bu, ağaç bazlı
   modelin lineer modeli %22 RMSE farkıyla geçmesinin nedeni —
   [figür 03](reports/figures/03_odometer_vs_price.png), [figür 10](reports/figures/10_age_odometer_interaction.png).
3. **Log hedef ticari olarak doğru seçim:** ham fiyat hedefi RMSE'de kazanıyor ($5,765 vs $6,261)
   ama MAPE'de 13 puan kaybediyor (%45 vs %32) — ucuz araçları sistematik olarak yanlış tahmin
   ediyor. İlanların çoğu $20k altında olduğu için MAPE'yi önceledik (Ablation A1).
4. **Tek değişkenli "prim"ler çoğunlukla yaş yanılsaması:** VIN'li ilanların 1.98x fiyat primi,
   yaş kontrol edilince 1.29x'e düşüyor. Tek-özellik kurallar yerine çok değişkenli model
   gerekliliğinin kanıtı — [figür 09](reports/figures/09_confound_check.png).
5. **Güven düzeyi en yüksek 53 şüpheli ilan:** birbirinden bağımsız iki sinyal (fiyat
   residual'ı + Isolation Forest) aynı anda uyarıyor. Tipik profil: ~$500'a listelenmiş 2-3
   yaşında pickup'lar (placeholder/scam) ve $123,456 gibi klavye-hatası fiyatlar —
   [docs/phase4_results.md](docs/phase4_results.md), [reports/suspicious_listings.csv](reports/suspicious_listings.csv).

## Metodolojik dürüstlük notları

Bir inceleyicinin soracağı zor sorular proje içinde cevaplı:

- **Leakage:** target encoding out-of-fold (satır kendi etiketini asla görmüyor); early stopping
  test setine değil train içinden ayrılan validation'a bakıyor; anomali residual'ları 5-fold OOF
  tahminlerden. Self-review sırasında yakalanan encoder sızıntısı, A3 ablation'ının sonucunu
  tersine çevirmişti — düzeltme ve etkisi [docs/phase3_results.md](docs/phase3_results.md) içinde belgeli.
- **Anomali eşiği istatistiksel değil operasyonel:** residual dağılımı kalın kuyruklu
  (|z|>3.5 oranı Gaussian beklentisinin ~80 katı). MAPE ~%37 iken orta seviye flag'lerin bir
  kısmı model hatası olabilir — flag'ler bu yüzden katmanlı (STRONG / MODERATE / structural).
- **Zaman boyutu yok:** veri 30 günlük tek fotoğraf (Nis-May 2021). Mevsimsellik iddiası yok;
  random split bu yüzden meşru. Araç yaşı bugüne göre değil, ilan tarihine göre hesaplı.
- **Split kirliliği ölçüldü, materyal değil:** yeniden ilan edilmiş araçların train/test
  sınırını geçme olasılığı doğrudan probe edildi (`scripts/probe_split_leakage.py`) — test
  satırlarının %4.6'sının train'de yakın-kopyası var, ama bunun genel RMSE'ye etkisi +%0.6 —
  eşiğin altında, düzeltme yapılmadı. Detay: [docs/phase6_results.md](docs/phase6_results.md).
- **Junk-ilan filtresi denendi, uygulanmadı (Phase 7A):** description'da "wanted to buy",
  ticari ekipman gibi kalıplar arandı; hem isabet zayıf çıktı hem de hedef segmentin
  (%50-150k) sadece %1'ine ulaşıyordu — kapıda durduruldu, veri temizlenmedi. Negatif
  sonuç olarak belgeli: [docs/phase7_results.md](docs/phase7_results.md).
- **Description'dan leakage-free trim özellikleri gerçek kazanç verdi (Phase 7B):** trim/
  donanım anahtar kelimeleri + ilan uzunluğu (sadece alfabetik eşleşme, rakam çıkarımı yok)
  RMSE'yi %5.3, MAPE'yi 4.6 puan iyileştirdi (Ablation A4) — default özellik setine
  eklendi, tüm rapor sayıları bu haliyle güncel. Detay: [docs/phase7_results.md](docs/phase7_results.md).

## Proje yapısı

```
data/raw/          vehicles.csv buraya konur (repoda yok; ~1.4 GB)
data/processed/    cleaned.parquet (clean_data.py uretir; 426,880 -> 197,814 satir)
src/               preprocess / features / models / evaluation / anomaly paketleri
scripts/           clean_data.py, run_eda.py, train.py, detect_anomalies.py,
                   predict_intervals.py + probe/ablation + check_consistency.py
notebooks/         presentation.ipynb (hikaye; is mantigi degil)
docs/              temizleme + sonuc dokumanlari (audit, cleaning, phase2/3/4/6/7)
reports/           13 figur + suspicious_listings.csv + presentation.pptx
```

Derinlik için faz dokümanları: [ham veri denetimi](docs/phase1_audit.md) ·
[temizleme kuralları + feature engineering](docs/cleaning_pipeline.md) ·
[EDA içgörüleri](docs/phase2_insights.md) · [model + ablation'lar](docs/phase3_results.md) ·
[anomali](docs/phase4_results.md) · [split integrity + tahmin aralıkları](docs/phase6_results.md) ·
[junk filtresi + description özellikleri](docs/phase7_results.md)
