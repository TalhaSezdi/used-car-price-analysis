# İkinci El Araç Fiyat Tahmini ve Anomali Tespiti

Craigslist ilanlarından oluşan ~427 bin satırlık veri seti üzerinde uçtan uca bir
veri bilimi projesi: veri temizleme, keşifçi veri analizi, fiyat tahmini
(regresyon) ve şüpheli ilan tespiti.

## Genel bakış

Bir aracın özniteliklerinden (yaş, kilometre, marka/model, durum, çekiş, ...) ilan
fiyatını tahmin eden bir regresyon modeli ve bu modelin artıklarına dayanan bir
anomali tespit katmanı. Veri temizlemeden model değerlendirmesine kadar tüm mantık
`src/` altında modüler paketler halinde; her aşama `scripts/` altındaki bir giriş
noktasıyla çalıştırılır. Not defterleri yalnızca sunum içindir.

## Sonuçlar

Test seti: 39,563 ilan (%20, fiyat desiline göre tabakalı örnekleme). Metrikler
dolar ölçeğinde (log dönüşümü `expm1` ile geri alınmıştır).

| Model | RMSE ($) | MAE ($) | MAPE (%) | R2 |
|---|---|---|---|---|
| Linear Regression | 8,046 | 4,346 | 55.6 | 0.64 |
| Random Forest | 6,955 | 3,621 | 42.9 | 0.73 |
| LightGBM | 6,261 | 3,149 | 32.3 | 0.78 |

En iyi model LightGBM. Öznitelik önem sıralaması (gain tabanlı) beklentiyle uyumlu:
yaş (%39), model (%16), kilometre (%10). Fiyat segmenti, marka ve yaş bazında hata
analizi: [docs/phase3_results.md](docs/phase3_results.md).

Nokta tahminine ek olarak, conformal quantile regression ile ilan başına %90 tahmin
aralığı üretilir; marjinal kapsama %89.8, öznitelik-koşullu (Mondrian) kalibrasyonla
bin başına kapsama %89-91. Ayrıntı: [docs/phase6_results.md](docs/phase6_results.md).

## Öne çıkan bulgular

1. Değer kaybı öne yüklü: medyan fiyat 5. yılda %47, 10. yılda %72 düşüyor.
   ([figür](reports/figures/02_depreciation.png))
2. Kilometre en güçlü sayısal değişken (korelasyon -0.51) ama doğrusal değil;
   etkisi ~150 bin milden sonra düzleşiyor ve yaşla etkileşimli. Ağaç tabanlı
   modelin doğrusal modele göre %22 RMSE üstünlüğünün başlıca nedeni bu.
   ([figür](reports/figures/10_age_odometer_interaction.png))
3. Hedefin log dönüşümü ucuz araçlardaki yüzde hatayı belirgin biçimde azaltıyor:
   ham fiyat hedefi RMSE'de öndeyken ($5,765 vs $6,261) MAPE'de 13 puan geride
   (%45 vs %32). İlanların çoğu $20 binin altında olduğundan MAPE önceliklendirildi.
4. Tek değişkenli "primler" büyük ölçüde yaş kaynaklı: VIN'li ilanların 1.98x fiyat
   primi, yaş sabitlenince 1.29x'e iniyor.
   ([figür](reports/figures/09_confound_check.png))
5. En yüksek güvenli 49 şüpheli ilan, iki bağımsız sinyalin (fiyat artığı +
   Isolation Forest) kesişiminden geliyor. Tipik örüntü: ~$500'a listelenmiş 2-3
   yaşındaki pickup'lar ve $123,456 gibi klavye hatası fiyatlar.
   ([rapor](reports/suspicious_listings.csv))

## Yöntem

- **Temizleme:** fiyat, yıl, kilometre ve başlık durumu filtreleri; VIN ve parmak
  izi tabanlı tekilleştirme (426,880 -> 197,814 satır). Kurallar ve gerekçeler:
  [docs/cleaning_pipeline.md](docs/cleaning_pipeline.md).
- **Öznitelikler:** yaş (ilan yılı - model yılı), yıllık ortalama kilometre, log
  dönüşümleri ve açıklama metninden çıkarılan sızıntısız trim/donanım sinyalleri
  (yalnızca alfabetik eşleşme; metinden sayı okunmaz).
- **Kodlama:** yüksek kardinaliteli `model` için out-of-fold hedef kodlama; düşük
  kardinaliteli değişkenler için one-hot. Tüm kodlayıcılar yalnızca eğitim bölmesine
  fit edilir.
- **Modeller:** doğrusal regresyon (referans), Random Forest ve LightGBM; hedef
  `log1p(price)`.
- **Anomali:** 5 katlı OOF artıklarına dayalı MAD-z skoru ile yapısal Isolation
  Forest'ın kesişimi, katmanlı eşiklerle.

Hedef dönüşümü, kodlama stratejisi ve öznitelik kümesi gibi kararlar birer ablation
çalışmasıyla alternatifleriyle karşılaştırıldı. Sonuçlar
[docs/phase3_results.md](docs/phase3_results.md) ve
[docs/phase7_results.md](docs/phase7_results.md) içinde.

## Sınırlamalar ve varsayımlar

- Veri, Nisan-Mayıs 2021 arası ~30 günlük tek bir kesittir. Mevsimsellik modellenmez;
  her satır bağımsız bir ilan olduğundan rastgele train/test bölmesi kullanıldı.
  Araç yaşı gerçek tarihe göre değil, ilan tarihine göre hesaplanır.
- Anomali eşikleri istatistiksel değil operasyoneldir: artık dağılımı kalın kuyruklu
  (|z|>3.5 oranı Gaussian beklentisinin ~80 katı), dolayısıyla eşik bir kapasite
  tercihidir. MAPE ~%37 olduğundan orta seviye işaretler model hatası da olabilir;
  bu nedenle işaretler katmanlıdır. Ground-truth etiket olmadığından doğruluk metriği
  raporlanmaz.
- Tekilleştirme sonrası kalan yakın-kopyaların train/test sınırını geçme etkisi
  ölçüldü: test satırlarının %4.6'sında yakın kopya var, genel RMSE'ye etkisi +%0.6
  (kabul eşiğinin altında). Ölçüm: [docs/phase6_results.md](docs/phase6_results.md).
- Tahmin aralıkları öznitelik-koşullu kapsama garantisi verir; gerçek fiyata göre uç
  segmentlerde (50-150 bin) kapsama düşer. Bu bir kalibrasyon sınırı değil, nadir
  pahalı araçlarda nokta tahminin kendi sınırıdır.

## Kurulum ve çalıştırma

Ham veri (`vehicles.csv`, ~1.4 GB) repoda yer almaz. Kaggle "Used Cars Dataset"
indirilip `data/raw/vehicles.csv` olarak yerleştirilmelidir (bkz.
[data/raw/README.md](data/raw/README.md)).

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/clean_data.py         # ham veri -> data/processed/cleaned.parquet
python scripts/run_eda.py            # figurler + EDA raporu
python scripts/train.py             # 3 model + ablation calismalari
python scripts/detect_anomalies.py   # anomali skorlama -> reports/suspicious_listings.csv
python scripts/predict_intervals.py  # conformal tahmin araliklari
```

Tekrarlanabilirlik: Python 3.12, tüm rastgele işlemler için `random_state=42`,
[requirements.txt](requirements.txt) içinde sabitlenmiş paket sürümleri.

## Proje yapısı

```
data/          ham (repoda yok) ve islenmis veri
src/           preprocess, features, models, evaluation, anomaly paketleri
scripts/       calistirilabilir giris noktalari + ablation/probe betikleri
notebooks/     presentation.ipynb (sunum)
docs/          temizleme ve sonuc dokumanlari
reports/       figurler ve supheli ilan raporu
```

Dokümanlar: [veri denetimi](docs/phase1_audit.md) ·
[temizleme ve öznitelikler](docs/cleaning_pipeline.md) ·
[EDA](docs/phase2_insights.md) · [modelleme ve ablation](docs/phase3_results.md) ·
[anomali tespiti](docs/phase4_results.md) ·
[split bütünlüğü ve tahmin aralıkları](docs/phase6_results.md) ·
[öznitelik denemeleri](docs/phase7_results.md)
