# İkinci El Araç Fiyatlama ve Anomali Tespiti

426.880 Craigslist ilanını temizleme, keşifçi veri analizi, fiyat tahmini,
tahmin aralığı üretimi ve şüpheli ilan tespitinden geçiren uçtan uca bir veri
bilimi projesi. Modüler iş mantığı `src/` altında, çalıştırılabilir iş akışları
`scripts/` altında tutulur; notebook yalnızca sunum katmanıdır.

## Akış

```text
data/raw/vehicles.csv (426,880 ilan)
  -> DataCleaner + FeatureEngineer
  -> data/processed/cleaned.parquet (197,814 ilan)
       |-> EDA -> 13 figure + business insights
       |-> 60/20/20 stratified split
       |     -> preprocessing -> Linear / Random Forest / LightGBM
       |     -> final LightGBM -> point estimate + conformal interval
       `-> 5-fold OOF predictions -> MAD-z + Isolation Forest
                                      -> suspicious_listings.csv
```

## Teknik yaklaşım

| Aşama | Uygulama |
|---|---|
| Temizleme | Fiyat `[500, 150000]`, yıl `[1970, 2022]` ve kilometre `[1, 500000]` filtreleri; VIN ve yalnızca VIN'siz kayıtlarda parmak izi ile tekilleştirme. Ham veri değiştirilmez. |
| Özellik üretimi | `age = posting_year - year`, yıllık kilometre, log dönüşümleri, alfabetik trim/donanım eşleşmeleri ve metin uzunluğu. Açıklamadan sayısal değer çıkarılmaz; ham metin modele girmez. |
| Ön işleme | Sayısal alanlarda medyan + eksiklik göstergesi; düşük kardinalitede one-hot; `model` alanında smoothing içeren 5-fold OOF target encoding. Fit edilen tüm dönüşümler yalnızca eğitim verisini kullanır. |
| Doğrulama | Fiyat desiline göre sabit tohumlu (`random_state=42`) %60 train / %20 validation / %20 test. Model karşılaştırması ile A1-A3 validation'da yürütülür; A4 için aşağıdaki sınırlamaya bakın. |
| Modelleme | `log1p(price)` hedefi üzerinde Linear Regression, Random Forest ve LightGBM. Metrikler `expm1` sonrası dolar ölçeğinde hesaplanır. |
| Belirsizlik | LightGBM quantile modelleri ve ayrı calibration kümesiyle conformalized quantile regression; Mondrian varyantında tahmin bandı orta noktasına göre grup-koşullu kalibrasyon. |
| Anomali | Her ilan için 5-fold OOF log-artıklarından robust MAD-z skoru; fiyattan bağımsız yapısal sinyal için Isolation Forest. Etiket bulunmadığından sahte bir doğruluk metriği raporlanmaz. |

Ayrıntılı temizleme kuralları ve satır kaybı analizi:
[cleaning_pipeline.md](docs/cleaning_pipeline.md) ve
[attrition_analysis.md](docs/attrition_analysis.md).

## Model sonuçları

Model seçimi validation kümesinde yapılmıştır:

| Model | RMSE ($) | MAE ($) | MAPE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 8.212 | 4.396 | %61,4 | 0,62 |
| Random Forest | 6.867 | 3.672 | %45,0 | 0,73 |
| **LightGBM** | **6.067** | **3.197** | **%34,3** | **0,79** |

Train + validation havuzuyla kurulan LightGBM (bu havuzdan ayrılan iç
early-stopping holdout'u ile), 39.563 satırlık test kümesinde **6.253 $ RMSE**,
**3.143 $ MAE**, **%32,4 MAPE** ve **0,78 R²** üretmiştir. En yüksek gain
payları `age` (%39), `model` (%16) ve `odometer` (%10) özelliklerindedir. Tüm
ablation ve segment analizleri [phase3_results.md](docs/phase3_results.md)
içindedir.

%90 Mondrian tahmin aralığı testte **%90,22 marjinal kapsama** sağlar; tahmin
fiyatı gruplarında kapsama yaklaşık **%89-%91** aralığındadır. Gerçek fiyatın
50-150 bin dolar kuyruğundaki düşük kapsama, kalibrasyondan çok nadir ve pahalı
araçlardaki nokta tahmin hatasından kaynaklanır. Ayrıntı:
[phase6_results.md](docs/phase6_results.md).

## Öne çıkan çıktılar

- Değer kaybı öne yüklüdür: medyan fiyat 5. yılda %47, 10. yılda %72 düşer.
  Kilometre etkisi doğrusal değildir ve yaklaşık 150 bin milden sonra zayıflar.
- Açıklamadan türetilen üç sızıntısız özellik, test RMSE'yi 6.591 $'dan
  6.253 $'a düşürmüştür. Kontrollü deney:
  [phase7_results.md](docs/phase7_results.md).
- 7.552 ilan artık sinyaliyle, 1.979 ilan yapısal sinyalle işaretlenmiştir.
  İki güçlü ve bağımsız sinyalin kesişimindeki **48 ilan** en yüksek öncelikli
  inceleme kümesidir. [suspicious_listings.csv](reports/suspicious_listings.csv)
  üç aksiyon sınıfından en yüksek skorlu 30 örneği içerir.

EDA grafikleri `reports/figures/`, yöntem ve örnekler
[phase2_insights.md](docs/phase2_insights.md) ve
[phase4_results.md](docs/phase4_results.md) altındadır.

## Kurulum ve çalıştırma

Python 3.12 önerilir. Repoya dahil edilmeyen ham Kaggle verisini
`data/raw/vehicles.csv` konumuna yerleştirin; kaynak bilgisi için
[data/raw/README.md](data/raw/README.md) dosyasına bakın.

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python scripts/clean_data.py
python scripts/run_eda.py
python scripts/train.py
python scripts/detect_anomalies.py
python scripts/predict_intervals.py
```

A4 açıklama özelliği deneyini ayrıca üretmek için:

```powershell
python scripts/ablation_description_features.py
```

Testler gerçek veri dosyasına ihtiyaç duymaz:

```powershell
python -m pytest -q
```

Mevcut paket: **122 test**. Bağımlılıklar tam sürümleriyle sabitlenmiştir.

## Proje yapısı

```text
src/          reusable preprocessing, features, models, evaluation, anomaly logic
scripts/      pipeline entry points and diagnostic probes
tests/        pytest suite mirroring src/
data/         raw input and generated parquet output
reports/      figures, presentation and suspicious-listing report
docs/         audits, experiment plans and measured results
notebooks/    pre-executed presentation notebook
```

## Kapsam ve sınırlamalar

- Veri, ABD pazarından Nisan-Mayıs 2021 dönemine ait yaklaşık 30 günlük bir
  kesittir; mevsimsellik ve zaman içindeki fiyat kayması modellenmez.
- Test kayıtlarının %4,6'sında train tarafında yakın kopya bulunmuştur; temiz
  alt kümede RMSE farkı yalnızca +%0,6 olduğu için grup bazlı yeniden bölme
  uygulanmamıştır.
- Açıklama özellikleri A4'te aynı test kümesi üzerinde karşılaştırıldıktan sonra
  varsayılan sete alınmıştır. Bu nedenle 6.253 $'lık post-A4 skor, tamamen
  dokunulmamış bir final tahmin değil, seçim sonrası ölçümdür; yeni bir holdout
  veya nested CV ile doğrulanmalıdır.
- Faz 8 yapısal refaktörü 122 testle doğrulanmış, ancak ham veri repoda olmadığı
  için tam veri akışı refaktör sonrasında yeniden çalıştırılmamıştır. Kayıtlı
  metriklerin kaynağı ve bu risk [phase8_results.md](docs/phase8_results.md)
  içinde açıkça belgelenmiştir.
