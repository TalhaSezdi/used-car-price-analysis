# İkinci El Araç Fiyatlama ve Anomali Tespiti

426.880 Craigslist ilanını temizleme, keşifçi veri analizi, fiyat tahmini,
tahmin aralığı üretimi ve şüpheli ilan tespitinden geçiren uçtan uca bir veri
bilimi projesi. Modüler iş mantığı `src/` altında OOP sınıfları olarak, iş
akışları `scripts/` altında ince orkestrasyon script'leri olarak tutulur;
notebook yalnızca sunum katmanıdır.

## Akış

```text
data/raw/vehicles.csv (426,880 ilan)
  -> DataCleaner + FeatureEngineer
  -> data/processed/cleaned.parquet (197,814 ilan)
       |-> EDA -> 13 figure + business insights
       |-> 60/20/20 stratified split
       |     -> FeaturePreprocessor -> Linear / Random Forest / LightGBM
       |     -> final LightGBM -> point estimate + conformal interval
       `-> 5-fold OOF predictions -> MAD-z + Isolation Forest
                                      -> suspicious_listings.csv
```

## Pipeline adımları (kod akışına göre)

Her adım tek bir script ile çalışır; script'ler yalnızca `src/` altındaki
sınıfları birbirine bağlar. Paylaşılan sabitler (`RANDOM_STATE=42`, segment
aralıkları, anomali eşiği, aralık alpha'sı) tek kaynak olarak
`src/config.py` içindedir.

### 1. Temizleme ve özellik üretimi — `scripts/clean_data.py`

`DataCleaner` (`src/preprocess/cleaner.py`) üç aralık filtresi uygular ve
her eşiğin bir gerekçesi vardır:

- **Fiyat `[500, 150000]`:** ham veride 0-1 dolarlık büyük bir placeholder
  kütlesi ve klavye hatası uç değerler var (maksimum 3,7 milyar $). Taban
  ciddi olmayan ilanları eler; tavan veri girişi hatalarını atar ve farklı
  bir fiyatlama rejimindeki egzotik/ticari araçları kapsam dışına alır.
- **Yıl `[1970, 2022]`:** veri 2021'de toplandığı için 2022 sonrası yıl
  imkansızdır; 1970 öncesi araçlar ise ayrı dinamikleri olan koleksiyon
  pazarına aittir.
- **Kilometre `[1, 500000]`:** ikinci el bir araçta 0 kilometre gerçekçi
  değildir; 500 bin mil üzeri neredeyse her zaman veri girişi hatasıdır.

Ayrıca `salvage`/`parts only` gibi title durumları atılır (piyasa değeri
modeline gürültü katan farklı bir fiyat rejimi) ve ilanlar tekilleştirilir:
aynı fiziksel aracın yeniden yayınlanması gerçek kopya olduğundan önce VIN
eşleşmesiyle, VIN'siz kayıtlarda ise parmak iziyle silinir — bu, aynı
aracın train ve test'e birden düşüp metrikleri şişirmesini önler. Her
kuralın sildiği satır sayısı `CleaningReport` ile kayıt altına alınır
(426.880 -> 197.814, %46,3 retention); ham dosya asla değiştirilmez. `FeatureEngineer`
(`src/features/engineer.py`) `age = posting_year - year` (referans yıl
posting tarihi, asla bugünün tarihi değil), yıllık kilometre ve log
dönüşümlerini üretir; `DescriptionFeatureExtractor` açıklama metninden üç
sızıntısız özellik çıkarır (trim/donanım eşleşmesi ve metin uzunluğu —
metinden hiçbir sayısal değer alınmaz, ham metin modele girmez). Çıktı:
`data/processed/cleaned.parquet`. Kural bazında retention hunisi ve satır
kaybı analizi: [cleaning_pipeline.md](docs/cleaning_pipeline.md),
[attrition_analysis.md](docs/attrition_analysis.md).

### 2. Keşifçi veri analizi — `scripts/run_eda.py`

`src/evaluation/plots.py` ve `insights.py` ile 13 figür üretir; her grafiğin
altında "bu bir pazar yeri için ne anlama gelir" yorumu vardır. Çıktılar:
`reports/figures/` ve [phase2_insights.md](docs/phase2_insights.md).

### 3. Model eğitimi ve karşılaştırma — `scripts/train.py`

`build_split` (`src/models/dataset.py`) fiyat desiline göre stratified
%60 train / %20 validation / %20 test böler. `FeaturePreprocessor`
(`src/models/encoders.py`) sayısal alanlara medyan + eksiklik göstergesi,
düşük kardinaliteye one-hot, binlerce değerli `model` kolonuna
`SafeTargetEncoder` ile smoothing'li 5-fold out-of-fold target encoding
uygular — hiçbir satır kendi etiketiyle encode edilmez ve tüm fit'ler
yalnızca eğitim verisini görür. `log1p(price)` hedefi üzerinde Linear
Regression, Random Forest ve LightGBM eğitilir; model seçimi ve A1-A3
ablation'ları (log vs ham hedef, `age`/`year` collinearity, encoding
stratejisi) validation kümesinde yapılır. Kazanan LightGBM train +
validation (%80) üzerinde yeniden eğitilir ve test kümesi tek kez ölçülür.
Metrikler `expm1` sonrası dolar ölçeğinde raporlanır; yaş, fiyat segmenti ve
marka bazında hata analizi [phase3_results.md](docs/phase3_results.md) içindedir.

### 4. Anomali tespiti — `scripts/detect_anomalies.py`

`ResidualAnomalyDetector` (`src/anomaly/detector.py`) her ilan için 5-fold
out-of-fold log-artığı üretir ve robust MAD-z skoru `|z| > 3.5` olanları
işaretler: tahminin çok altındaki fiyat dolandırıcılık sinyali, çok
üstündeki fiyat veri girişi / spam sinyalidir. Artıklar dolar yerine log
ölçeğinde kullanılır çünkü dolar artıkları fiyatla birlikte büyür
(heteroscedastic) — sabit bir dolar eşiği pahalı araçları sistematik olarak
fazla, ucuzları eksik işaretlerdi. `3.5` eşiği bir istatistiksel nadirlik
iddiası değil, operasyonel bir kapasite seçimidir: artık dağılımı kalın
kuyruklu olduğundan Gauss varsayımıyla beklenen oranın ~80 katı ilan
(%3,8) işaretlenir ve eşik, insan inceleme kuyruğunun kaldırabileceği hacme
göre belirlenmiştir. `IsolationForestDetector`
fiyattan bağımsız yapısal tuhaflıkları yakalar. Etiket bulunmadığından sahte
bir doğruluk metriği raporlanmaz; bunun yerine en yüksek skorlu ilanlar tek
tek incelenip gerekçelendirilir. Çıktı:
[suspicious_listings.csv](reports/suspicious_listings.csv) ve
[phase4_results.md](docs/phase4_results.md).

### 5. Tahmin aralıkları — `scripts/predict_intervals.py`

`ConformalIntervalModel` (`src/models/intervals.py`) LightGBM quantile
modelleri ve ayrı bir calibration kümesiyle conformalized quantile
regression uygular (%90 nominal kapsama, `INTERVAL_ALPHA = 0.10`).
`MondrianConformalIntervalModel` kalibrasyonu tahmin bandı orta noktasına
göre grup-koşullu yapar, böylece kapsama ucuz ve pahalı segmentlerde ayrı
ayrı tutturulur. Sonuçlar: [phase6_results.md](docs/phase6_results.md).

### 6. Kontrollü deney A4 — `scripts/ablation_description_features.py`

Açıklamadan türetilen üç özelliğin (`desc_trim_luxury`, `desc_equip_count`,
`desc_len_log`) katkısını aynı split üzerinde ölçen ayrı deney; sonuçlar
[phase7_results.md](docs/phase7_results.md) içindedir.

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

Mevcut paket: **122 test**, `tests/` klasörü `src/` yapısını birebir
aynalar. Bağımlılıklar `requirements.txt` içinde tam sürümleriyle
sabitlenmiştir.

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
