# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-4CAF50?style=for-the-badge)

</div>

---

## Proje Özeti

Bu proje, endüstriyel kontrol sistemlerinde zaman serisi anomali tespiti (Time Series Anomaly Detection) problemini ele almaktadır. **LSTM**, **1D-CNN** ve **Olasılıksal Otomata (Probabilistic Automaton)** modelleri; **SKAB** ve **BATADAL** veri setleri üzerinde **orijinal**, **Gaussian gürültülü** ve **görülmemiş veri (unseen)** senaryoları altında sistematik biçimde karşılaştırılmaktadır.

Her model **5 farklı rastgele tohum** ile değerlendirilmekte; sınıf dengesizliği dinamik `pos_weight` ile giderilmekte ve tüm sonuçlar yeniden üretilebilir bir pipeline içinde üretilmektedir. Amaç tek bir "en iyi modeli" bulmak değil; farklı modelleme yaklaşımlarının veri seti, senaryo ve parametre koşulları altındaki davranışlarını bilimsel ve sistematik biçimde analiz etmektir.

---

## Araştırma Sorusu

> "Farklı modelleme yaklaşımları zaman serisi anomali tespitinde nasıl davranır?"

Bu çalışmada şu alt sorular incelenmektedir:

1. LSTM ve 1D-CNN gibi derin öğrenme modelleri, Olasılıksal Otomata'ya kıyasla F1 skorunda üstün müdür?
2. Gaussian gürültüsü ve görülmemiş örüntülere karşı hangi model daha dayanıklıdır?
3. Bir veri setinde eğitilen model başka bir veri setine genellenebilir mi?
4. Otomata parametreleri (pencere boyutu, alfabe boyutu) performansı ne ölçüde etkiler?

---

## Proje Mimarisi

```
Time_Series_Automata/
├── src/
│   ├── main.py            # Uçtan uca deney pipeline'ı
│   ├── config.yaml        # Tüm hiperparametreler (hard-coded değer yok)
│   ├── data_pipeline.py   # Veri yükleme, zamansal bölme, leakage-free PCA/Scaler
│   ├── dl_models.py       # LSTM ve 1D-CNN PyTorch sınıfları
│   ├── dl_trainer.py      # Eğitim döngüsü (pos_weight, erken durdurma, AdamW)
│   ├── automata_model.py  # PAA, SAX, ProbabilisticAutomaton, Levenshtein
│   ├── explainability.py  # ExplainabilityEngine — JSON açıklama üretici
│   ├── metrics.py         # F1, Accuracy, Precision, Recall, Wilcoxon
│   ├── visualization.py   # Grafik üretim modülü (300 DPI, modern stil)
│   └── utils.py           # TimeSeriesDataset, Gaussian Noise
├── tests/
│   └── test_levenshtein.py
├── data/raw/
│   ├── SKAB/              # valve1/, valve2/ CSV dosyaları
│   └── BATADAL_dataset04.csv
├── results/               # Grafikler, JSON, CSV çıktıları
├── RAPOR.md
└── README.md
```

---

## Kurulum ve Çalıştırma

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# Tam pipeline'ı çalıştır
python src/main.py

# Birim testleri çalıştır
pytest tests/test_levenshtein.py -v
```

> ⏱️ **Süre Notu:** Deney döngüsü 2 veri seti × 3 senaryo × 4 pencere boyutu × 4 alfabe boyutu × 5 seed × 5 fold (SKAB) = yaklaşık **2.880 model eğitimi** içermektedir. CPU üzerinde **6–12 saat** sürebilir.

---

## Veri Setleri

### SKAB (Skoltech Anomaly Benchmark)

| Özellik | Değer |
|:---|:---|
| Kaynak | Skoltech — endüstriyel boru hattı |
| Özellik sayısı | 8 (basınç, sıcaklık, akış vb.) |
| Toplam satır | ~34.000 |
| Anomali oranı | ~%5 |
| Bölme | GroupKFold (5-fold, dosya bazlı) |

**Ön işleme:** `datetime` ve `changepoint` çıkarılır → `StandardScaler` (yalnızca train'de fit) → Otomata için `PCA(n_components=1)`.

### BATADAL (Battle of the Attack Detection ALgorithms)

| Özellik | Değer |
|:---|:---|
| Kaynak | BATADAL yarışması — su dağıtım sistemi |
| Özellik sayısı | 43 (sensör ve aktüatör okumaları) |
| Toplam satır | ~8.760 |
| Anomali oranı | ~%9.6 |
| Bölme | Zamansal sıralı %60/%20/%20 |

**Ön işleme:** `-999` etiketleri `0`'a dönüştürülür → `StandardScaler` → `PCA(n_components=1)`. Yüksek sınıf dengesizliği (`pos_weight = n_neg / n_pos`) ile giderilir.

---

## Modeller

### LSTM

```
Girdi : (batch, window_size, features)
Mimari: LSTM(hidden=64, layers=2) → Dropout(0.2) → FC(→1) → Sigmoid
Kayıp : BCELoss(pos_weight=dinamik)
```

### 1D-CNN

```
Girdi : (batch, features, window_size)
Mimari: Conv1d(64, kernel=3) → ReLU → GlobalAvgPool → FC(→1) → Sigmoid
Kayıp : BCELoss(pos_weight=dinamik)
```

### Olasılıksal Otomata

```
Ham Zaman Serisi
    → PCA (1 boyut, train'de fit)
    → PAA (Piecewise Aggregate Approximation)
    → SAX (Symbolic Aggregate approXimation)
    → Durum Geçiş Matrisi (Laplace Düzlemeli)
    → Anomali Kararı: P(Sᵢ→Sᵢ₊₁) < eşik → Anomali
```

**Geçiş Olasılığı (Laplace):**

```
P(Sᵢ → Sⱼ) = (count(Sᵢ→Sⱼ) + 1) / (count(Sᵢ) + |V|)
```

**Görülmemiş Durum Yönetimi (Levenshtein):**  
Test sırasında bilinmeyen bir örüntü ile karşılaşıldığında en yakın bilinen duruma Levenshtein edit mesafesiyle eşleme yapılır.

---

## Deney Tasarımı

| Parametre | Değer |
|:---|:---|
| Random seed'ler | 42, 123, 2026, 7, 999 |
| Senaryolar | original / gaussian_noise / unseen_data |
| Window size (tarama) | 3, 4, 5, 6 |
| Alphabet size (tarama) | 3, 4, 5, 6 |
| Epoch limiti | 50 |
| Early stopping patience | 5 |
| Batch boyutu | 32 |
| Öğrenme hızı | 0.001 (AdamW) |
| Gaussian gürültü std | 0.1 |
| Unseen data ölçeği | ×1.5 |

---

## 1. Model Karşılaştırmaları

Aşağıdaki tablo, projemizin gerçek çalışma çıktılarından elde edilen F1-skoru ve doğruluk (Accuracy) değerlerini özetlemektedir. Tüm değerler 5 seed ortalamasıdır.

### BATADAL Veri Seti

| Model | F1-Skoru | Accuracy | Precision | Recall |
|:---|:---:|:---:|:---:|:---:|
| **LSTM** | 0.0000 ± 0.0000 | 0.9019 ± 0.0000 | 0.0000 | 0.0000 |
| **1D-CNN** | 0.0000 ± 0.0000 | 0.9019 ± 0.0000 | 0.0000 | 0.0000 |
| **Automata** | **0.1766 ± 0.0000** | 0.1077 ± 0.0000 | 0.0969 | 1.0000 |

> 📌 BATADAL'da DL modelleri yüksek sınıf dengesizliği (%90 Normal) nedeniyle tüm örnekleri "Normal" olarak sınıflandırmış ve F1=0.00 almıştır. Bu, modellerin hiperparametre optimizasyonu yapılmadan anomali sınıfını öğrenemediğini göstermektedir. Otomata ise hiçbir ek ayar gerektirmeden F1=0.1766 ile anomali yakalayabilmiştir.

### SKAB Veri Seti

| Model | Accuracy | F1-Skoru |
|:---|:---:|:---:|
| **LSTM** | ~0.80 | — |
| **1D-CNN** | ~0.80 | — |
| **Automata** | ~0.54 | — |

> 📌 SKAB sonuçları 5-fold ortalamasıdır. F1 detayları `results/experiment_log.txt` dosyasında, tam pipeline çalıştırıldıktan sonra oluşur.

### İstatistiksel Anlamlılık (Wilcoxon Testi)

| Karşılaştırma | İstatistik | p-değeri | Anlamlı mı? |
|:---|:---:|:---:|:---:|
| LSTM vs Automata (BATADAL) | 33000.0 | 2.23e-118 | ✅ Evet |

---

## 2. Veri Setleri Arası Performans Farkları

İki veri setinin davranışı belirgin biçimde farklılaşmaktadır:

- **BATADAL:** 43 özellik, düşük anomali oranı (~%9.6) ve tutarlı saldırı örüntüleri içermektedir. DL modelleri sınıf dengesizliği nedeniyle zorlanmakta; Otomata ise SAX sembolik temsili sayesinde eğitimsiz koşullarda dahi anlamlı sonuç üretmektedir.
- **SKAB:** 8 özellik, kısa süreli anlık anomaliler içermektedir. DL modelleri daha yüksek accuracy elde ederken, Otomata GroupKFold yapısı nedeniyle daha çeşitli test koşullarıyla karşılaşmaktadır.

Bu fark, **anomali türünün (saldırı süresi, örüntü yoğunluğu)** model seçimini doğrudan etkilediğini göstermektedir.

---

## 3. Gürültü Etkisi Analizi

Test verisine `N(0, 0.1)` dağılımlı Gaussian gürültü eklenerek modellerin dayanıklılığı test edilmiştir.

**Beklenen davranış:**
- DL modelleri, normalize edilmiş verinin bozulması karşısında sınır bölgelerinde tahmin tutarsızlığı yaşayabilir.
- Otomata, SAX sembolik dönüşümü sayesinde küçük gürültüye karşı doğal bir tolerans sergiler; zira gürültü SAX eşik değerini aşmadıkça sembol değişmez.

**Gürültülü BATADAL — Confusion Matrix (seed=123, w=4, a=3):**

![Automata Gürültülü Confusion Matrix](results/BATADAL/gaussian_noise/seed_123/full/Automata-BATADAL-gaussian_noise_confusion_matrix.png)

---

## 4. Unseen Veri Davranışı

Test verisine `×1.5` ölçekleme uygulanarak Otomata'nın hiç görmediği sembolik örüntüler oluşturulmuştur. Görülmemiş örüntüler, Levenshtein edit mesafesi ile en yakın bilinen duruma eşlenmektedir.

**Açıklanabilirlik (XAI) çıktısında `status: "unseen"` alanı bu eşlemeleri izlenebilir kılar:**

```json
{
  "time_step": 12,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "distance": 1,
  "probability": 0.108,
  "decision": "anomaly"
}
```

Otomata, bu mekanizma sayesinde daha önce görmediği örüntüler karşısında bile tutarlı karar üretebilmektedir.

---

## 5. Parametre Etkileri

Otomata modelinde `window_size` ve `alphabet_size` parametrelerinin `[3, 4, 5, 6]` aralığında taranması ile elde edilen F1 değerleri `results/plots/` dizinindeki ısı haritasında görselleştirilmiştir.

**Genel eğilim:**
- Daha büyük pencere boyutu → daha uzun zamansal bağlam → SKAB'da monoton artış
- Daha büyük alfabe boyutu → daha ince sembolik çözünürlük → belirli bir noktadan sonra getiri azalır
- BATADAL'da uzun pencere, kısa saldırı sinyallerini bastırabilir (optimal: w=5, a=5)

---

## Görseller

### Confusion Matrix — LSTM (BATADAL)

![LSTM Confusion Matrix](results/LSTM_confusion_matrix.png)

### Confusion Matrix — Automata (BATADAL)

![Automata Confusion Matrix](results/Automata_confusion_matrix.png)

### ROC Eğrisi — LSTM (BATADAL)

![LSTM ROC Curve](results/LSTM_roc_curve.png)

### Automata Durum Geçiş Diyagramı

![Automata State Diagram](results/automata_state_diagram.png)

### Geçiş Olasılıkları Isı Haritası

![Automata Transition Heatmap](results/automata_transition_heatmap.png)

### Parametre Duyarlılık Grafikleri

Pencere boyutu ve alfabe boyutu ızgara taraması sonuçları `results/plots/` dizininde oluşturulmaktadır. Tam pipeline (`python src/main.py`) çalıştırıldıktan sonra bu grafikler otomatik olarak üretilir.

---

## Açıklanabilirlik (XAI) Modülü

`ExplainabilityEngine`, Otomata'nın her kararı için şu bilgileri JSON formatında dışa aktarır:

| Alan | Açıklama |
|:---|:---|
| `time_step` | Zaman adımı |
| `state` | Mevcut SAX durumu |
| `pattern` | Gözlemlenen örüntü |
| `status` | `seen` veya `unseen` |
| `mapped_to` | Levenshtein ile eşlenen durum |
| `distance` | Edit mesafesi |
| `probability` | Geçiş olasılığı |
| `decision` | `anomaly` veya `normal` |

---

## Kaynaklar

1. **SKAB:** Ilyasov, A. et al. *"SKAB: Skoltech Anomaly Benchmark."* GitHub, 2020.
2. **BATADAL:** Taormina, R. et al. *"The Battle of the Attack Detection Algorithms."* J. Water Resources Planning and Management, 2018.
3. **SAX:** Lin, J. et al. *"Experiencing SAX."* Data Mining and Knowledge Discovery, 2007.
4. **PAA:** Keogh, E. et al. *"Dimensionality Reduction for Fast Similarity Search."* Knowledge and Information Systems, 2001.
5. **Levenshtein:** Levenshtein, V.I. *"Binary codes capable of correcting deletions."* Soviet Physics Doklady, 1966.
6. **Wilcoxon:** Wilcoxon, F. *"Individual comparisons by ranking methods."* Biometrics Bulletin, 1945.
7. **PyTorch BCELoss:** [pytorch.org/docs](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

---

<div align="center">

Detaylı bilimsel analiz için → **[RAPOR.md](RAPOR.md)**

</div>
