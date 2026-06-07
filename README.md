# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-4CAF50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)

</div>

---

## 📋 2. Proje Özeti

Bu proje, endüstriyel kontrol sistemlerinde **zaman serisi anomali tespiti** (Time Series Anomaly Detection) için dört farklı modelleme yaklaşımını — **LSTM, 1D-CNN, Olasılıksal Otomata** — sistematik biçimde karşılaştırmaktadır.

**SKAB** ve **BATADAL** veri setleri üzerinde **orijinal**, **gürültülü (Gaussian Noise)** ve **görülmemiş (Unseen)** senaryolar altında deneyler yürütülmüş; her model **5 farklı rastgele tohumla (seed: 42, 123, 2026, 7, 999)** değerlendirilmiştir.

Sınıf dengesizliği sorunu ağırlıklı kayıp fonksiyonuyla (weighted loss) giderilmiş, her veri setine özgü karar eşiği (threshold) kullanılmıştır. Proje; **açıklanabilirlik (XAI)**, **istatistiksel test (Wilcoxon)** ve **parametre duyarlılık analizi** modüllerini de kapsamakta olup tamamen yeniden üretilebilir (reproducible) bir pipeline sunmaktadır.

---

## 🔬 3. Araştırma Sorusu

> **"Farklı modelleme yaklaşımları zaman serisi anomali tespitinde nasıl davranır?"**

Endüstriyel sistemlerde anomali tespiti; yüksek sınıf dengesizliği, zaman bağımlılığı ve görülmemiş durum sorunları nedeniyle zorlu bir problemdir. Bu çalışmada şu alt sorular incelenmektedir:

- Derin öğrenme modelleri (LSTM, 1D-CNN) deterministik bir yapıya sahip **Olasılıksal Otomata**'ya kıyasla F1 skorunda gerçekten üstün müdür?
- Gaussian gürültüsü ve görülmemiş durum enjeksiyonuna karşı hangi model daha **dayanıklıdır**?
- Bir veri seti üzerinde eğitilen model, başka bir veri setine **genellenebilir** mi (cross-dataset transfer)?
- Otomata parametreleri (**pencere boyutu**, **alfabe boyutu**) performansı ne ölçüde etkiler?

---

## 🗂️ 4. Proje Mimarisi

```
Time_Series_Automata/
│
├── src/
│   ├── main.py               # Uçtan uca çalıştırma pipeline'ı
│   ├── config.yaml           # Tüm hiperparametreler (hard-coded değer yasak)
│   ├── data_pipeline.py      # Veri yükleme, zamansal bölme, leakage-free PCA/Scaler
│   ├── dl_models.py          # LSTM ve 1D-CNN PyTorch sınıfları (logit çıktı)
│   ├── dl_trainer.py         # Eğitim döngüsü (pos_weight, erken durdurma, seed)
│   ├── automata_model.py     # ProbabilisticAutomata, SAX, PAA, Levenshtein
│   ├── explainability.py     # ExplainabilityEngine — JSON açıklama üretici
│   ├── metrics.py            # F1, Accuracy, Precision, Recall, Wilcoxon
│   ├── visualization.py      # Grafik üretim modülü (300 DPI)
│   └── utils.py              # TimeSeriesDataset, Gaussian Noise ekleyici
│
├── tests/
│   └── test_levenshtein.py   # pytest birim testleri (Levenshtein)
│
├── data/
│   └── raw/
│       ├── SKAB/             # valve1/, valve2/ CSV'leri
│       └── BATADAL_dataset04.csv
│
├── results/                  # Üretilen grafikler, JSON ve CSV çıktıları
│   ├── BATADAL/
│   │   ├── original/
│   │   ├── gaussian_noise/
│   │   └── unseen_data/
│   └── SKAB/
│       ├── original/
│       ├── gaussian_noise/
│       └── unseen_data/
│
├── RAPOR.md                  # Detaylı bilimsel analiz raporu
├── requirements.txt
└── README.md
```

---

## ⚙️ 5. Kurulum ve Çalıştırma

### 5.1. Gereksinimler

```bash
pip install -r requirements.txt
```

### 5.2. Tam Pipeline'ı Çalıştırma

```bash
python src/main.py
```

> ⏱️ **Not:** Test işlemleri 5 farklı seed, 2 farklı veri seti ve 3 farklı senaryo altında koşturulduğundan (toplam 30 model eğitimi) işlemci gücüne bağlı olarak **30–40 dakika** sürebilir. Sonuçlar `results/` klasörüne görsel, JSON ve log formatlarında kaydedilir.

### 5.3. Birim Testler

```bash
pytest tests/test_levenshtein.py -v
```

### 5.4. Hızlı Import Kontrolü

```bash
python -c "
from data_pipeline import DataPipeline
from dl_models import LSTMAnomalyDetector, CNN1DAnomalyDetector
from automata_model import ProbabilisticAutomaton, apply_sax, apply_paa
from explainability import ExplainabilityEngine
from metrics import calculate_metrics, run_wilcoxon_test
print('ALL IMPORTS OK')
"
```

---

## 📦 6. Veri Setleri

### 6.1. SKAB (Skoltech Anomaly Benchmark)

| Özellik | Değer |
| :--- | :--- |
| Kaynak | Skoltech — açık kaynak endüstriyel boru hattı verisi |
| Etiket sütunu | `anomaly` |
| Toplam satır | ~34.000 |
| Özellik sayısı | 8 (basınç, sıcaklık, akış vb.) |
| Anomali oranı | ~%5 |
| Bölme stratejisi | Zamansal sıralı (temporal) — %60 / %20 / %20 |

**Ön işleme adımları:**
- `datetime` ve `changepoint` sütunları kaldırılır.
- `StandardScaler` yalnızca eğitim verisi üzerinde `fit` edilir (veri sızıntısı önlenir).
- Otomata için `PCA(n_components=1)` ile tek boyuta indirgeme yapılır.

### 6.2. BATADAL (Battle of the Attack Detection ALgorithms)

| Özellik | Değer |
| :--- | :--- |
| Kaynak | BATADAL yarışması — su dağıtım altyapısı sensör verisi |
| Dosya | `BATADAL_dataset04.csv` |
| Etiket sütunu | `ATT_FLAG` |
| Toplam satır | ~8.760 |
| Özellik sayısı | 43 (sensör ve aktüatör okumaları) |
| Anomali oranı | ~%9.6 |
| Bölme stratejisi | Zamansal sıralı (temporal) — %60 / %20 / %20 |

**Ön işleme adımları:**
- `DATETIME` sütunu kaldırılır; sütun adlarındaki boşluklar temizlenir.
- `-999` olarak kodlanmış **Normal** etiketleri `0`'a dönüştürülür; yalnızca `1` değerleri Anomali sayılır.
- Sınıf dengesizliği ~%90.4 Normal → DL modeller `pos_weight = n_neg / n_pos` ile eğitilir.

---

## 🤖 7. Modeller

### 7.1. LSTM (Long Short-Term Memory)

Uzun vadeli bağımlılıkları öğrenebilen tekrarlayan sinir ağı. Kapı mekanizmaları (forget, input, output gate) sayesinde gradyan sönmesi sorununu aşar.

```
Girdi: (batch, window_size=4, features)
Mimari: LSTM(hidden=64, layers=2) → Dropout(0.2) → FC(64→1)
Kayıp:  BCELoss
Çıktı:  Sigmoid olasılık → threshold ile binary karar
```

### 7.2. 1D-CNN (1-Boyutlu Evrişimli Sinir Ağı)

Yerel zamansal örüntüleri kısa pencereler üzerinde evrişim filtresiyle öğrenir. Eğitim süresi LSTM'e kıyasla belirgin biçimde daha kısadır.

```
Girdi: (batch, features, window_size)
Mimari: Conv1d(64, kernel=3) → ReLU → GlobalAvgPool → FC(64→1)
Kayıp:  BCELoss
Çıktı:  Sigmoid olasılık → threshold ile binary karar
```

### 7.3. Olasılıksal Otomata (Probabilistic Automaton)

Veri odaklı, yorumlanabilir ve eğitim maliyeti son derece düşük bir deterministik model.

#### 7.3.1. Algoritma Akışı

```
Ham Zaman Serisi
       │
       ▼
  PCA (1 boyut)        ← Eğitim verisinde fit; test'e yalnızca transform
       │
       ▼
  Sliding Window       ← window_size = [3, 4, 5, 6] (config'den)
       │
       ▼
  PAA (Piecewise Aggregate Approximation)
       │   Her penceredeki segmentlerin ortalaması alınır
       ▼
  SAX (Symbolic Aggregate approXimation)
       │   Gauss dağılımı kesme noktalarına göre semboller atanır
       ▼
  Durum Dizisi         ← örn. ["aab", "abb", "bbc", ...]
       │
       ▼
  Geçiş Matrisi (Laplace Düzlemeli)
       │
       ▼
  Anomali Kararı       ← P(Sᵢ → Sᵢ₊₁) < threshold → Anomali
```

#### 7.3.2. Geçiş Olasılığı Formülü (Laplace Düzleme)

$$P(S_i \rightarrow S_j) = \frac{\text{count}(S_i \rightarrow S_j) + 1}{\text{count}(S_i) + |V|}$$

burada $|V|$ sözlük (vocabulary) büyüklüğüdür.

#### 7.3.3. Görülmemiş Durum Yönetimi (Levenshtein Eşleme)

Test sırasında sözlükte bulunmayan bir durum ile karşılaşıldığında **Levenshtein mesafesiyle** en yakın bilinen duruma eşleme yapılır:

$$d_{Lev}(s_1, s_2) = \min(\text{insert, delete, substitute})$$

---

## 🧪 8. Deney Tasarımı

### 8.1. Senaryolar

| Senaryo | Açıklama |
| :--- | :--- |
| **Original** | Ham test verisi — herhangi bir değişiklik uygulanmaz |
| **Gaussian Noise** | Test verisine `N(0, 0.1)` dağılımlı gürültü eklenir |
| **Unseen Data** | Test verisine `×1.5` ölçek uygulanarak Otomata'nın hiç görmediği örüntüler oluşturulur |

### 8.2. Deneysel Protokol

| Parametre | Değer |
| :--- | :--- |
| Rastgele tohumlar | `[42, 123, 2026, 7, 999]` |
| Eğitim / Doğrulama / Test | %60 / %20 / %20 (zamansal sıralı) |
| Maksimum epoch | 50 |
| Erken durdurma (patience) | 5 |
| Batch boyutu | 32 |
| Öğrenme hızı | 1e-3 (Adam optimizer) |
| Otomata Window Size (tarama) | `[3, 4, 5, 6]` |
| Otomata Alphabet Size (tarama) | `[3, 4, 5, 6]` |

---

## 📊 9. Sonuçlar

### 9.1. Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Std)

Tüm değerler **5 seed** üzerinden ortalama ± standart sapma olarak verilmiştir.

| Model | SKAB F1 | SKAB Accuracy | BATADAL F1 | BATADAL Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **LSTM** | 0.6954 ± 0.0628 | 0.6902 ± 0.0610 | 0.9022 ± 0.0207 | 0.8945 ± 0.0224 |
| **1D-CNN** | **0.7795 ± 0.0347** | 0.7761 ± 0.0378 | 0.7249 ± 0.0062 | 0.6647 ± 0.0095 |
| **Automata** | 0.5250 ± 0.0000 | 0.6417 ± 0.0000 | 0.8195 ± 0.0000 | 0.8161 ± 0.0000 |

> 📌 **Not:** Otomata `std = 0.000` — model deterministik yapısı nedeniyle tüm seedlerde özdeş sonuç üretir.

---

### 9.2. Tablo 2: Gürültü ve Unseen Senaryo Robustness Analizi

| Model | Dataset | Orijinal F1 | Gürültülü F1 | Unseen F1 | Δ Gürültü | Δ Unseen |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| LSTM | BATADAL | 0.9022 | 0.9031 | 0.8008 | +0.0009 | **-0.1014** |
| 1D-CNN | BATADAL | 0.7249 | 0.7268 | 0.6875 | +0.0019 | -0.0374 |
| **Automata** | **BATADAL** | **0.8195** | **0.8240** | **0.8126** | **+0.0045** | **-0.0070** |
| 1D-CNN | SKAB | 0.7795 | 0.7782 | 0.7665 | -0.0013 | -0.0130 |
| LSTM | SKAB | 0.6954 | 0.6949 | 0.7105 | -0.0005 | +0.0151 |
| **Automata** | **SKAB** | **0.5250** | **0.5247** | **0.5524** | **-0.0003** | **+0.0274** |

> 💡 **Gözlem:** Tüm modeller Gaussian gürültüsüne karşı son derece dayanıklıdır (|Δ| < 0.005). BATADAL'da LSTM Unseen senaryosunda ~%10 F1 düşüşü yaşarken, **Otomata Levenshtein eşlemesi sayesinde en stabil model** olarak öne çıkmaktadır (Δ = -0.007).

---

### 9.3. Tablo 3: Cross-Dataset Genellenebilirlik (seed=42)

| Eğitim → Test | Model | F1 |
| :--- | :--- | :---: |
| SKAB → BATADAL | **Automata** | **0.5768** |
| SKAB → BATADAL | LSTM | 0.1123 |
| SKAB → BATADAL | 1D-CNN | 0.1289 |
| BATADAL → SKAB | **Automata** | **0.5164** |
| BATADAL → SKAB | LSTM | 0.1720 |
| BATADAL → SKAB | 1D-CNN | 0.1720 |

> 🔍 **Yorum:** Otomata, SAX sembolik kodlaması sayesinde alan bağımsız (domain-agnostic) soyut örüntüleri temsil edebildiğinden cross-dataset senaryosunda DL modellerine kıyasla **4–5 kat daha iyi genellenebilirlik** sergilemektedir.

---

### 9.4. Tablo 4: Otomata Parametre Duyarlılık Analizi — SKAB (F1, threshold=0.01)

| Alphabet \ Window | w=3 | w=4 | w=5 | w=6 |
| :---: | :---: | :---: | :---: | :---: |
| **a=3** | 0.030 | 0.030 | 0.046 | 0.080 |
| **a=4** | 0.051 | 0.098 | 0.149 | 0.231 |
| **a=5** | 0.088 | 0.189 | 0.264 | 0.288 |
| **a=6** | 0.133 | 0.259 | 0.323 | **0.333** |

> ✅ **En iyi SKAB:** `window=6, alphabet=6` → F1 = **0.333**

### Tablo 5: Otomata Parametre Duyarlılık Analizi — BATADAL (F1, threshold=0.01)

| Alphabet \ Window | w=3 | w=4 | w=5 | w=6 |
| :---: | :---: | :---: | :---: | :---: |
| **a=3** | 0.022 | 0.084 | 0.172 | 0.171 |
| **a=4** | 0.118 | 0.151 | 0.158 | 0.161 |
| **a=5** | 0.145 | 0.157 | **0.169** | 0.154 |
| **a=6** | 0.128 | 0.162 | 0.160 | 0.152 |

> ✅ **En iyi BATADAL:** `window=5, alphabet=5` → F1 = **0.169**

### Tablo 6: Modellerin Çalışma Süresi (Runtime) Karşılaştırması

| Model | Training Time (sn) | Inference Time (sn) |
| :--- | :---: | :---: |
| **LSTM** | 124.5 | 4.2 |
| **1D-CNN** | 86.3 | 2.1 |
| **Automata** | **1.8** | **0.4** |

> ⚡ Otomata, DL modellerine kıyasla eğitimde **~70x**, çıkarımda **~10x** daha hızlıdır.

---

## 🔍 10. Parametre Duyarlılık Görselleri

Otomata modelinin `window_size` ve `alphabet_size` parametrelerinin `[3,4,5,6]` aralığında ızgara taramasından (grid search) elde edilen F1 ısı haritaları:

| SKAB Isı Haritası | BATADAL Geçiş Diyagramı |
|:---:|:---:|
| ![SKAB Heatmap](results/automata_transition_heatmap.png) | ![BATADAL State](results/automata_state_diagram.png) |

---

## 💡 11. Olasılıksal Açıklanabilirlik Modülü (XAI)

`src/explainability.py` içindeki `ExplainabilityEngine` sınıfı, otomatanın her kararı için yorumlanabilir bir **JSON çıktısı** üretir. Görülmemiş durumlar Levenshtein mesafesiyle en yakın bilinen duruma eşlenir.

### 11.1. Örnek JSON Çıktısı

```json
{
  "time_step": 5,
  "state": "aab",
  "pattern": "adc",
  "status": "unseen",
  "mapped_to": "abc",
  "probability": 0.108,
  "decision": "anomaly"
}
```

| Alan | Açıklama |
| :--- | :--- |
| `state` | Mevcut SAX sembolik durumu |
| `pattern` | Gözlemlenen geçiş dizisi |
| `status` | `seen`: sözlükte mevcut; `unseen`: Levenshtein ile eşlendi |
| `mapped_to` | En yakın bilinen durum |
| `probability` | P(state → mapped_to) geçiş olasılığı |
| `decision` | `anomaly` (olasılık < eşik) veya `normal` |

### 11.2. Confusion Matrix Görselleri

| LSTM - BATADAL | Automata - BATADAL |
|:---:|:---:|
| ![LSTM CM](results/LSTM_confusion_matrix.png) | ![Automata CM](results/Automata_confusion_matrix.png) |

### 11.3. ROC Eğrisi

| LSTM ROC |
|:---:|
| ![LSTM ROC](results/LSTM_roc_curve.png) |

---

## 📈 12. İstatistiksel Testler (Wilcoxon)

5 seed'deki F1 skorları arasındaki medyan farkını parametrik olmayan biçimde test eden **Wilcoxon İşaret-Sıralama Testi** uygulanmıştır.

| Karşılaştırma | Dataset | p-değeri | Yorum |
| :--- | :--- | :---: | :--- |
| LSTM vs 1D-CNN | SKAB | 0.3125 | Anlamlı değil |
| LSTM vs Automata | SKAB | 0.0625 | Sınırda — güçlü trend var |
| LSTM vs Automata | BATADAL | 0.0625 | Sınırda — güçlü trend var |
| 1D-CNN vs Automata | BATADAL | 0.0625 | Sınırda — güçlü trend var |

> ⚠️ 5 seed ile Wilcoxon testi düşük istatistiksel güce sahiptir. p=0.0625 değerleri güçlü bir performans eğilimine işaret eder ancak klasik α=0.05 eşiğini geçememektedir.

---

## 🏁 13. Bulgular ve Tartışma

### 13.1. BATADAL: Derin Öğrenme Üstünlüğü

BATADAL'da **LSTM F1 = 0.9022** ile en yüksek performansı elde etmiştir:
- Veri seti ardışık saldırı oturumları içermekte; LSTM'in hafıza mekanizması bu uzun vadeli bağımlılıkları etkin modelleyebilmektedir.
- Sınıf dengesizliği (~%90 Normal) `BCELoss` ile giderilmiştir.

### 13.2. SKAB: 1D-CNN Liderliği

SKAB'da **1D-CNN** beklenmedik biçimde en iyi sonucu vermiştir (**F1 = 0.7795**). SKAB'ın kısa süreli anlık anomalileri, CNN'in yerel örüntü öğrenme kapasitesine daha iyi uymaktadır.

### 13.3. Otomata: Stabilite ve Yorumlanabilirlik

| Özellik | Otomata | DL Modeller |
| :--- | :--- | :--- |
| Eğitim süresi | **< 2 saniye** | 86–125 saniye |
| Yorumlanabilirlik | **JSON açıklama** | Kara kutu |
| Cross-dataset F1 | **0.52–0.58** | 0.11–0.17 |
| Unseen dayanıklılığı | **En stabil (Δ = -0.007)** | BATADAL'da Δ = -0.10 |
| Seed varyansı | **0.000** | 0.006–0.095 |

---

## 📚 14. Kaynaklar

1. **SKAB Veri Seti:** Ilyasov, A. et al. *"SKAB: Skoltech Anomaly Benchmark."* GitHub, 2020.
2. **BATADAL Veri Seti:** Taormina, R. et al. *"The Battle of the Attack Detection Algorithms."* Journal of Water Resources Planning and Management, 2018.
3. **SAX:** Lin, J. et al. *"Experiencing SAX: A Novel Symbolic Representation of Time Series."* Data Mining and Knowledge Discovery, 2007.
4. **PAA:** Keogh, E. et al. *"Dimensionality Reduction for Fast Similarity Search in Large Time Series Databases."* Knowledge and Information Systems, 2001.
5. **Levenshtein Mesafesi:** Levenshtein, V.I. *"Binary codes capable of correcting deletions, insertions, and reversals."* Soviet Physics Doklady, 1966.
6. **Wilcoxon Testi:** Wilcoxon, F. *"Individual comparisons by ranking methods."* Biometrics Bulletin, 1945.
7. **PyTorch BCELoss:** [pytorch.org/docs](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

---

<div align="center">

**Proje kapsamlı bilimsel analiz raporu için → [RAPOR.md](RAPOR.md)**

*Reproducible • Explainable • Modular*

</div>
