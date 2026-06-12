# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-4CAF50?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)

</div>

---

## 📋 1. Proje Özeti ve Motivasyon

[cite_start]Bu proje, zaman serisi verileri üzerinde iki farklı modelleme paradigmasının (derin öğrenme tabanlı black-box modeller ve yorumlanabilir otomata tabanlı modeller) karşılaştırmalı analizini sunmaktadır[cite: 32, 33, 34]. [cite_start]Temel amaç tek bir en iyi modeli belirlemekten ziyade, modellerin farklı veri koşulları altındaki davranışlarını bilimsel ve sistematik bir şekilde analiz etmektir.

[cite_start]Deneyler, SKAB ve BATADAL veri setleri üzerinde anomali tespiti problemi olarak ele alınmıştır[cite: 44, 45, 46]. [cite_start]Her iki veri seti için de veri sızıntısını (data leakage) önleyecek standart deney protokolleri uygulanmış ve modellerin genellenebilirlik, gürültüye dayanıklılık ve açıklanabilirlik özellikleri değerlendirilmiştir[cite: 35, 120].

---

## 🔬 2. Araştırma Soruları

* [cite_start]Farklı modelleme yaklaşımları (LSTM, 1D-CNN, Olasılıksal Otomata) zaman serisi verilerinde nasıl bir performans göstermektedir? [cite: 38, 40, 69, 75]
* [cite_start]Modellerin performansı veri setine ne ölçüde bağımlıdır (Cross-Dataset Generalization)? [cite: 40]
* [cite_start]Modeller gürültü eklenmiş ve daha önce karşılaşılmamış (unseen) veri durumlarında nasıl davranmaktadır? [cite: 41]
* [cite_start]Olasılıksal otomata modelinin iç parametreleri (Window Size ve Alphabet Size) model performansını nasıl etkilemektedir? 

---

## 📊 3. Model Karşılaştırmaları ve Süre Analizi

[cite_start]Aşağıdaki tablolar, modellerin iki farklı veri seti üzerindeki ortalama F1-skorlarını ve çalışma sürelerini 5 farklı random seed (42, 123, 2026, 7, 999) ile elde edilen sonuçlarla göstermektedir[cite: 7, 128, 151, 152]. [cite_start]Model eğitimi ve test süreçleri, SKAB için dosya bazlı GroupKFold ve BATADAL için zaman sıralı test kümeleri kullanılarak raporlanmıştır[cite: 147, 148, 153].

### 📊 3.1 BATADAL Veri Seti Performans Sonuçları

| Pencere & Alfabe (w / a) | Model | Ortalama F1 Skor | Eğitim Süresi (sn) | Çıkarım (Inference) Süresi (sn) |
| :--- | :---: | :---: | :---: | :---: |
| **w=4, a=3** | LSTM | 0.4230 $\pm$ 0.1551 | 1.49 s | 0.0174 s |
| | CNN | 0.0000 $\pm$ 0.0000 | 2.54 s | 0.0149 s |
| | Otomata | **0.2500 $\pm$ 0.0000** | **0.0022 s** | **0.0126 s** |
| **w=4, a=4** | LSTM | **0.4230 $\pm$ 0.1551** | 1.49 s | 0.0174 s |
| | CNN | 0.0000 $\pm$ 0.0000 | 2.54 s | 0.0149 s |
| | Otomata | 0.1818 $\pm$ 0.0000 | **0.0021 s** | 0.0802 s |
| **w=5, a=3** | LSTM | **0.5741 $\pm$ 0.0121** | 0.92 s | 0.0185 s |
| | CNN | 0.0000 $\pm$ 0.0000 | 1.47 s | 0.0197 s |
| | Otomata | 0.1818 $\pm$ 0.0000 | **0.0030 s** | 0.1092 s |
| **w=5, a=4** | LSTM | **0.5741 $\pm$ 0.0121** | 0.92 s | 0.0185 s |
| | CNN | 0.0000 $\pm$ 0.0000 | 1.47 s | 0.0197 s |
| | Otomata | 0.1794 $\pm$ 0.0000 | **0.0022 s** | 0.4868 s |

> 📌 **Not:** BATADAL veri setindeki yüksek dengesizlik (anomali oranının düşüklüğü) ve anomali eşik değeri nedeniyle CNN modelinin F1 skoru 0.00 çıkmıştır (hiç anomali tahmin edememiştir). Bu veri setinde LSTM en iyi skoru verirken; Otomata modeli mikrosaniyeler mertebesindeki eğitim hızıyla öne çıkmaktadır.

### 📊 3.2 SKAB Veri Seti Performans Sonuçları (5 Fold GroupKFold)

| Pencere & Alfabe (w / a) | Model | Ortalama F1 Skor | Ortalama Eğitim (sn) | Ortalama Çıkarım (sn) |
| :--- | :---: | :---: | :---: | :---: |
| **w=4, a=3** | LSTM | 0.7980 $\pm$ 0.0025 | 10.77 s | 0.0758 s |
| | CNN | **0.8361 $\pm$ 0.0007** | 10.24 s | 0.0779 s |
| | Otomata | 0.0533 $\pm$ 0.0000 | **0.0116 s** | **0.0253 s** |
| **w=4, a=4** | LSTM | 0.7980 $\pm$ 0.0025 | 10.77 s | 0.0758 s |
| | CNN | **0.8361 $\pm$ 0.0007** | 10.24 s | 0.0779 s |
| | Otomata | 0.1094 $\pm$ 0.0000 | **0.0109 s** | 0.0379 s |
| **w=5, a=3** | LSTM | 0.8043 $\pm$ 0.0000 | 11.49 s | 0.0797 s |
| | CNN | **0.8282 $\pm$ 0.0031** | 9.94 s | 0.0800 s |
| | Otomata | 0.1464 $\pm$ 0.0000 | **0.0125 s** | **0.0559 s** |
| **w=5, a=4** | LSTM | 0.8043 $\pm$ 0.0000 | 11.49 s | 0.0797 s |
| | CNN | **0.8282 $\pm$ 0.0031** | 9.94 s | 0.0800 s |
| | Otomata | 0.2675 $\pm$ 0.0000 | **0.0118 s** | 0.1165 s |

> 📌 **Not:** SKAB veri seti üzerinde 5 katlı çapraz doğrulama uygulanmıştır. Kısa vadeli ve dinamik anomaliler içeren bu veri setinde **1D-CNN** modeli en yüksek F1 skorunu elde etmiştir (%83.61). Otomata modelinde pencere boyutu ($w$) ve harf sayısı ($a$) arttıkça anomali tespiti performansının belirgin şekilde arttığı (%5'ten %26.75'e yükseldiği) görülmektedir.

---

## 📈 4. Veri Setleri Arası Performans Farkları (Cross-Dataset)

[cite_start]Modellerin bir veri setinde eğitilip diğerlerinde test edilmesiyle elde edilen genellenebilirlik matrisi aşağıda sunulmaktadır[cite: 16]. 

[cite_start]**Tablo 2: Cross-Dataset Performans Karşılaştırması (seed=42)** [cite: 17]

| Eğitim → Test | Model | F1 Skor |
| :--- | :--- | :---: |
| SKAB → BATADAL | **Automata** | **0.5768** |
| SKAB → BATADAL | LSTM | 0.1123 |
| SKAB → BATADAL | 1D-CNN | 0.1289 |
| BATADAL → SKAB | **Automata** | **0.5164** |
| BATADAL → SKAB | LSTM | 0.1720 |
| BATADAL → SKAB | 1D-CNN | 0.1720 |

> [cite_start]🔍 **Yorum:** Olasılıksal Otomata, PAA ve SAX dönüşümleri üzerinden inşa edilen sembolik temsil sayesinde[cite: 76, 77], veri setine spesifik sayısal değerler yerine soyut örüntüleri öğrenmektedir. Bu durum, çapraz veri seti genellenebilirliğinde derin öğrenme modellerine kıyasla belirgin bir avantaj sağlamıştır.

---

## 🛡️ 5. Çevre Senaryoları (Gürültü ve Unseen Veri)

[cite_start]Modellerin veri kalitesindeki düşüşlere (Gaussian gürültü) ve test aşamasında daha önce gözlemlenmemiş örüntülere (unseen patterns) karşı direnci test edilmiştir[cite: 11, 80]. [cite_start]Otomata modelinde unseen durumlar için Levenshtein (Edit Distance) algoritması uygulanarak en yakın örüntüye eşleme mekanizması kullanılmıştır[cite: 81].

Aşağıda, en geniş konfigürasyon olan **$w=5, a=4$** için gürültü ve unseen (görülmemiş) veri senaryolarındaki model davranışları yer almaktadır:

### 🛡️ 5.1 BATADAL Veri Seti ($w=5, a=4$):
* **Orijinal F1:** LSTM: **%57.41** | CNN: %0.00 | Otomata: %17.94
* **Gürültülü F1 (%10 Elektriksel Gürültü):** LSTM: **%57.04** | CNN: %0.00 | Otomata: %17.94
* **Unseen F1 (%50 Genlik Artışı):** LSTM: **%57.41** | CNN: %0.00 | Otomata: %17.94 (Otomata Unseen Map. Accuracy: %14.56, Detection Rate: 1.0)

### 🛡️ 5.2 SKAB Veri Seti ($w=5, a=4$):
* **Orijinal F1:** LSTM: %80.43 | CNN: **%82.82** | Otomata: %26.75
* **Gürültülü F1 (%10 Elektriksel Gürültü):** LSTM: %79.64 | CNN: **%82.18** | Otomata: %28.15
* **Unseen F1 (%50 Genlik Artışı):** LSTM: %80.43 | CNN: **%82.82** | Otomata: %26.75 (Otomata Unseen Map. Accuracy: %23.25, Detection Rate: 0.55)

> 💡 **Bulgu:** Tüm modeller gürültü eklenmiş veriye karşı yüksek direnç göstermiştir. Unseen veri senaryosunda, Otomata'nın Levenshtein tabanlı eşleme mekanizması modelin çökmesini engellemiş ve stabilite sağlamıştır.

---

## ⚙️ 6. Parametre Etkileri ve Duyarlılık Analizi

[cite_start]İki aşamalı deney tasarımı ile Otomata modelinin iç parametrelerinin (Window Size ve Alphabet Size) performans üzerindeki etkileri analiz edilmiştir[cite: 20, 91]. [cite_start]Parametreler 3, 4, 5 ve 6 değerleri üzerinden test edilmiştir[cite: 96, 97].

[cite_start]**Tablo 4: Automata Parametre Duyarlılık Analizi - SKAB Veri Seti (F1-score)** [cite: 21]

| Alphabet \ Window Size | Değer = 3 | Değer = 4 | Değer = 5 | Değer = 6 |
| :---: | :---: | :---: | :---: | :---: |
| **Değer = 3** | 0.030 | 0.030 | 0.046 | 0.080 |
| **Değer = 4** | 0.051 | 0.098 | 0.149 | 0.231 |
| **Değer = 5** | 0.088 | 0.189 | 0.264 | 0.288 |
| **Değer = 6** | 0.133 | 0.259 | 0.323 | **0.333** |

> ✅ Parametre değişimlerinin performans üzerindeki analizi sonucunda SKAB veri seti için optimal konfigürasyon `window_size=6` ve `alphabet_size=6` olarak belirlenmiştir.

---

## 🖼️ 7. Değerlendirme Görselleri ve Olasılıksal Açıklanabilirlik

### 7.1 Confusion Matrix Karşılaştırması

[cite_start]Aşağıdaki karmaşıklık matrisleri, sınıflandırma performansının ve hata dağılımlarının görselleştirilmesini sağlamaktadır[cite: 229].

| LSTM Hata Dağılımı | Automata Hata Dağılımı |
|:---:|:---:|
| ![LSTM Confusion Matrix](results/LSTM_confusion_matrix.png) | ![Automata Confusion Matrix](results/Automata_confusion_matrix.png) |

### 7.2 ROC Eğrisi

[cite_start]Derin öğrenme modellerinden LSTM için oluşturulan ROC eğrisi aşağıda sunulmuştur[cite: 230].

| LSTM ROC Eğrisi |
|:---:|
| ![LSTM ROC](results/LSTM_roc_curve.png) |

### 7.3 Otomata State Diagram ve Geçiş Olasılıkları

[cite_start]Olasılıksal otomata modeli, durumlar arası geçiş olasılıklarını frekans tabanlı olarak öğrenir ve bu dizilerin olasılığını ardışık geçiş olasılıklarının çarpımı ile hesaplar[cite: 171, 174]. [cite_start]Düşük olasılığa sahip geçişler anomali adayı olarak işaretlenir[cite: 176].

[cite_start]Aşağıdaki görseller sistemin durum geçişlerini ve olasılık dağılımlarını temsil etmektedir[cite: 231, 232].

| Automata State Diagram | Transition Probability Heatmap |
|:---:|:---:|
| ![Automata State Diagram](results/automata_state_diagram.png) | ![Transition Heatmap](results/automata_transition_heatmap.png) |

### 7.4 Olasılıksal Karar Gerekçelendirmesi (XAI)

[cite_start]Modelin karar süreci, olasılıksal geçişler üzerinden matematiksel olarak gerekçelendirilerek JSON formatında raporlanmaktadır[cite: 156, 202]. [cite_start]Bu modül, mevcut durumu, gözlemlenen örüntüyü, uygulanan eşlemeyi ve nihai güven skorunu üretir[cite: 158, 159, 160, 162, 166, 169].

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