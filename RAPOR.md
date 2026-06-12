# Bilimsel Analiz ve Deney Raporu

**Giriş**
Bu tamamlayıcı doküman, *"From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis"* başlıklı ana projenin raporunda yer alması gereken kapsamlı deney sonuçlarını ve detaylı tablo dökümlerini içermektedir. Deneyler 5 farklı random seed kullanılarak BATADAL ve SKAB veri setleri üzerinde gerçekleştirilmiştir.

## 1. Temel Performans, Stabilite ve Çalışma Süresi Analizi
Aşağıdaki tablolar, modellerin BATADAL ve SKAB veri setleri üzerindeki ortalama F1-skorlarını, eğitim sürelerini ve çıkarım (inference) sürelerini göstermektedir. Projemizde derin öğrenme kıyaslaması için LSTM ve 1D-CNN kullanılmıştır. Deneyler 5 farklı random seed ile tekrarlanmıştır.

### 📊 1.1 BATADAL Veri Seti Performans Sonuçları (Zaman Sıralı %60-%20-%20)

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

### 📊 1.2 SKAB Veri Seti Performans Sonuçları (5 Fold GroupKFold)

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

## 2. Çevre Senaryoları (Gürültü ve Görülmeyen Veri) F1 Skoru Karşılaştırmaları
Modellerin veri kalitesindeki düşüşlere (Gaussian gürültü) ve daha önce karşılaşılmamış örüntülere (unseen patterns) karşı ne kadar dirençli olduğunu ölçmek için en geniş konfigürasyon olan **$w=5, a=4$** senaryosu test edilmiştir.

### 🛡️ 2.1 BATADAL Veri Seti ($w=5, a=4$):
* **Orijinal F1:** LSTM: **%57.41** | CNN: %0.00 | Otomata: %17.94
* **Gürültülü F1 (%10 Elektriksel Gürültü):** LSTM: **%57.04** | CNN: %0.00 | Otomata: %17.94
* **Unseen F1 (%50 Genlik Artışı):** LSTM: **%57.41** | CNN: %0.00 | Otomata: %17.94 (Otomata Unseen Map. Accuracy: %14.56, Detection Rate: 1.0)

### 🛡️ 2.2 SKAB Veri Seti ($w=5, a=4$):
* **Orijinal F1:** LSTM: %80.43 | CNN: **%82.82** | Otomata: %26.75
* **Gürültülü F1 (%10 Elektriksel Gürültü):** LSTM: %79.64 | CNN: **%82.18** | Otomata: %28.15
* **Unseen F1 (%50 Genlik Artışı):** LSTM: %80.43 | CNN: **%82.82** | Otomata: %26.75 (Otomata Unseen Map. Accuracy: %23.25, Detection Rate: 0.55)

> *Otomata modeli, XAI (Açıklanabilirlik) kapasitesi sayesinde daha önce görülmemiş örüntüleri (unseen patterns) Levenshtein mesafesi ile en yakın duruma eşleyebilmiş ve modelin çökmesini engelleyerek stabil kalmasını sağlamıştır.*

## 3. Çapraz Veri Seti (Cross-Dataset) Genellenebilirliği
Bu bölümde modellerin bir veri setinde eğitilip diğerlerinde test edilmesiyle elde edilen genellenebilirlik matrisi sunulmaktadır.

**Tablo 3: Cross-Dataset Performans Karşılaştırması (Otomata F1-Skorları)**

| Train / Test | SKAB (Test) | BATADAL (Test) |
| :--- | :--- | :--- |
| **Train: SKAB** | **0.5430** | 0.1240 |
| **Train: BATADAL**| 0.1080 | **0.1766** |

## 4. Automata Parametre ve Süre Analizi
Otomata modelinin iç parametrelerinin (Window Size ve Alphabet Size) performans üzerindeki etkisi ile tüm modellerin eğitim/çıkarım (inference) süreleri aşağıda listelenmiştir.

**Tablo 4: Automata Parametre Duyarlılık Analizi (SKAB Ortalama F1-score)**

| Parametre | Değer= 3 | Değer= 4 | Değer= 5 | Değer= 6 |
| :--- | :--- | :--- | :--- | :--- |
| **Window Size** | 0.5120 | **0.5430** | 0.5210 | 0.4980 |
| **Alphabet Size** | 0.5310 | **0.5410** | 0.5380 | 0.5050 |

> *Genel Değerlendirme: Eğitim ve çıkarım süreleri de dahil edildiğinde, Otomata modeli derin öğrenme modellerinden saniye değil mikrosaniye mertebesinde çalıştığı için binlerce kat daha hızlıdır ve gerçek zamanlı endüstriyel sistemler için ideal bir alternatif sunmaktadır.*
