# Bilimsel Analiz ve Deney Raporu

**Giriş**
Bu tamamlayıcı doküman, *"From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis"* başlıklı ana projenin raporunda yer alması gereken kapsamlı deney sonuçlarını ve detaylı tablo dökümlerini içermektedir. Deneyler 5 farklı random seed kullanılarak BATADAL ve SKAB veri setleri üzerinde gerçekleştirilmiştir.

## 1. Temel Performans ve Stabilite
Aşağıdaki tablo, modellerin iki farklı veri seti üzerindeki ortalama F1-skorlarını ve 5 farklı random seed ile elde edilen standart sapma değerlerini göstermektedir. Projemizde derin öğrenme kıyaslaması için LSTM ve 1D-CNN kullanılmıştır.

**Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)**

| Model | SKAB | BATADAL |
| :--- | :--- | :--- |
| **LSTM** | 0.8012 ± 0.0210 | 0.0000 ± 0.0000 |
| **1D-CNN** | 0.7854 ± 0.0340 | 0.0000 ± 0.0000 |
| **Automata** | 0.5430 ± 0.0410 | 0.1766 ± 0.0000 |

> *Not: Veri dengesizliğinin çok yüksek olduğu BATADAL veri setinde hiperparametre optimizasyonu yapılmamış Derin Öğrenme modelleri tüm verileri "Normal" olarak sınıflandırdığı için F1 skorları 0.0 olarak hesaplanmıştır. Buna karşın Otomata modeli eğitimsiz (baseline) koşullarda dahi %17.6 F1 skoru ile anomali yakalayabilmiştir.*

## 2. Gürültü ve Unseen Veri Analizi (Robustness)
Modellerin veri kalitesindeki düşüşlere ve daha önce karşılaşılmamış örüntülere (unseen patterns) karşı ne kadar dirençli olduğunu ölçmek için Gaussian gürültü eklenmiş veri seti ve görülmemiş veri senaryosu test edilmiştir.

**Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi (BATADAL & SKAB Ortalama)**

| Model | Orijinal (F1) | Gürültülü (F1) | Unseen Analizi (Det. Rate) | Unseen Analizi (Map. Acc.) |
| :--- | :--- | :--- | :--- | :--- |
| **LSTM** | 0.4006 | 0.3812 | N/A (Black-Box) | N/A |
| **1D-CNN** | 0.3927 | 0.3705 | N/A (Black-Box) | N/A |
| **Automata** | 0.3598 | 0.3598 | 98.4% | 94.2% |

> *Otomata modeli, XAI (Açıklanabilirlik) kapasitesi sayesinde daha önce görülmemiş örüntüleri (unseen patterns) Levenshtein mesafesi ile en yakın duruma eşleyebilmiş ve eşleme doğruluğu (Mapping Accuracy) %94.2 olarak ölçülmüştür.*

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

**Tablo 5: Modellerin Çalışma Süresi (Runtime) Karşılaştırması**

| Model | Training Time (sn) | Inference Time (sn) |
| :--- | :--- | :--- |
| **LSTM** | 124.5 | 4.2 |
| **1D-CNN** | 86.3 | 2.1 |
| **Automata** | **1.8** | **0.4** |

> *Otomata modeli, PAA ve SAX dönüşümleri sayesinde ağırlık (weight) güncellemesi gerektirmediğinden Derin Öğrenme modellerine kıyasla eğitim (Training) süresinde ~70 kat, çıkarım (Inference) süresinde ~10 kat daha hızlı çalışmaktadır.*
