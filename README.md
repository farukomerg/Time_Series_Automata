# From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![NetworkX](https://img.shields.io/badge/NetworkX-3.0%2B-green)

Bu proje, zaman serisi anomali tespiti (Time Series Anomaly Detection) problemlerinde derin öğrenme (Black-Box) modelleri ile olasılıksal otomat (Probabilistic Automata) algoritmalarını karşılaştırmak ve "Açıklanabilir Yapay Zeka (XAI)" kavramını otomatlar aracılığıyla sunmak amacıyla geliştirilmiştir.

## 🚀 Proje Hakkında
Günümüzde LSTM ve 1D-CNN gibi derin öğrenme modelleri zaman serisi analizinde yüksek doğruluk (Accuracy) oranlarına ulaşsa da, modelin "neden" o kararı verdiği anlaşılamamaktadır (Black-box problemi). Bu proje, sürekli zaman serisi verisini PAA ve SAX yöntemleriyle ayrık sembollere (alfabe) dönüştürerek olasılıksal bir durum makinesi (Probabilistic Automaton) inşa eder.

**Öne Çıkan Özellikler:**
* **Data Leakage Koruması:** SKAB ve BATADAL veri setleri için özel Train/Val/Test bölütlemesi ve izolasyon.
* **Derin Öğrenme Kıyaslaması:** PyTorch tabanlı LSTM ve 1D-CNN modellerinin 5 farklı seed ile entegre eğitimi.
* **Açıklanabilirlik (XAI):** Otomata modelinin her bir tahmini, "Güven Skoru", "Geçiş Olasılığı" ve "Levenshtein Eşleşmesi" ile detaylı bir **JSON** formatında dışarı aktarılır.
* **Modern Görselleştirme:** Matplotlib ve NetworkX ile şık Confusion Matrix, ROC eğrisi ve Durum Geçiş (State Transition) ağları.
* **Dayanıklılık Testleri:** Modeller Gaussian Noise (Gürültü) ve daha önce görülmemiş veri (Unseen Patterns) senaryoları altında Wilcoxon istatistik testi ile sınanır.

## ⚙️ Kurulum ve Kullanım
Proje ortamında gerekli kütüphaneleri kurmak için:

```bash
pip install -r requirements.txt
```

Projeyi ve test senaryolarını çalıştırmak için ana orkestrasyon dosyasını çağırmanız yeterlidir:
```bash
python src/main.py
```
*(Not: Test işlemleri 5 farklı seed, 2 farklı veri seti ve 3 farklı senaryoda koştuğundan işlemci gücünüze bağlı olarak 30-40 dakika sürebilir. Sonuçlar `results/` klasörüne görsel, CSV ve JSON formatında kaydedilir.)*

## 📊 Rapor ve Bilimsel Çıktılar
Projenin performans ve stabilite analizleri, parametre duyarlılığı ve detaylı metrik sonuçları için lütfen **[RAPOR.md](RAPOR.md)** dosyasına göz atın.
