# Social Network Ads: Purchase Prediction 🛍️

Bu proje, Sosyal Ağ Kullanıcılarının yaş (**Age**) ve tahmini maaşlarına (**EstimatedSalary**) dayanarak belirli bir ürünü satın alma veya almama (**Purchased**) durumlarını analiz eden ve tahminleyen bir makine öğrenmesi (Denetimli Öğrenme / Supervised Learning) laboratuvar çalışmasıdır.

## 📋 Proje Özeti
Bu çalışmanın amacı, temel makine öğrenmesi algoritmalarını ve sınıflandırma mantığını uygulamalı bir şekilde kavramaktır. İki farklı yaklaşım üzerinden ikili sınıflandırma (Binary Classification) çözümleri üretilmiştir:
1. **Lojistik Regresyon**
2. **Lineer Regresyon** (0.5 Eşik/Threshold sınıflandırmasına uyarlanmış varyasyon)

Her iki model için bağımsız Python (`.py`) dosyaları geliştirilmiş, algoritmaların başarım ve karar sınırları (Decision Boundaries) görselleştirilerek karşılaştırılmıştır.

---

## ⚙️ Kurulum ve Gereksinimler

Projenin kendi cihazınızda bağımlılık sorunları yaratmadan çalışabilmesi için sisteminizde en az **Python 3.8+** kurulu olması tavsiye edilir. Aşağıdaki kütüphanelere ihtiyaç duyulmaktadır:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Gerekli kütüphaneleri yüklemek için terminalinizde aşağıdaki komutu çalıştırabilirsiniz:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📂 Dosya Yapısı

- `logistic_regression_lab3.py`: Lojistik analiz modelini eğiten, metrikleri (Accuracy, Confusion Matrix, Precision, Recall, F1) konsola bildiren ve modelin renk ayrışımlı karar sınırlarını çizen Python scripti.
- `linear_regression_lab3.py`: Geleneksel regresyon çıktısını olan, 0.5 çizgisi üzerinden ikili formata deviren (Thresholding) ve sonuçları lojistik model ile aynı test mantığı üzerinden sınıflayan Python scripti.
- `Social_Network_Ads.csv`: Toplamda 4 ana sütundan (User ID, Gender, Age, EstimatedSalary) ve hedef kolondan (Purchased) oluşan ham veri seti dosyası.
- `Lab03_UNilayG.txt`: Veri ön işlemeyi, sonuç analizlerini ve modeller arası istatistiksel karşılaştırmayı detaylandıran Proje/Hoca Teslim Raporu.
- `README.md`: Projenin şu an okumakta olduğunuz genel dokümantasyon dosyası.

---

## 🚀 Kullanım

Projedeki herhangi bir makine öğrenmesi modelini uçtan uca çalıştırıp sonuçları ekranda görmek için, terminalinizi bu klasör konumunda açıp Python ile çalıştırmanız yeterlidir:

**Lojistik Regresyon Modelini İncelemek İçin:**
```bash
python logistic_regression_lab3.py
```

**Lineer Regresyon (Eşik Uyarlamalı) Modelini İncelemek İçin:**
```bash
python linear_regression_lab3.py
```

> **Not:** Scriptler, `pathlib` kütüphanesi sayesinde dinamik dosya tespiti yapacak şekilde yazılmıştır. Yani dosyayı dizin fark etmeksizin çağırdığınızda sistem çökmeden `Social_Network_Ads.csv` dosyasını kendi ana konumundan kusursuzca bulacaktır. Çıkan sınır grafiklerini inceledikten sonra kapatarak terminaldeki performans metriklerini okuyabilirsiniz.

---

## 📊 Öne Çıkan Sonuçlar ve Mimari Detaylar

- **Feature Scaling (Özellik Ölçeklendirme):** "Yaş" (onlar basamağında) ve "Tahmini Maaş" (binler basamağında) arasındaki devasa numaratik farkın denklemlerde ağırlık anomalisi (bias) yaratmaması amacıyla, özellikler `StandardScaler` mekanizması ile matematiksel olarak standartlaştırılmıştır.
- **Model Benzerlikleri ve Eğitim Dengesi:** Veriler %75 Train ve %25 Test (%0-%1 sınıf dağılımı korunacak olan `stratify=y` özelliği eşliğinde) olarak tabakalı bölünmüştür. Verinin lineer bir çizgi üzerinden rahat algılanabilir bir dağılıma (linearly separable) sahip olması neticesinde; 0.5 Threshold kullanan **Lineer Modeli** ve sigmoid eğrisi kullanan **Lojistik Modeli** bu ayrkımdaki test gruplarında birbiri ile birebir tamamen aynı matematiksel istatistik tutarlılığına (Doğruluk: %84.00, F1 Skoru: %75.00) ulaşmayı başarmıştır. 
- **Yazılımsal Hassasiyet (Lejant Doğrulaması):** Yapılan görselleştirmeler sırasında Matplotlib altyapısının test verisi renkleriyle sınıf isimlerini yanlış otomasyona tuttuğu tespit edilmiştir (İzleyiciye 0 sınıfını kırmızı, 1 sınıfını mavi aktaran ters eşleşme kroniği). Bu görsel illüzyonu önlemek ve kesin doğrulama sunmak için her iki grafiğe de `matplotlib.lines.Line2D` handle'ları (manuel işaretçiler) uygulanarak endüstri standardında net görsel dökümler sağlanmıştır. Bütün bu akıllı işlemlerin sebepleri, temiz kod (Clean Code) standartlarınca her bir komut satırının yanına Türkçe olarak yorumlanmıştır.

---
*Geliştirici:* **Ümmühan Nilay Güney** 🎓 
*Ders Kurumu:* Denetimli Öğrenme / Makine Öğrenmesi (Lab 3)
