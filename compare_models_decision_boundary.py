import pandas as pd  # Veri setini tablo formatında okuyup işlemek için pandas kütüphanesini içe aktarır.
import numpy as np  # Sayısal hesaplamalar ve ızgara üretimi için numpy kütüphanesini içe aktarır.
import matplotlib.pyplot as plt  # Model karar sınırlarını görselleştirmek için matplotlib kütüphanesini içe aktarır.
from pathlib import Path  # Dosya yolunu güvenli ve platform bağımsız yönetmek için Path sınıfını içe aktarır.
from sklearn.model_selection import train_test_split  # Veriyi eğitim ve test olarak bölmek için gerekli fonksiyonu içe aktarır.
from sklearn.preprocessing import StandardScaler  # Özellikleri aynı ölçek düzeyine getirmek için StandardScaler sınıfını içe aktarır.
from sklearn.linear_model import LogisticRegression, LinearRegression  # Lojistik ve lineer regresyon modellerini içe aktarır.


def main() -> None:  # Programın ana çalışma akışını tanımlar.
    csv_path = Path(__file__).resolve().parent / "Social_Network_Ads.csv"  # Veri dosyasının bu script ile aynı klasördeki yolunu oluşturur.
    df = pd.read_csv(csv_path)  # CSV dosyasını DataFrame formatında belleğe yükler.

    df = df.drop(columns=["User ID", "Gender"])  # Analizde kullanılmayacak User ID ve Gender sütunlarını kaldırır.

    X = df[["Age", "EstimatedSalary"]]  # Model giriş özelliklerini Age ve EstimatedSalary olarak seçer.
    y = df["Purchased"]  # Tahmin edilmek istenen hedef değişkeni Purchased olarak ayırır.

    X_train, X_test, y_train, y_test = train_test_split(  # Veriyi eğitim ve test kümelerine böler.
        X,  # Özellik matrisi.
        y,  # Hedef vektörü.
        test_size=0.25,  # Test kümesi oranını %25 olarak belirler.
        random_state=42,  # Tekrarlanabilir sonuçlar için rastgelelik tohumunu sabitler.
        stratify=y,  # Sınıf oranını eğitim ve test kümesinde korumak için tabakalı bölme uygular.
    )

    scaler = StandardScaler()  # Özellik ölçekleme için StandardScaler nesnesini oluşturur.
    X_train_scaled = scaler.fit_transform(X_train)  # Eğitim verisinde ölçekleme parametrelerini öğrenip eğitimi dönüştürür.
    X_test_scaled = scaler.transform(X_test)  # Test verisini aynı ölçekleme parametreleri ile dönüştürür.

    logistic_model = LogisticRegression(random_state=42, max_iter=1000)  # Lojistik Regresyon modelini tanımlar.
    logistic_model.fit(X_train_scaled, y_train)  # Lojistik modeli ölçeklenmiş eğitim verisi ile eğitir.

    linear_model = LinearRegression()  # Lineer Regresyon modelini tanımlar.
    linear_model.fit(X_train_scaled, y_train)  # Lineer modeli ölçeklenmiş eğitim verisi ile eğitir.

    x1_min = X_test_scaled[:, 0].min() - 1.0  # İlk özelliğin (Age ölçekli) alt görselleştirme sınırını belirler.
    x1_max = X_test_scaled[:, 0].max() + 1.0  # İlk özelliğin (Age ölçekli) üst görselleştirme sınırını belirler.
    x2_min = X_test_scaled[:, 1].min() - 1.0  # İkinci özelliğin (EstimatedSalary ölçekli) alt görselleştirme sınırını belirler.
    x2_max = X_test_scaled[:, 1].max() + 1.0  # İkinci özelliğin (EstimatedSalary ölçekli) üst görselleştirme sınırını belirler.
    grid_step = 0.02  # Karar bölgelerinin daha net görünmesi için ızgara adım aralığını belirler.

    xx, yy = np.meshgrid(  # İki boyutlu özellik uzayı için yoğun bir ızgara üretir.
        np.arange(x1_min, x1_max, grid_step),  # İlk eksen için ızgara noktalarını oluşturur.
        np.arange(x2_min, x2_max, grid_step),  # İkinci eksen için ızgara noktalarını oluşturur.
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Izgara noktalarını model tahmini için iki sütunlu forma dönüştürür.

    logistic_grid_pred = logistic_model.predict(grid_points).reshape(xx.shape)  # Lojistik modelin ızgara üzerindeki sınıf tahminlerini hesaplar.

    linear_grid_continuous = linear_model.predict(grid_points).reshape(xx.shape)  # Lineer modelin ızgara üzerindeki sürekli tahminlerini hesaplar.
    linear_grid_class = (linear_grid_continuous >= 0.5).astype(int)  # Lineer tahminleri 0.5 eşik değeri ile sınıf etiketine dönüştürür.

    y_test_np = y_test.to_numpy()  # Test hedef değişkenini numpy dizisine dönüştürür.

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)  # İki modeli yan yana gösterecek iki alt grafik oluşturur.

    axes[0].contourf(xx, yy, logistic_grid_pred, alpha=0.35, cmap="coolwarm")  # Sol grafikte lojistik modelin karar bölgelerini çizer.
    axes[0].scatter(  # Sol grafikte gerçek sınıf 0 test noktalarını gösterir.
        X_test_scaled[y_test_np == 0, 0],
        X_test_scaled[y_test_np == 0, 1],
        c="steelblue",
        edgecolor="black",
        s=45,
        label="Gerçek Sınıf 0",
    )
    axes[0].scatter(  # Sol grafikte gerçek sınıf 1 test noktalarını gösterir.
        X_test_scaled[y_test_np == 1, 0],
        X_test_scaled[y_test_np == 1, 1],
        c="firebrick",
        edgecolor="black",
        s=45,
        label="Gerçek Sınıf 1",
    )
    axes[0].set_title("Lojistik Regresyon Decision Boundary")  # Sol grafik başlığını ayarlar.
    axes[0].set_xlabel("Age (StandardScaler sonrası)")  # Sol grafik x ekseni etiketini ayarlar.
    axes[0].set_ylabel("EstimatedSalary (StandardScaler sonrası)")  # Sol grafik y ekseni etiketini ayarlar.
    axes[0].legend(loc="upper left")  # Sol grafiğe sınıf açıklamalarını ekler.

    axes[1].contourf(xx, yy, linear_grid_class, alpha=0.35, cmap="coolwarm")  # Sağ grafikte lineer modelin eşik sonrası karar bölgelerini çizer.
    axes[1].contour(  # Sağ grafikte lineer modelin 0.5 eşik sınırını kesikli siyah çizgi ile gösterir.
        xx,
        yy,
        linear_grid_continuous,
        levels=[0.5],
        colors="black",
        linewidths=2,
        linestyles="--",
    )
    axes[1].scatter(  # Sağ grafikte gerçek sınıf 0 test noktalarını gösterir.
        X_test_scaled[y_test_np == 0, 0],
        X_test_scaled[y_test_np == 0, 1],
        c="steelblue",
        edgecolor="black",
        s=45,
        label="Gerçek Sınıf 0",
    )
    axes[1].scatter(  # Sağ grafikte gerçek sınıf 1 test noktalarını gösterir.
        X_test_scaled[y_test_np == 1, 0],
        X_test_scaled[y_test_np == 1, 1],
        c="firebrick",
        edgecolor="black",
        s=45,
        label="Gerçek Sınıf 1",
    )
    axes[1].set_title("Lineer Regresyon (0.5 Threshold) Decision Boundary")  # Sağ grafik başlığını ayarlar.
    axes[1].set_xlabel("Age (StandardScaler sonrası)")  # Sağ grafik x ekseni etiketini ayarlar.
    axes[1].legend(loc="upper left")  # Sağ grafiğe sınıf açıklamalarını ekler.

    plt.suptitle("İki Modelin Decision Boundary Karşılaştırması", fontsize=14)  # Tüm figür için üst başlık ekler.
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])  # Üst başlıkla çakışma olmaması için yerleşimi sıkılaştırır.

    output_path = csv_path.parent / "model_comparison_decision_boundary.png"  # Yan yana karşılaştırma görselinin kayıt yolunu belirler.
    plt.savefig(output_path, dpi=220)  # Karşılaştırma görselini yüksek çözünürlüklü PNG olarak kaydeder.
    plt.close()  # Bellekte açık figürü kapatarak kaynak kullanımını azaltır.

    print(f"Karşılaştırma görseli kaydedildi: {output_path.name}")  # Kaydedilen karşılaştırma dosyasının adını ekrana yazdırır.


if __name__ == "__main__":  # Script doğrudan çalıştırıldığında main fonksiyonunu çalıştırır.
    main()  # Programın ana akışını başlatır.
