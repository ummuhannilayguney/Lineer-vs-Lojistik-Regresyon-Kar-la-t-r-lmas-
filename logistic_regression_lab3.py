# ==============================================================================
# BİRİNCİ KOD: LOJİSTİK REGRESYON MODELİ
# ==============================================================================
import pandas as pd  # Veri okuma, düzenleme ve tablo oluşturma işlemleri için pandas kütüphanesini projeye dahil eder.
import numpy as np  # Vektörel hesaplamalar ve ızgara (meshgrid) yapıları oluşturmak için numpy kütüphanesini içe aktarır.
import matplotlib.pyplot as plt  # Veriyi görselleştirmek ve karar sınırlarını çizmek için matplotlib grafikleri dahil eder.
import seaborn as sns  # Grafikleri görsel açıdan daha profesyonel ve estetik hale getirmek için seaborn kütüphanesini ekler.
from pathlib import Path  # Dosya yollarını işletim sistemi bağımsız ve dinamik şekilde yönetmek için tanımlı Path kütüphanesini dâhil eder.
from sklearn.model_selection import train_test_split  # Veriyi modeli eğitmek ve test etmek üzere alt gruplara bölmek için içe aktarır.
from sklearn.preprocessing import StandardScaler  # Modelin hatasız çalışması için değişkenleri aynı sayısal ölçeğe getiren sınıfı dahil eder.
from sklearn.linear_model import LogisticRegression  # İkili sınıflandırma (0 veya 1) yapmak için lojistik regresyon modelini projeye ekler.
from sklearn.metrics import (  # Performansın sayısal dökümünü almak için scikit-learn içerisindeki ölçüm metriklerini topluca açar.
    accuracy_score,  # Doğru bilinen tahminlerin tüm tahminlere oranını (doğruluk) hesaplayan formülü dahil eder.
    confusion_matrix,  # Gerçek ile tahmini karşılaştırarak hataların tiplerini (TP, TN, FP, FN) gösteren matrisi ekler.
    precision_score,  # Pozitif olduğu öngörülenlerin gerçekten ne kadarının pozitif olduğunu hesaplamak için dahil eder.
    recall_score,  # Gerçekte pozitif olan verilerin ne kadarının model tarafından yakalanabildiğini hesaplayan metriği ekler.
    f1_score  # Hassasiyet (precision) ve duyarlılığın (recall) ağırlıklı ortak başarısını ölçmek için F1 skorunu dahil eder.
)

def load_and_preprocess_data(filepath="Social_Network_Ads.csv"):  # Ham veriyi okuyup makine öğrenmesine hazır hale getiren veri işleme fonksiyonu.
    try:  # Dosyanın aniden bulunamaması durumunda sistemi çökertmemek için hata denetleme (try) bloğunu başlatır.
        script_dir = Path(__file__).resolve().parent  # Çalışan Python dosyasının (script'in) bulunduğu klasörün tam yolunu dinamik olarak tespit eder.
        full_path = script_dir / filepath  # Dosyanın ismini, bulunduğu klasörün dizin yolu ile birleştirerek kesin konumunu oluşturur.
        df = pd.read_csv(full_path)  # CSV uzantılı veri dosyasını okuyarak bir pandas veri çerçevesine (DataFrame) kopyalar.
    except FileNotFoundError:  # Dosya çalışma dizininde bulunmazsa oluşacak olan bulunamama hatasını yakalar.
        raise FileNotFoundError(f"Hata: {filepath} dosyası {script_dir} konumunda bulunamadı. Lütfen dizini kontrol edin.")  # Sistemi güvenli klasöre yönlendirecek uyarıyı fırlatır.

    df = df.drop(columns=["User ID", "Gender"])  # Kişniyin satın alma eğilimini etkilemeyen, yersiz değişkenleri matristen siler.
    
    X = df[["Age", "EstimatedSalary"]]  # Modele girdi olacak ve tahmine yön verecek olan 'Yaş' ve 'Tahmini Maaş' sütunlarını filtreler.
    y = df["Purchased"]  # Yapay zekanın ulaşmaya çalışacağı cevap niteliğindeki 'Satın Alındı mı?' kararını etiket (y) olarak belirler.
    
    return X, y  # Diğer işlemlerde rahatça kullanılabilmesi için X (özellikler) ve y (hedef) değişkenlerini fonksiyonun dışına verir.

def evaluate_model(y_true, y_pred):  # Yapay zekanın sınavlarda aldığı puanlar gibi, tahminlerinin ne kadar isabetli olduğunu raporlayan fonksiyon.
    accuracy = accuracy_score(y_true, y_pred)  # Hedefteki satın alma eylemini toplamda yüzde kaç ihtimalle doğru yakaladığını hesaplar.
    cm = confusion_matrix(y_true, y_pred)  # Modelin doğru bildiklerini diagonale, yanıldıklarını ise diğer köşelere yazan 2x2 matrisi kurar.
    precision = precision_score(y_true, y_pred, zero_division=0)  # Model 'Alacak' dediklerinin kaçında haklı çıktığını tespit ederek başarıyı ölçer.
    recall = recall_score(y_true, y_pred, zero_division=0)  # Gerçekte 'Alan' kişilerin ne kadarının yapay zekanın gözünden kaçmadığını ortaya koyar.
    f1 = f1_score(y_true, y_pred, zero_division=0)  # Precision ve Recall arasında dengesizlik varsa, uyumlu bir harmoni yakalamak adına F1 oranını çeker.

    print("\n" + "="*45)  # Konsol ekranında görsel karmaşayı ayırmak için kalın eşit çizgi bariyeri bastırır.
    print("LOJİSTİK REGRESYON - PERFORMANS METRİKLERİ")  # Bu metriklerin Lojistik Regresyon modeline ait olduğunu belirten başlık atar.
    print("="*45)  # Başlığı tasarımsal olarak alttan kapatmak ve tablo görünümü vermek için ayraç çeker.
    print(f"Doğruluk (Accuracy)  : % {accuracy*100:.2f}")  # Modelin genel net doğruluğunu % şeklinde, ondalıklı kısmın sadece 2 hanesini alarak yazar.
    print(f"Kesinlik (Precision) : % {precision*100:.2f}")  # Pozitif iddialardaki kesin karar başarısını yüzde olarak hizalı şekilde yansıtır.
    print(f"Duyarlılık (Recall)  : % {recall*100:.2f}")  # Hedefi şaşırmama ve doğru teşhis etme yeteneğini oransal olarak yüzdelik yazar.
    print(f"F1 Skoru (F1-Score)  : % {f1*100:.2f}")  # Özellikle veriler dengesizse önem taşıyan F1 doğruluk özetini matematiksel şekliyle ekrana verir.
    print("\nKarmaşıklık Matrisi (Confusion Matrix):")  # Ekrana 2x2 şeklindeki hata matrisinin geldiğini haber veren okunaklı ara başlık yazdırır.
    print(cm)  # Matrisi tam olarak satır ve sütundaki iç içe değerleri okunacak şekilde tablo halinde konsola çıkartır.

def plot_decision_boundary(X_scaled, y, model, title="Lojistik Regresyon Karar Sınırı"):  # Ölçeklenmiş test verilerini alıp modelin dünyayı nasıl ikiye ayırdığını çizen görselleştirme.
    x1_step = 0.01  # Izgara ağı çizilirken bilgisayarın yaş (x) ekseninde atacağı milimetrik, sık adımların aralığını tayin eder.
    x2_step = 0.01  # Benzer şekilde maaş (y) eksenindeki sanal noktacıkların aralıklarını küçük tutarak çizgiyi keskinleştirir.
    
    x1_min, x1_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1  # Çizilecek haritada yaş değerlerinin sağa ve sola taşmaması için sınır çekip marj bırakır.
    x2_min, x2_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1  # Haritanın dikey eksenindeki sınır değerlere (min, max) 1 birimlik sanal bir tampon bölgesi ekler.
    
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, x1_step),  # Belirlenen limitler ve adımlar ışığında, her noktanın birbirini gördüğü matematiksel mesh (örgü) kurar.
                         np.arange(x2_min, x2_max, x2_step))  # Aynı örgüyü dikey boyut (y) için entegre ederek alanı milyonlarca koordinata dönüştürür.
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Dev örgüdeki her tekil X ve Y noktası için "Sence satın alır mı?" diye sorup lojistik denklemi uygular.
    Z = Z.reshape(xx.shape)  # Çıkan milyon tane düz tahmin cevabını, haritanın orijinal dikdörtgen koordinat şekline geri kıvırarak oturtturur.
    
    plt.figure(figsize=(10, 6))  # 10 birim genişlik ve 6 birim yükseklikte olan profesyonel bilgisayar grafiği çerçevesi oluşturur.
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')  # Modelin tahmini '0' olan bölgeleri açık maviye, '1' olanları açık kırmızıya boyayarak karar yüzeyini çizer.
    
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='bwr', edgecolor='k', s=60)  # Gerçek veri noktalarını haritaya daire (scatter) olarak oturtur.
    
    plt.title(title, fontsize=14, fontweight='bold')  # Çizilen haritanın merkezine okunaklı, kalın ve resmi bir ana başlık yerleştirme işlemi yapar.
    plt.xlabel('Yaş (StandardScaler ile Ölçeklenmiş)', fontsize=12)  # X ekseninde verinin ne anlama geldiğini netleştiren ve formatı aktaran alt metin basar.
    plt.ylabel('Tahmini Maaş (StandardScaler ile Ölçeklenmiş)', fontsize=12)  # Y ekseninin tahmini maaşı baz aldığını ancak skalasının daraltıldığını yan tarafa yazar.
    
    import matplotlib.lines as mlines  # Lejant kutusunu (açıklama sekmesini) manuel olarak renklendirmek için dâhil edilir.
    handle_0 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, markeredgecolor='k', alpha=0.5, label='0: Satın Almadı')  # Lejant üzerinde doğru mavi noktayı yaratır.
    handle_1 = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, markeredgecolor='k', alpha=0.5, label='1: Satın Aldı')  # Lejant üzerinde doğru kırmızı noktayı yaratır.
    plt.legend(handles=[handle_0, handle_1], title="Gerçek Sınıflar")  # Garantili renk ayrımı sunan güncel listeyi grafiğe basar.
    
    plt.tight_layout()  # Kutu içine sığmayan ya da birbirine giren etiket/eksen grafiklerini kenarlara muntazaman optimize ederek boşlukları düzenler.
    plt.show()  # Arka planda kurgulanan ve işlemciye yüklenen tüm çizim objelerini aktif olarak kullanıcının pop-up ekranında sunar.

def main():  # Bütün işlemleri senkronize sırayla çalıştırıp tetiklemeyi tek ana çatı altında toplayan başlatıcı iskelet fonksiyon.
    X, y = load_and_preprocess_data()  # Analiz edilecek veriyi fonksiyon vasıtasıyla CSV'den çekerek bağımlı/bağımsız olarak böldürür.
    
    X_train, X_test, y_train, y_test = train_test_split(  # sklearn kütüphanesi yardımıyla veriyi öğrenmeye ve sınava tabi tutmaya yarayan bölme işlemine başlar.
        X, y,  # Bölünecek iki asıl hedef listesini, özellik kolonları (X) ve sonuç etiketleri (y) olarak sunar.
        test_size=0.25,  # İçerideki verinin yüzde tam olarak 25'ini makine hiç görmesin diye teste (quiz) saklamak üzere ayırır.
        random_state=42,  # Veri setini bölerken bir kargaşa yaratmadan her kod çalıştığında aynı veri listesini dağıtsın diye rastgeleliği sabitler.
        stratify=y  # Özellikle 0 ve 1 oranlarını bozmayıp eğitim kümesine de, teste de aynı popülasyon dengesinde ve sayısında dağılmayı deklare eder.
    )
    
    scaler = StandardScaler()  # Feature scaling amacı ile standart normal dağılım mekanizmasını kurmak üzere obje örneğini hafızaya alır.
    X_train_scaled = scaler.fit_transform(X_train)  # Eğitim seti için ortalama ve sapmaları ezberler, ardından veriyi -3 ve 3 birimleri arasına küçültmek için uygular.
    X_test_scaled = scaler.transform(X_test)  # Geleceğe dair sızdırılma (leak) olmasın diye sadece ezberlenmiş kalıplar baz alınarak sınav (test) setini de aynı aralığa çevirir.
    
    log_reg = LogisticRegression(random_state=42, max_iter=1000)  # Lojistik Regresyon beynini var eder, hata minimizasyonu yaparken max döngü adımını 1000'de limitler.
    log_reg.fit(X_train_scaled, y_train)  # Sınırları küçültülmüş veritabanını okuyan algoritmanın kuralları şekillendirme ve eğitilme fazını çalıştırır.
    
    y_pred = log_reg.predict(X_test_scaled)  # Eğitimi biten yazılımın, bilmediği sınav verisini kendi bulduğu denklemden geçirip 0 veya 1 sınıflandırmasını yapmasını ister.
    
    evaluate_model(y_test, y_pred)  # Elde edilen test kağıdıyla, gerçek orijinal sonuçlar yüzleştirilerek matematiksel doğruluğun raporlanmasını emreder.
    
    plot_decision_boundary(X_test_scaled, y_test.values, log_reg)  # Matematiksel denklemin geometrik haritaya dökülerek makinenin 2 boyutlu sınırlarını renklerle çizer.

if __name__ == "__main__":  # Çalıştırılan python.py dosyasının sadece içten komutla değil, modül kütüphanesi olarak çekilirse otomatik işlem yapmasını engeller.
    main()  # Sadece bu koddan çalıştırılmış onayı mevcut ise, ana süreç motorunu hareketlendirip kodun derleyicide akmasını sağlar.
