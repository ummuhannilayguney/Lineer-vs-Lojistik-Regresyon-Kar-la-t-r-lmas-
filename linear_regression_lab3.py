# ==============================================================================
# İKİNCİ KOD: LİNEER REGRESYON MODELİ VE 0.5 THRESHOLD (EŞİK) SINIFLANDIRMASI
# ==============================================================================
import pandas as pd  # Verileri veritabanı benzeri tablolar ile esnek ve fonksiyonel yapılandırmak için pandas'ı import eder.
import numpy as np  # Doğrusal cebir, dizi mekanikleri ve ağ (mesh) matematiği kurabilmek üzerine uzmanlaşan kütüphaneyi ekler.
import matplotlib.pyplot as plt  # Çıkarılan sayısal neticeleri geometrik bir düzlem üzerinden piksellere yayarak görsel çıktı sağlamak için alır.
import seaborn as sns  # Çizilen haritayı göz alıcı, modern formatta ve renk paletleriyle şekillendirmek için dahil işlemi gerçekleştirir.
from pathlib import Path  # Dosya yollarını işletim sistemi bağımsız ve dinamik şekilde yönetmek için tanımlı Path kütüphanesini dâhil eder.
from sklearn.model_selection import train_test_split  # Elimizdeki pastayı eğitim (%75) ile deneme/test (%25) şeklinde ayrıştırmak için formülü içeri sunar.
from sklearn.preprocessing import StandardScaler  # Farklı varyanslarda olan kolonları tek standarta sokup, modelin kafasının karışmasını önlemek için içeri aktarır.
from sklearn.linear_model import LinearRegression  # İki düzlem arasında en uygun kesinti doğrusunu çizmeye yarayan regresyon modelinin modülünü dökümana çeker.
from sklearn.metrics import (  # Hem sürekli tahmin (Lineer) hem sınıf tahmin denemeleri performansı için ilgili scikit-learn metriklerine kapı açar.
    mean_squared_error,  # Makinenin hatasının karelerinin ortalamasını, yani ceza maliyetini (MSE) hesaplatmaya yönelik aracı getirir.
    r2_score,  # Eğitilen çizginin, verilerin asıl davranışını yüzde ne kadar oranla açıklayabildiğini kanıtlayan R-kare bağıntısını çeker.
    accuracy_score,  # 0.5 eşiğine zorlandıktan sonra doğruluk/isabet oranını saydırmak için kullanılan sınıflandırma modülünü alır.
    confusion_matrix,  # Lineer model bazında kaç tanesi yanlış bölgeye düştü inceleyebilmek için karmaşıklık dökümünü listeler.
    precision_score,  # Eşik çizgisi üzerindekiler referans alındığında "Ben buna aldım diyorsam almıştır" iddialarını doğrulayan modül.
    recall_score,  # Eşik çizgisinin modelde oluşturduğu kısıt sonucu kaç tane gerçeği kapsayıp kaçını gözden kaçırdığını ölçümleyen komut.
    f1_score  # Eğitimsiz/dengesiz dağılımlı setlerde threshold (eşik) değerlerinin harmonik ağırlıklı ortak başarısını tespit eder.
)

def load_and_preprocess_data(filepath="Social_Network_Ads.csv"):  # Ham CSV dokümanının sisteme nüfuz edip işleneceği aşamanın fonksiyonu.
    try:  # Veriyi diskten çekme işlemini güvenli bölgeden yapar, eğer isim yanlış veya dosya silinmişse sistemi kurtarma yolunu arar.
        script_dir = Path(__file__).resolve().parent  # Çalışan Python dosyasının (script'in) bulunduğu klasörün tam yolunu dinamik olarak tespit eder.
        full_path = script_dir / filepath  # Dosyanın ismini, bulunduğu klasörün dizin yolu ile birleştirerek kesin konumunu oluşturur.
        df = pd.read_csv(full_path)  # CSV verisindeki virgülleri ayrıştırıp kolon dizaynı haline dökerek pandas data frame'i üzerinden var eder.
    except FileNotFoundError:  # Veri dosyasının bulunamaması gibi ölümcül hatalarla karşılaşırsa bu spesifik bloka dallanma emri yollar.
        raise FileNotFoundError(f"Hata: {filepath} dosyası {script_dir} konumunda bulunamadı. Lütfen dizini kontrol edin.")  # Geliştiriciye veya kullanıcıya problemi aktarıp kodu kırar.

    df = df.drop(columns=["User ID", "Gender"])  # Tahmin algoritmasını gereksiz istatistik arayışına sokmamak için kimlik ve cinsiyet yapısını imha eder.
    
    X = df[["Age", "EstimatedSalary"]]  # Modele referans oluşturacak sütunları bağımsız girdi (Features) olan X matrisine kopyalayıp korur.
    y = df["Purchased"]  # Sistemin eğitilmesi ile öngördürmesi planlanan çıktı olan satın alma hedefini y (Target) vektörüne deklare eder.
    
    return X, y  # Oluşan tertemiz ve bölünmüş yapıyı daha genel amaçlı manipülasyon için programın hafızasından dışarı ulaştırır.

def evaluate_linear_model(y_true, y_pred_continuous, threshold=0.5):  # Hem sürekli hem eşiklendirilmiş başarı oranlarını hesaplama ve yazdırma yönergesi.
    mse = mean_squared_error(y_true, y_pred_continuous)  # Regresyon noktalarının gerçek y eksenine uzaklıkların kare maliyet toplamını ölçer.
    r2 = r2_score(y_true, y_pred_continuous)  # Varyansların birbirini açıklayabilmesi ile şekillenen bağımsız değişken uyum skoru olan R-kare'yi hesaplar.
    
    y_pred_class = (y_pred_continuous >= threshold).astype(int)  # 0.5 kesme değerinin üstünü 1 (Satın Aldı), altını 0 (Almadı) şeklinde manipüle edip zorunlu sınıflandırır.
    
    accuracy = accuracy_score(y_true, y_pred_class)  # 0.5 eşiği vasıtasıyla uydurduğumuz sınıfın orijinal gerçeğe oranını genel yüzdeleme bazında elde eder.
    cm = confusion_matrix(y_true, y_pred_class)  # Kesme çizgisi hatası ile oluşan "Eksiği fazla tahmin ettim", "Alanı da almadı sandım" istatistik matrisini çizer.
    precision = precision_score(y_true, y_pred_class, zero_division=0)  # Threshold üstünde kalanların kendi içinde ne kadar sağlam gerçeklik barındırdığını hesaplar.
    recall = recall_score(y_true, y_pred_class, zero_division=0)  # Threshold altındaki ve üstündeki gerçekten alan kişilerin sezilebilme performansını verir.
    f1 = f1_score(y_true, y_pred_class, zero_division=0)  # Sınıflandırma problemi olan durumlarda lineer hataların F1 algoritmasındaki ahengini sayar.

    print("\n" + "="*45)  # Konsolda görüntü temizliği sağlamak ve görsel olarak algıyı odaklaştırmak için sınır şeridini çizer.
    print("LİNEER REGRESYON - PERFORMANS METRİKLERİ")  # Elde edilecek dökümlerin temel lineer metoda ait olduğunu betimleyen resmi ana başlığı yazdırır.
    print("="*45)  # Tasarım bloğunu estetik açıdan alt çizgi ile destekleyerek döküm kısmına resmi geçiş yolunu aydınlatır.
    print(f"Sürekli Tahmin MSE      : {mse:.4f}")  # Modelin ana algoritması olan regresyonun asıl maliyeti (hatası) formatında 4 küsuratla basar.
    print(f"Sürekli Tahmin R2       : {r2:.4f}")  # Verideki dağılımın Lineer modeli hangi oranda tuttuğunu R-Kare verisi eşliğinde ekrana sığdırır.
    print("-" * 45)  # Regresyon doğası ile Threshold(Eşik) dünyasını kesmek ve ayrıştırmak için ince tireli ayırıcı bariyer kurar.
    print(f"Denklem Eşiği (Threshold): {threshold}")  # Model sürekli değerleri yuvarlamak için kestiği noktayı ve toleransını ekrandan fısıldar.
    print(f"Doğruluk (Accuracy)     : % {accuracy*100:.2f}")  # Çizgi üzerinden sıfıra ve bire dönüştürülen mantığın 100 üzerinden geçerlilik yüzdesini sunar.
    print(f"Kesinlik (Precision)    : % {precision*100:.2f}")  # Model pozitif olarak atandırdıklarından yüzde kaç doğruya erişti diye testini tamamlayıp basar.
    print(f"Duyarlılık (Recall)     : % {recall*100:.2f}")  # Sistem, bütün 1 gerçeklerinin yüzde ne kadarını bu eşik ile içine aldı analizini görselleştirir.
    print(f"F1 Skoru (F1-Score)     : % {f1*100:.2f}")  # F1'in ortalamalı harman verisini ekranda okunaklı yüzdeler bütünü halinde sergiler niteliktetir.
    print("\nKarmaşıklık Matrisi (Eşiklendirilmiş Sınıflandırma):")  # Eşik çizgisi nedeniyle değişen hataların listeleneceği başlık ibaresini beyan eder.
    print(cm)  # Numpy serisine çevrilen Hata-Matris listesinin formüllerden arındırılmış yalın 2x2 matrisini yazdırır.

def plot_decision_boundary(X_scaled, y, model, threshold=0.5, title="Lineer Regresyon (Threshold) Karar Sınırı"):  # Lineer modelin alanını renklerle ayırdığı haritalandırma motorunu besleyen grafik bölümü.
    x1_step, x2_step = 0.01, 0.01  # Izgaradaki çözünürlüğü maksimize edip eğrinin veya düzlemin pixellenmemesi için düşük çözünürlük ayarı atar.
    
    x1_min, x1_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1  # Grafik soluna ve sağına yaş üzerinden taşmasın diye grafiğe esnek 1 birim limit verir.
    x2_min, x2_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1  # Maaşların skalasına sığacak şekilde grafiği alttan üstten kırpıp, marjinal mesafe koydurur.
    
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, x1_step),  # Yüzlerce noktanın X ekseni ve Y ekseni uzayını, birbirine entegre mesh matrisi yapar.
                         np.arange(x2_min, x2_max, x2_step))  # Eksenin dikey boyutu ve yatay boyutu matematiksel sanal alan (örümcek ağı) olgusu yaratır.
    
    Z_continuous = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Ağ üzerindeki her zerre için regresyon doğrusunu çalıştırıp sanal küsurat tahminlerini derler.
    Z_class = (Z_continuous >= threshold).astype(int)  # 0.5 ve üzerindeki sanal küsuratları kırmızıya (1), altını maviye (0) boyayacak format atamasını idare eder.
    Z_class = Z_class.reshape(xx.shape)  # Çizilmiş milyon vektörü bir şerit olmaktan kurtarıp, karekteristik 2D alan formülüne yapıştırır.
    Z_continuous = Z_continuous.reshape(xx.shape)  # Aynı formül eklentisini siyah eşik (threshold) çizgisinin nereden geçtiğini göstermek için bir daha düzenler.
    
    plt.figure(figsize=(10, 6))  # 10x6 boyutlarında geniş bir tuval açarak matplotlib altyapısında grafik işlemcisini render ettirmeye girişir.
    plt.contourf(xx, yy, Z_class, alpha=0.3, cmap='bwr')  # Alanların içindeki arkaplanını (Sınıf 1 ve 0 alanları yüzeyi) hafif saydam ve fırça (contourf) ile doldurur.
    
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y, palette='bwr', edgecolor='k', s=60)  # Sınavda çıkan öğrencileri (test dataları) scatter nokta baloncukları olarak sahneye çizer.
    
    plt.contour(xx, yy, Z_continuous, levels=[threshold], colors="black", linewidths=2, linestyles="--")  # Regresyon denkleminin tam 0.5 verdiği noktaları kalın uçlu siyah kesik kalemle böler.
    
    plt.title(title, fontsize=14, fontweight='bold')  # Olayın lineer boyutunu açıklayan tepe isimliğin yazı birimi ayarlarını ve görsel kalınlığını idare ile yükler.
    plt.xlabel('Yaş (StandardScaler ile Ölçeklenmiş)', fontsize=12)  # X okunun neyi kapsadığını açıklamak amaçlı "Yaş" ismini ölçek betimlemesiyle ok hizasına sürer.
    plt.ylabel('Tahmini Maaş (StandardScaler ile Ölçeklenmiş)', fontsize=12)  # Y okunu ölçeklemesini anlatacak metin katarak y ekseni yan boşluğuna hizalı mühürler.
    
    import matplotlib.lines as mlines  # Lejant kutusu için manuel işaretçiler (handle) yaratmaya yarayan kütüphane.
    handle_0 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, markeredgecolor='k', alpha=0.5, label='0: Satın Almadı')  # 0 sınıfı için.
    handle_1 = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=8, markeredgecolor='k', alpha=0.5, label='1: Satın Aldı')  # 1 sınıfı için.
    handle_thresh = mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label=f'{threshold} Eşik Çizgisi')  # Eşik çizgisi için referans.
    plt.legend(handles=[handle_0, handle_1, handle_thresh], title="Karşılaştırma")  # Renklerin doğruluğunu garantiye alan yeni lejant kutusunu yerleştirir.
    
    plt.tight_layout()  # Veriler arası boğulma yaşamamak ve piksellerin margin/padding ayarını otomatik dengelenmesi vasıtasıyla son kontrolünü tasarlar.
    plt.show()  # Yaratılan sanatsal ve istatistiksel çizimin ön ekranda, kullanıcının gözlemleyebilmesine müsait pencereyle pop up yapılmasını komutlar.

def main():  # Lineer sistemi ve analiz zincirini kronolojik şekilde sıraya koyup komuta zincirini sağlayan baş fonksiyon iskeleti yapısı.
    X, y = load_and_preprocess_data()  # Analitik evrenin yapıtaşları olan dosyalardan okuma motorunu devr alır, özellikleri makineye takdim eder.
    
    X_train, X_test, y_train, y_test = train_test_split(  # sklearn model select ile birlikte orantılı zeka ayrıştırma yöntemini uygulayıp setlerini yarar.
        X, y,  # Özellikleri elinde tutan değişkenlerin verilerini test etmek için kesip biçilecek hammadde şeklinde işleme iter.
        test_size=0.25,  # Toplam pastanın (datanın) dörte birinin sınavlarda performansı denetlemek için özel bir klasör köşesine park edilmesini söyler.
        random_state=42,  # Test işlemi her tur yenilense bile modelin veri kopyası hep aynı kalsın (fix olsun) prensibine göre çekirdeği çakar.
        stratify=y  # Alıp almayan kişiler dengesinin (%30'a %70 gibi) train datasında ve test datasında bozulmadan tıpkısı oranıyla paylaştırılmasını yapar.
    )
    
    scaler = StandardScaler()  # Birtakım sayılar onlu diğerleri binli olduğu anlarda tüm sayılar standartlaşsın diye scaler mekanizmasını harekete eyler.
    X_train_scaled = scaler.fit_transform(X_train)  # Eğitim testini inceleyerek ortalamayı (ortalama=0, sapma=1) mantığı ile kurar ve verileri pres yapılmış eze dönüştürür.
    X_test_scaled = scaler.transform(X_test)  # Modelin kafasını testleri de öğreterek bozmamak amaçlı, sadece train bilgisini entegre eden transform mantığını uygulatır.
    
    lin_reg = LinearRegression()  # Lineer (nokta oturtmaya yönelik süregelen) tahmin modülünü çağırarak matematiksel beyin kurgusunu sanal olarak tetikler.
    lin_reg.fit(X_train_scaled, y_train)  # Standardize edilmiş sayılardan öğrenerek X'in Y üzerinde yaptığı denklemin (mx + b) m ve b ağırlıklarını eğitim ile kurgular.
    
    y_pred_continuous = lin_reg.predict(X_test_scaled)  # Kurgulanmış model denkleminin üstünden saklanıp korunan sınav test verilerini akıtarak küsuratlı lineer sayı doğurur.
    
    evaluate_linear_model(y_test, y_pred_continuous, threshold=0.5)  # Eşik noktasını vererek asıl sonuçların eşik sonrasıyla karşılaştırılmamasını denetleyen metrik komut bağlar.
    
    plot_decision_boundary(X_test_scaled, y_test.values, lin_reg, threshold=0.5)  # Her sayısal nokta ve tahmin neticesinde boy göstererek karar tablosunu eşik çizgisi ile ayar.

if __name__ == "__main__":  # Çalıştırılan python kod bloğunun direkt terminal üzerinden bu dosya adıyla execute edilip edilmediğine ait referans denetim bloğudur.
    main()  # Sadece bu dosya komutu girildiğinde denetimi serbest bırakıp sistem süreçlerini ve döngülerini başlatan enjeksiyon ateşlemesidir.
