#ICINDEKILER
''' 1- BAZI NOTLAR / PRATIK KULLANIMLAR '''
# 1-1 Ekran boyutuna gore tum degiskenleri gormek icin,
# 1-2 Verisetindeki sayisal degiskenleri istedigin aralikta inceleyebilmek icin,
# 1-3 Kaggle i kullanarak datayi dogrudan jupiterden indirebilirmek icin,
# 1-4 Plotly komutlari pycharmda calismazsa web'de calistirabilirmek icin,
# 1-5 Scatter plotta noktlar ust uste biniyor ve kumelenmeler net bir sekilde gozukmuyorsa,
# 1-6 Tarih iceren bir verisetinde calisiyorsan,
# 1-7 Verisetinden %x lik kismi cekerek random bir altkume olusturmak icin,
# 1-8 Verisetinden cektigim orneklem verisetini ne kadar iyi temsil ediyor?,
# 1-9 Verisetinin %95 guven araligindaki kismini incelemek icin,
# 1-10 Korelasyon analizi yaparken suna dikkat edilmelidir!,
# 1-11 Calistirdigin kodun ne kadar zamanda calistigini gormek istiyorsan(runtime),
''' 2- OLASILIK'''
# 2-1 Bernoulli dagilimi
# 2-2 Binom dagilimi
# 2-3 Poisson dagilimi
# 2-4 Normal dagilim
''' 3- HIPOTEZ TESTLERI '''
# 3-1 Hipotez testi
# 3-2 Dagilim normal mi?
# 3-3 Tek orneklem T testi
# 3-4 Nonparametrik Tek orneklem T testi
# 3-5 Bagimsiz iki orneklem T testi (AB Testi)
# 3-6 Varyans Homojen mi?
''' 4- DEGISKENLER ARASI ILISKI'''
# 4-1 Iki degiskenin birbiriyle hem dagilimlarini hem de korelasyonlarini gormek istiyorsan
# 4-2 Tum degiskenlerin hedef degiskenle arasindaki korelasyonlar
# 4-3 istatistiki acidan degiskenleri incelemek icin;
''' 5- EKSIK DEGERLER ICIN BAZI YONTEMLER '''
# 5-1 Eksik veri MCAR testiyle degerlendirilebilir
# 5-2 Eksik veriyi kendisinden once veya sonra gelen degerle doldurmak icin
# 5-3 Makine ogrenmesi yontemiyle eksik degerlerin doldurulmasi
''' 6- AYKIRI DEGER COZUM YONTEMLERI '''
# 6-1 Ortalamanin uzerine standart sapmasini ekleyip cikararak aykiri degerleri disarida birakarak
# 6-2 Aykiri degerler icin bir diger yontem de baskilama yontemi
# 6-3 Cok degiskenli aykiri gozlem analizini lof denilen yontemle inceleyeme
''' 7- DEGISKEN DONUSUMLERI '''
# 7-1 Labelencoder
# 7-2 one-hot donusumu
# 7-3 Bir ve digerleri donusumu
# 7-4 Surekli degiskeni kategorik degiskene donusturme
''' 8- DEGISKEN STANDARTLASTIRMA, NORMALLESTIRME '''
# 8-2 Normalizasyon
# 8-3 Eger belli bir aralikta donusturme yapmak istiyorsak
# 8-4 Butun sutunlara Label encoder
''' 9- PCA ANALIZI (HANGI DEGISKENLERI MODELE ALMALIYIM?)'''
# 9-1 PCA

''' ///// SUPERVISED LEARNING \\\\\ '''
''' 10- ML DOGRUSAL MODELLER '''
# 10-1 Basit dogrusal regresyon
# 10-2 Multicollinearity nasil kontrol edilir? VIF metodu;
# 10-3 Degiskenin logaritmasi nasil alinir?
# 10-4 Peki sklearn ile modelde kullanmamiz gereken degiskenleri nasil sececegiz?
# 10-5 Statsmodels ile dogrusal model kurma
# 10-6 coklu dogrusal regresyon
# 10-7 PCR modeli
# 10-8 PLS modeli
# 10-9 Ridge regresyon
# 10-10 Lasso regresyon
# 10-11 ElasticNET regresyon modeli
# 10-12 Train test split
''' 11 ML DOGRUSAL OLMAYAN MODELLER '''
# 11-1 KNN modeli
# 11-2 SVR modeli
# 11-3 MLP(Yapay sinir aglari)
# 11-4 CART Modeli
# 11-5 Bagging regresyon
# 11-6 Random Forest
# 11-7 Degisken onemleri anlamliliklari
# 11-8 GBM modeli
# 11-9 XGBOOST
# 11-10 xgboost'un kendi veri yapisi kullanilarak uygulama
# 11-11 LightGBM
# 11-12 CatBoost
''' 12- ML SINIFLANDIRMA MODELLERI '''
# 12-1 Logistic Regresyon
# 12-2 Siniflandirmada ben modelin ciktisina gore kendim belli bir atama yapmak istiyorsam;
# 12-3 Naive Bayes
# 12-4 KNN modeli
# 12-5 SVC
# 12-6 MLP (YSA)
# 12-7 CART
# 12-8 Random Forests.
# 12-9 GBM
# 12-10 XGBOOST
# 12-11 LightGBM
# 12-12 CatBoost
''' 13 TUM MODELLERIN KARSILASTIRILMASI '''
# 13 TUM MODELLERIN KARSILASTIRILMASI
# Tum modellerin ogrendiklerini birlestirip yeni bir skor elde etmek
''' ///// UNSUPERVISED LEARNING \\\\\ '''
''' 14 UNSUPERVISED LEARNING '''
# 14-1 K-means ile kendi icinde homojen ama birbirinden farkli kumeler olusturma
# 14-2 Hiyerarsik kumeleme
''' 15 DEEP LEARNING '''
# 15-1 ANN


''' 1- BAZI NOTLAR / PRATIK KULLANIMLAR '''
#%%
# 1-1 Ekran boyutuna gore tum degiskenleri gormek istersen yapman gereken ayarlama;
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
#%%
#%%
# 1-2 Guzel bir describe ayarlamasi, verisetindeki sayisal degiskenleri istedigin aralikta inceleyebilmek icin;
num_cols = list(df.select_dtypes(include=[np.number]))
df[num_cols].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
#%%
#%%
# 1-3 Kaggle i kullanarak datayi dogrudan jupiterden indirebilirmek icin;
!pip install kaggle
# kaggle da account sayfasina git. Create New API Token e tikla ve kaggle.json dosyasini indir.
#!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/kaggle/kaggle.json
# bunu burada calistiramadim terminalde calisti
#artik kaggledaki verisetlerine erisebiliriz.
!kaggle competitions list
#%%
#%%
# 1-4 Plotly komutlari pycharmda calismazsa web'de calistirabilirmek icin;
import plotly.io as pio
pio.renderers.default = "browser"
#%%
#%%
# 1-5 Scatter plotta noktlar ust uste biniyor ve kumelenmeler net bir sekilde gozukmuyorsa
# jittering uygulanarak daha net gozlemlenebilir;
def jitter(data, stdev):
    N = len(data)
    return data + np.random.randn(N) * stdev
# sigma = standart sapma
plt.scatter(jitter(X, sigma), jitter(Y, sigma), c=y)
#%%
#%%
# 1-6 Tarih iceren bir verisetinde calisiyorsan;
print 'Train min/max date: %s / %s' % (train.Date.min().date(), train.Date.max().date())
print 'Number of days in train: %d' % ((train.Date.max() - train.Date.min()).days + 1)
print 'Number of days in test:  %d' % (( test.Date.max() -  test.Date.min()).days + 1)
#%%
#%%
# 1-7 Verisetinden %x lik kismi cekerek random bir altkume olusturmak icin
s = train.sample(frac = 0.1)
#%%
#%%
# 1-8 Verisetinden cektigim orneklem verisetini ne kadar iyi temsil ediyor?
#random choice
import numpy as np
x = np.random.randint(0,50,1000)
orneklem = np.random.choice(a = x, size=100)
orneklem
x.mean()
orneklem.mean()
#%%
#%%
# 1-9 Verisetinin %95 guven araligindaki kismini incelemek icin;
fiyatlar = np.random.randint(10,100,1000)
import statsmodels.stats.api as sms
sms.DescrStatsW(fiyatlar).tconfint_mean()
#bu bize %95 guven araligi veriyor
#%%
#%%
# 1-10 Korelasyon analizi yaparken suna dikkat edilmelidir!. Eger iki degiskenin korelasyonuna bakiyorsan ve her iki degiskenin dagilimi da normalse o zaman pearson korelasyon katsayisini, aksi halde spierman korelasyon katsayisini kullanmalisin.(python daki corr fonksiyonu dagilimin normal oldugunu kabul ederek calisir, eger dagilim normal degilse ise yaramaz)
#eger normallik saglanmiyorsa iki degisken icin, aralarindaki iliskiyi nasil kontrol edecegiz?
stats.spearmanr(df['tip'],df['totalbill']) # eger p-value < 0.05 olsaydi degiskenler arasinda korelasyon yoktur diyecektik. bu arada soldaki deger korelasyon katsayisidir sonucta cikan
#%%
#%%
# 1-11 Calistirdigin kodun ne kadar zamanda calistigini gormek istiyorsan(runtime);
%%time
x = np.random.randint(1,20,1)
#%%
#%%
# 2- OLASILIK
#%%
# 2-1 Bernoulli dagilimi = basarili basarisiz gibi iki sonuclu olaylar ile ilgilenildiginde bu kesikli dagilim kullanilir
from scipy.stats import bernoulli
p = 0.6 # tura gelme olasiliginin 0.6 ciktigini bildigimizi varsayalim
rv = bernoulli(p)
rv.pmf(k=1) # k=1 ilgilendigimiz olay yani tura gelme olsailigi, k=0 icin de yazi gelme olasiligidir
#%%
#%%
# 2-2 Binom dagilimi bagimsiz n deneme sonucu k basarili olma olasiligi ile ilgilenildiginde kullanilan dagilimdir
from scipy.stats import binom
p = 0.01 # reklama tiklama olasiligi
n = 100 # deneme sayisi
rv = binom(n,p)
print(rv.pmf(1)) # reklami 100 defa goren 1 kisinin reklama tiklama olasiligi
print(rv.pmf(5)) # reklami 100 defa goren 5 kisinin reklama tiklama olasiligi
#%%
# 2-3 Poisson dagilimi belirli bir zaman araliginda belirli bir alanda nadiren rastlanan olaylarin olasiligini hesaplamak icin kullanilir
#bir olayin nadir olay olabilmesi icin n*p<5 olmalidir
from scipy.stats import poisson
lambda_ = 0.1 # lambda degeri neyse
rv = poisson(mu=lambda_)
print(rv.pmf(k=0)) # hic hata olmamasi olasiligi
#%%
# 2-4 Normal dagilima sahip oldugu bilinen bir grafigin belli bir yerinde olma olasiligi
from scipy.stats import norm
1-norm.cdf(90,80,5) # 90=hesaplanmak istenen deger, 80 = egrinin ortalamasi, 5=standart sapmasi
# normal dagilim egrisini dusun parabolik, onun altinda kalan alana gore bu islemler yapiliyor
# ornegin 85 ile 90 arasindaki kismin gerceklesme olasiligi;
norm.cdf(90,80,5) - norm.cdf(85,80,5)
#%%
#%%
# 4- HIPOTEZ TESTLERI (Tek Orneklem T Testi, Iki Orneklem T Testi)
#%%
# 3-1 Hipotez testi
# Acaba web sitemizde gecirilen sure gercekten 170 saniye mi?
#H0: mu=170
#H1:mu!=170
olcumler  =np.random.randint(160,180,20)
import scipy.stats as stats
# once varsayimlari kontrol edelim
# 3-2
# 1-normallik varsayimi (Dagilim normal mi?)
pd.DataFrame(olcumler).plot.hist() # histogram araciligiyla dagilimin normal olup olmadigina bakilabilir
import pylab
stats.probplot(olcumler, dist='norm', plot=pylab)
pylab.show() #bu da qqplot yontemi, normalligi gozlemleyebilmek icin 2. yontem. Bu grafikte noktalar kirmizi cizgiye yakin olmali normallik varsayimi icin
from scipy.stats import shapiro # bu da normallik varsayimini denetler
shapiro(olcumler) # sol taraftaki test istatistigi, sag taraftaki de p-value. P-value alfadan kucukse orneklem ile populasyon arasinda anlamli bir farklilik yoktur diyebiliriz
# simdi hipoteze gecebiliriz
# 3-3 Tek orneklem T testi
stats.ttest_1samp(olcumler,popmean=170) # p-value alfa degerinden kucukse H0 hipotezi reddedilir
#%%
#yukaridaki tek orneklem hipotez testinde eger varsayimlar saglanmazsa nonparametrik tek orneklem testi yapilabilir
# 3-4 Nonparametrik Tek orneklem T testi
from statsmodels.stats.descriptivestats import sign_test
sign_test(olcumler,170) # sag taraftaki p-value
#%%
#Iki grup ortalamasi arasinda karsilastirma yapilmak istenildiginde bagimsiz iki orneklem T testi (AB Testi) kullanilir
# 3-5 Bagimsiz iki orneklem T testi (AB Testi)
# bu testi uygualaybilmemiz icin iki varsayimin dogrulanmasi gerekiyor
# 1-normallik varsayimi
#yine shapiro ile yapilabilir
# 2- varyans homojenligi varsayimi
# 3-6 Varyans Homojen mi?
stats.levene(df.A, df.B) # eger p-value<0.05 olsaydi varyans homojen degildir diyecektik
#hipotez testine gecelim simdi
stats.ttest_ind(df['A'],df['B'], equal_var=True) # p-value < 0.05 oldugunda iki grup arasinda anlamli bir farklilik vardir diyebiliriz
#%%
#%%
# 4- DEGISKENLER ARASI ILISKI
#%%
# 4-1 Iki degiskenin birbiriyle hem dagilimlarini hem de korelasyonlarini gormek istiyorsan
sns.jointplot(x = "TV", y = "sales", data = df, kind = "reg")
#%%
#%%
# 4-2 Tum degiskenlerin hedef degiskenle arasindaki korelasyonlar
num_columns_correlations = numeric_columns.corr()
print(num_columns_correlations['SalePrice'].sort_values(ascending = False),'/n')
#%%
#%%
# 4-3 istatistiki acidan degiskenleri incelemek icin;
!pip install researchpy
import researchpy as rp
#sayisal degiskenler icin
rp.summary_cont(df[['totalbill','tip','size']])
#N = gozlem sayisi
#SD = standart sapma
#SE = standart hata
#%95E = guven araliklari
#kategorik degiskenler icin
np.summary_cat(df[['day','sex']])
#kovaryans icin
df[['tip','totalbill']].cov()
#korelasyon icin
df[['tip','totalbill']].corr()
#%%
#%%
# 5- EKSIK DEGERLER ICIN BAZI YONTEMLER
#%%
# 5-1 Eksik veri MCAR testiyle degerlendirilebilir
df[df.isnull().all(axis=1)] # verisetinde tum degerleri eksik olan satirlari getiriyor
df.dropna(how='all') # yalnizca butun degerleri eksik olan satirlari siler
df.apply(lambda x: x.fillna(x.mean()),axis=0) # verisetindeki tum eksik degerleri ortalamalariyla dolduran kod
!pip install missingno
import missingno as ms
ms.bar(df) # degiskenler icerisindeki eksiklikleri gorsellestiriyor
ms.matrix(df) # bu da yine eksiklikleri iliskisel olarak gosteriyor
ms.heatmap(df) # eksik degerleri olan degiskenlerin birbirleriyle eksiklik korelasyonlarini verir
#Kategorik degisken kiriliminda deger atama
df['maas'].fillna(df.groupby('departmant')['maas'].transform('mean')) # maas'taki eksik degerleri departmanlarin maas ortalamalarina gore doldurdu
# 5-2 Eksik veriyi kendisinden once veya sonra gelen degerle doldurmak icin
# kendisinden sonra gelenle doldrumak icin;
df['departmant'].fillna(method='bfill') # once gelenle doldrumak icin bfill yerine ffill kullanilir.
# 5-3 Makine ogrenmesi yontemiyle eksik degerlerin doldurulmasi
!pip install ycimpute
# once KNN ile olan yontemi gosterelim
from ycimpute.imputer import knnimput
var_names = list(df)
n_df = np.array(df)
dff = knnimput.KNN(k=4).complete(n_df) # boylece eksik degerleri doldurmus olduk, fakat numpy array suan
dff = pd.DataFrame(dff, columns=var_names)
#simdi random forest ile ayni eksikleri tamamlayalim
from ycimpute.imputer import iterforest
n_df = np.array(df)
var_names = list(df)
dff = iterforest.IterImput().complete(n_df)
dff = pd.DataFrame(dff, columns=var_names)
#simdi ayni islemleri EM algoritmasiyla yapalim
from ycimpute.imputer import EM
n_df = np.array(df)
var_names = list(df)
dff = EM().complete(n_df)
dff = pd.DataFrame(dff, columns=var_names)
#%%

# 6- AYKIRI DEGER COZUM YONTEMLERI
#%%
#aykiri deger cozumleri
#Aykiri degerle nasil bas edilir? Bir yontemi su, ornegin;
# 6-1 Ortalamanin uzerine standart sapmasini ekleyip cikararak aykiri degerleri disarida birakarak veya 2 tane standart sapma ekleyip cikarip disarida kalan degerleri silerek.
#Ya da boxplot kullanarak,
#IQR = Q3-Q1
#ALT ESIK DEGER = Q1-1.5*IQR
#UST ESIK DEGER = Q3 + 1.5*IQR
sns.boxplot(x=df['tip'])
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR= Q3-Q1
alt_sinir = Q1-1.5*IQR
ust_sinir = Q3 + 1.5*IQR
(df < alt_sinir | df > ust_sinir) # deyip kontrol et kac deger var
aykiri = (df < alt_sinir | df > ust_sinir)
df[aykiri].index # bu sana aykiri degerlerin indexlerini verir.
# eger index kullanmadan df olusturmak istersen;
temiz_df = df[~(df < alt_sinir | df > ust_sinir).any(axis=1)] # boylece tum aykiri degerleri temizlemis olduk

# 6-2 Aykiri degerler icin bir diger yontem de baskilama yontemi. Ornegin alt sinirin altinda kalan degerleri alt sinirla degistirme, ust sinirin da ustunde kalan degerleri ust sinirla degistirme gibi.
df[df<alt_sinir] = alt_sinir
df[df> ust_sinir] = ust_sinir
# 6-3 Cok degiskenli aykiri gozlem analizini lof denilen yontemle inceleyeme. En cok bu kullanilmali zaten gercek projeler icin
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=20,contamination=0.1)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:20]
esik_deger = np.sort(df_scores)[13] # degerin bir anda cok artip cok azaldigi bas ve sonlardaki iki degeri esik deger olarak kabul edebilirsin,bunun icin skorlari incelemelisin
df[df_scores< esik_deger] # bu sekilde de yeni df i olusturabilirsin
#baskilama
aykirilar = df[aykiri_tf] # olsun
baski_deger = df[df_scores == esik_deger]
res = aykirilar.to_records(index=False) # indexlerden kurtuldum
res[:] = baski_deger.to_records(index=False)
df[aykirilar] = pd.DataFrame(res,index=df[aykirilar].index) # aykiri degerleri esik degerleri ile doldurduk
#%%

# 7- DEGISKEN DONUSUMLERI
#%%
# 7-1 Labelencoder
from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()
df['yeni_sex'] = lbe.fit_transform(df['sex'])
# 7-2 one-hot donusumu
new_df_ = pd.get_dummies(new_df,columns=['HouseStyle'],prefix=['HouseStyle'])
train = pd.get_dummies(train, columns=v, drop_first=True)
# 7-3 Bir ve digerleri donusumu
df['yeni_day'] = np.where(df['day'].str.contains('sun'),1,0) # yani gunlerden sun olanlara 1 digerlerine de 0 ver diyor bu kod
# 7-4 Surekli degiskeni kategorik degiskene donusturme
from sklearn.preprocessing import KBinsDiscretizer
dff = df.select_dtypes(include=['float64','int64'])
dff_ = preprocessing.KBinsDiscretizer(n_bins=[3,2,2], encode='ordinal',strategy='quantile').fit(dff) # dff surekli degisken olmali, quantile ile 4 ceyrek olacak sekilde surekli degiskenleri boler
new = dff_.transform(dff_)
#%%
#%%
# 8- DEGISKEN STANDARTLASTIRMA, NORMALLESTIRME
# 8-1 Degisken standardizasyonu
from sklearn import preprocessing
preprocessing.scale(df)  # verisetindeki tum degiskenleri standartlastirmis oldu
# 8-2 Normalizasyon
from sklearn import preprocessing
preprocessing.normalize(df)
# 8-3 Eger belli bir aralikta donusturme yapmak istiyorsak
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler(feature_range=(10,20)) # ornegin 10 ile 20 arasinda olsun istedik butun degerlerin
scaler.fit_transform(df)
# 8-4 Butun sutunlara Label encoder
for c in train.columns[train.dtypes == 'object']:
    X[c] = X[c].factorize()[0]
#%%

# 9- PCA ANALIZI
# 9-1 PCA . Cok degiskenli verinin ana ozelliklerini daha az sayida degiskenle temsil etmeye yarar
# PCA uygulayabilmek icin once degiskenleri standartlastirmaliyiz
from sklearn.preprocessing import StandardScaler
df = StandardScaler().fit_transform(df)
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)    # 3 degiskenle tum veriyi temsil etmis oluyoruz
pca_fit = pca.fit_transform(df)
bilesen_df = pd.DataFrame(data = pca_fit,
                          columns = ["birinci_bilesen","ikinci_bilesen","ucuncu_bilesen"])
bilesen_df.head()
pca.explained_variance_ratio_     # bu 3 degiskenin verisetini aciklama yuzdesi burada cikan rakamlarin toplamidir. Her biri bir degiskenin aciklama yuzdesidir
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
#%%
#%%
# 10- ML DOGRUSAL MODELLER
#%%
#dogrusal ML modelleri
# 10-1 Basit dogrusal regresyon
import statsmodels.api as sm
X = df[["TV"]]
X = sm.add_constant(X) # sabit 1 olan bir sutun eklemek gerekiyormus hoca da detay vermedi
y = df["sales"]
lm = sm.OLS(y,X)
model = lm.fit()
model.summary() # duzeltilmis R kare degeri bagimsiz degiskenini bagimli degiskeni aciklama basarisi, f istatistigi model anlamli mi degil mi onun bilgisini verir, p-value 0.05 ten kucukse degiskenler anlamli demektir, beta 0 = consts yazan deger, beta 1 de bi altindaki deger, P>t diye yazan P p-value aslinda.
# coef'in yanindaki std err katsayinin dogrulugunu ifade eder.
# P>t yazan yer pvalue. Eger 0.05 ten kucukse o degisken bagimli degiskeni tahmin etmekte cok onemli demektir.
# R kare degeri ise senin modelin bu problemi aciklamada ne kadar basarili? Eger 1 ise %100 basarili demektir.
# Adj R kare ise diyelim ki onceki kurdugun modele birden fazla degisken ekledin. Bu durumda belki R kare degerin artmis olabilir ama eger Adj R kare degerin de artmadiysa bu son ekledigin degiskenlerin modele faydasinin olmadigini ifade eder.
# Iste bu yuzden adj R kare degerine bakarak yorumlama yapilmalidir.
# F-statistic'in altindaki Prob(F-statistic) pvalue degerini ifade eder. Eger bu deger 0.05 ten kucukse modelimiz anlamli demektir.
# F-statistik ne kadar dusukse modelin anlamliligi da o kadar azdir.
# Regresyon modelinin basarili olabilmesi lineer olmasina, varyansin homojenligine, modeli aldigin degiskenlerin bagimli degiskenle iliskili olduguna ve ayni bilgiyi tasiyan benzer degiskenlerin fazladan modele alinmamasina baglidir
# 10-2 Multicollinearity nasil kontrol edilir? VIF metodu;
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
# eger VIF = 1 ise no multicollinearity. 1 ile 5 arasindaysa yine iyi, ama 5 ten fazlaysa multicollinearity var demektir, degiskenleri incelemeli ayni bilgiyi tasiyan degiskenler temizlenmeli

# 10-3 Degiskenin logaritmasi nasil alinir?
df['log_x'] = np.log(df['x'])
# log un tersini almak istersen;
np.exp(ifade)
# eger df teki tum degerlerin virgulden sonra iki basamaginin gorunmesini istiyorsan;
pd.set_option('display.float_format',lambda x: '%.2f' % x)

# Eger dogrusal model uygulayamiyacak kadar daginik bir verisetiyse x ve y degiskenlerinin logaritmalarini alip tekrar scatter plotta ciz. Dogrusala donmus olur
# Eger ayni modeli sklearn ile kurmus olsaydik parametrelere su sekilde erisebilirdik;
# model.skore(x,y) == bu bize R kare skorunu verir
# 10-4 Peki sklearn ile modelde kullanmamiz gereken degiskenleri nasil sececegiz?
#future_selection metodu ile;
from sklearn.feature_selection import f_regression
f_regression(x,y)
p_values = f_regression(x,y)[1]
# e ile yazilan sayilari su sekilde duzenleyebilirsin
p_values.round(3) # virgulden sonra 3 basamaga yuvarlama
# 10-5 Statsmodels ile dogrusal model kurma
#yukaridaki ayni modeli su sekilde de kurabilirdik
import statsmodels.formula.api as smf
lm = smf.ols("sales ~ TV", df)
model = lm.fit()
model.summary()
model.params
model.conf_int()
model.f_pvalue
model.rsquared_adj
model.mse_model # MSE hatasi
model.fittedvalues[0:5]
#kurdugumuz modeli gorsellestirmek istersek
g = sns.regplot(df["TV"], df["sales"], ci=None, scatter_kws={'color':'r', 's':9})
g.set_title("Model Denklemi: Sales = 7.03 + TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
import matplotlib.pyplot as plt
plt.xlim(-10,310)
plt.ylim(bottom=0);
#%%
# 10-6 coklu dogrusal regresyon
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
X = df.drop("sales", axis = 1)
y = df["sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state= 42)
import statsmodels.api as sm
lm = sm.OLS(y_train, X_train)
model = lm.fit()
model.summary()
#%%
# 10-7 PCR modeli
#ornegin 100 tane degisken var verisetinde ve ben 10 tanesiyle cogunlugunu temsil edebiliyorsam bu durumda bilesen boyut indirgemesi yapilir bu yaklasimla
#bu modelin faydasi ne peki? coklu dogrusal model problemini ortadan kaldiriyor. yani 10 tane degisken ile ifade ediyor ve bu 10 degiskenin birbiriyle olan korelasyonlari da dusuk oluyor
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pca = PCA()
X_reduced_train = pca.fit_transform(scale(X_train))
np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:10] # bilesenlerin verisetini aciklama oranlari % olarak
# amac zaten verisetini daha az degiskene indirgemekti
lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:10], y_train) # butun bilesenleri degil de sadece 10 bilesenle modeli kurmak istersek
y_pred = pcr_model.predict(X_reduced_test[:,0:10])
print(np.sqrt(mean_squared_error(y_test, y_pred)))
#simdi ayni modeli cv ile yapilandiralim, kac tane degiskenle modele devam etmemiz gerektigine gorsellestirerek karar verelim
from sklearn import model_selection
cv_10 = model_selection.KFold(n_splits = 10,
                             shuffle = True,
                             random_state = 1)
lm = LinearRegression()
RMSE = []
for i in np.arange(1, X_reduced_train.shape[1] + 1):
    score = np.sqrt(-1 * model_selection.cross_val_score(lm,
                                                         X_reduced_train[:, :i],
                                                         y_train.ravel(),
                                                         cv=cv_10,
                                                         scoring='neg_mean_squared_error').mean())
    RMSE.append(score)
plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maaş Tahmin Modeli İçin PCR Model Tuning');
# bu grafikten optimum bilesen sayisinin ne oldugunu ogrendik, simdi final modelini kuralim
pcr_model = lm.fit(X_reduced_train[:,0:6], y_train) # eger optimum bilesen sayisi 6 ise.
y_pred = pcr_model.predict(X_reduced_test[:,0:6])
print(np.sqrt(mean_squared_error(y_test, y_pred)))
#%%
# 10-8 PLS modeli de PCR ile benzer amacli. Farklari su PCR bagimli degiskeni katmadan degiskenleri incelerken PLS bagimli degiskenin kovaryansini da maksimum sekilde ozetlemeye calisir
from sklearn.cross_decomposition import PLSRegression, PLSSVD
pls_model = PLSRegression().fit(X_train, y_train)
#CV
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
#Hata hesaplamak için döngü
RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

#Sonuçların Görselleştirilmesi
plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary'); # bu gorselde en iyi parametrenin 2 oldugunu gorduk

pls_model = PLSRegression(n_components = 2).fit(X_train, y_train)
y_pred = pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 10-9 Ridge regresyon. Amac hatalara ceza uygulayarak bu hatalari minimize etmek
from sklearn.linear_model import Ridge
lambdalar = 10 ** np.linspace(10, -2, 100) * 0.5
ridge_model = Ridge()
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas = lambdalar,
                   scoring = "neg_mean_squared_error",
                   normalize = True)
ridge_cv.fit(X_train, y_train)
ridge_tuned = Ridge(alpha = ridge_cv.alpha_,
                   normalize = True).fit(X_train,y_train)
np.sqrt(mean_squared_error(y_test, ridge_tuned.predict(X_test)))
#%%
# 10-10 Lasso regresyon. Amac hatalara ceza uygulayarak bu hatalari minimize etmek. Ridge den farki katsayilari 0 a iyice yaklastirmasi hatta bazilarini 0 yaparak degisken secimi de yapmasi
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas = None,
                         cv = 10,
                         max_iter = 10000,
                         normalize = True)
lasso_cv_model.fit(X_train,y_train)
lasso_tuned = Lasso(alpha = lasso_cv_model.alpha_)
lasso_tuned.fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 10-11 ElasticNET regresyon modeli. Dogrusal regresyonun en gelistirilmis halidir. Ridge gibi cezalandirma yapar, Lasso gibi de degisken secimi yapar
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
enet_cv_model = ElasticNetCV(cv = 10, random_state = 0).fit(X_train, y_train)
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
#%%
# 10-12 Train test split
from sklearn.model_selection import train_test_split
y = df['Exited'].copy()
x = df.drop('Exited', axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
#%%
#%%
mms = MinMaxScaler()
x_train_normed = mms.fit_transform(x_train)
x_test_normed= mms.fit_transform(x_test)
#%%
#%%
predictions = []
for tree in rf.estimators_:
    predictions.append(tree.predict_proba(X_val)[None, :])

predictions = np.vstack(predictions)

cum_mean = np.cumsum(predictions, axis=0)/np.arange(1, predictions.shape[0] + 1)[:, None, None]

scores = []
for pred in cum_mean:
    scores.append(accuracy_score(y_val, np.argmax(pred, axis=1)))

plt.figure(figsize=(10, 6))
plt.plot(scores, linewidth=3)
plt.xlabel('num_trees')
plt.ylabel('accuracy');
#%%
#%%
# 11 ML DOGRUSAL OLMAYAN MODELLER
# 11-1 KNN modeli. Tahminler gozlem benzerligine gore yapilir.
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
knn = KNeighborsRegressor()
knn_params = {'n_neighbors': np.arange(1,30,1)}
knn_cv_model = GridSearchCV(knn, knn_params, cv = 10)
knn_cv_model.fit(X_train, y_train)
knn_cv_model.best_params_["n_neighbors"]
RMSE = []
RMSE_CV = []
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train,y_pred))
    rmse_cv = np.sqrt(-1*cross_val_score(knn_model, X_train, y_train, cv=10,
                                         scoring = "neg_mean_squared_error").mean())
    RMSE.append(rmse)
    RMSE_CV.append(rmse_cv)
    print("k =" , k , "için RMSE değeri: ", rmse, "RMSE_CV değeri: ", rmse_cv )
knn_tuned = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, knn_tuned.predict(X_test)))
#%%
# 11-2 SVR modeli bir marjin araligina maksimum noktayi en kucuk hata ile alabilecek sekilde dogruyu ya da egriyi belirlemektir
from sklearn.svm import SVR
svr_rbf = SVR('rbf')
svr_params = {"C": [0.01, 0.1,0.4,5,10,20,30,40,50]}
svr_cv_model = GridSearchCV(svr_rbf,svr_params, cv = 10)
svr_cv_model.fit(X_train, y_train)
svr_cv_model.best_params_
svr_tuned = SVR("rbf", C = pd.Series(svr_cv_model.best_params_)[0]).fit(X_train,
                                                                        y_train)
y_pred = svr_tuned.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 11-3 MLP(Yapay sinir aglari). Inputlara belli agirliklar veriliyor ve ciktilar uretiliyor, eger hata payi yuksekse inputlarin agirliklari degistirilerek tekrar deneniyor
#MLP icin standartlastirma ekstra onemlidir
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.neural_network import MLPRegressor
mlp_model = MLPRegressor()
mlp_params = {'alpha': [0.1, 0.01,0.02,0.005],
             'hidden_layer_sizes': [(20,20),(100,50,150),(300,200,150)],
             'activation': ['relu','logistic']}
mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv = 10)
mlp_cv_model.fit(X_train_scaled, y_train)
mlp_cv_model.best_params_
mlp_tuned = MLPRegressor(alpha = 0.02, hidden_layer_sizes = (100,50,150))
mlp_tuned.fit(X_train_scaled, y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 11-4 CART Modeli. Heterojen veri setini alip amaca gore homojen olarak gruplar.
from sklearn.tree import DecisionTreeRegressor
cart_model = DecisionTreeRegressor()
cart_params = {"min_samples_split": range(2,100),
               "max_leaf_nodes": range(2,10)}
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10)
cart_cv_model.fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeRegressor(max_leaf_nodes = 9, min_samples_split = 37)
cart_tuned.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 11-5 Bagging regresyon. Birden fazla rastgele karar agaci olusturuluyor ve her birinin tahmini kontrol ediliyor.
from sklearn.ensemble import BaggingRegressor
bag_model = BaggingRegressor(bootstrap_features = True)
bag_params = {"n_estimators": range(2,20)}
bag_cv_model = GridSearchCV(bag_model, bag_params, cv = 10)
bag_cv_model.fit(X_train, y_train)
bag_cv_model.best_params_
bag_tuned = BaggingRegressor( n_estimators = 14, random_state = 45)
bag_tuned.fit(X_train, y_train)
y_pred = bag_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 11-6 Random Forest. Olusturulan rastgele agaclarin tahmin sonuclarini inceler ve ortak bir sonuc olusturur. Iyi sonuc veren agaclara agirlik verir bu model
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state = 42)
rf_params = {'max_depth': list(range(1,10)),
            'max_features': [3,5,10,15],
            'n_estimators' : [100, 200, 500, 1000, 2000]}
rf_cv_model = GridSearchCV(rf_model,
                           rf_params,
                           cv = 10,
                            n_jobs = -1)
rf_cv_model.fit(X_train, y_train)
rf_cv_model.best_params_
rf_tuned = RandomForestRegressor(max_depth  = 8,
                                 max_features = 3,
                                 n_estimators =200)
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 11-7 Degisken onemleri anlamliliklari
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Değişken Önem Düzeyleri")
#%%
# 11-8 GBM modeli. Zayif ogrenicileri bir araya getirip guclu bir ogrenici olusturuluyor
from sklearn.ensemble import GradientBoostingRegressor
gbm_model = GradientBoostingRegressor()
gbm_params = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2],
    'max_depth': [3, 5, 8,50,100],
    'n_estimators': [200, 500, 1000, 2000],
    'subsample': [1,0.5,0.75],
}
gbm_cv_model = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv_model.fit(X_train, y_train)
gbm_cv_model.best_params_
gbm_tuned = GradientBoostingRegressor(learning_rate = 0.1,
                                      max_depth = 5,
                                      n_estimators = 200,
                                      subsample = 0.5)

gbm_tuned = gbm_tuned.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
Importance = pd.DataFrame({"Importance": gbm_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Değişken Önem Düzeyleri")
#%%
# 11-9 XGBOOST. GBM in gelistirilmis halidir.
import xgboost as xgb
from xgboost import XGBRegressor
# 11-10 xgboost'un kendi veri yapisi kullanilarak uygulama
# bu kismi uygulayarak devam et. Bu xgboost un kendi veri yapisi. Eger bunu kullanirsan daha basarili olursun.
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }
model = xgb.train(
    params, # params i burada yukaridaki parametrelerden birer tanesini ver, hepsini verirsen uzun surer
    dtrain,
    num_boost_round=999,
    evals=[(dtest, "Test")],
    early_stopping_rounds=10
)
cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    seed=42,
    nfold=5,
    metrics={'mae'},
    early_stopping_rounds=10
)
cv_results
# daha detayli bilgi icin bu linke bak
# https://blog.cambridgespark.com/hyperparameter-tuning-in-xgboost-4ff9100a3b2f

#normal uygulama
#XGBoost hyper-parameter tuning
def hyperParameterTuning(X_train, y_train):
    param_tuning = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'n_estimators' : [100, 200, 500],
        'objective': ['reg:squarederror']
    }

    xgb_model = XGBRegressor()

    gsearch = GridSearchCV(estimator = xgb_model,
                           param_grid = param_tuning,
                           #scoring = 'neg_mean_absolute_error', #MAE
                           #scoring = 'neg_mean_squared_error',  #MSE
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1)

    gsearch.fit(X_train,y_train)

    return gsearch.best_params_
hyperParameterTuning(X_train, y_train)
xgb_model = XGBRegressor(
        objective = 'reg:squarederror',
        colsample_bytree = 0.5,
        learning_rate = 0.05,
        max_depth = 6,
        min_child_weight = 1,
        n_estimators = 1000,
        subsample = 0.7)

%time xgb_model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_val, y_val)], verbose=False)

y_pred_xgb = xgb_model.predict(X_val)

mae_xgb = mean_absolute_error(y_val, y_pred_xgb)

print("MAE: ", mae_xgb)
#%%
# 11-11 LightGBM modeli. Daha hizli sonuc donduruyor xgboost a gore. BFS yerine DFS(depth first search)
from lightgbm import LGBMRegressor
lgbm = LGBMRegressor()

lgbm_grid = {
    'colsample_bytree': [0.4, 0.5,0.6,0.9,1],
    'learning_rate': [0.01, 0.1, 0.5,1],
    'n_estimators': [20, 40, 100, 200, 500,1000],
    'max_depth': [1,2,3,4,5,6,7,8] }

lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv=10, n_jobs = -1, verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate = 0.1,
                           max_depth = 7,
                           n_estimators = 40,
                          colsample_bytree = 0.6)

lgbm_tuned = lgbm_tuned.fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
# 11-12 CatBoost. Kategorik degiskenlerle mucadele edebilen bir model.
from catboost import CatBoostRegressor
catb = CatBoostRegressor()
catb_grid = {
    'iterations': [200,500,1000,2000],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'depth': [3,4,5,6,7,8] }
catb = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb, catb_grid, cv=5, n_jobs = -1, verbose = 2)
catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostRegressor(iterations = 200,
                               learning_rate = 0.01,
                               depth = 8)

catb_tuned = catb_tuned.fit(X_train,y_train)
y_pred = catb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#%%
#%%
# 12- ML SINIFLANDIRMA MODELLERI
#%%
#SINIFLANDIRMA MODELLERI
# 12-1 Logistic Regresyon. siniflandirma problemi icin bagimli ve bagimsiz degisken arasindaki iliskiyi aciklayan dogrusal model
from sklearn.linear_model import LogisticRegression
loj = sm.Logit(y, X)
loj_model= loj.fit()
loj_model.summary()
loj = LogisticRegression(solver = "liblinear")
loj_model = loj.fit(X_train,y_train)
accuracy_score(y_test, loj_model.predict(X_test))
cross_val_score(loj_model, X_test, y_test, cv = 10).mean()
logit_roc_auc = roc_auc_score(y, loj_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()
# bu siniflandirma problemlerinde kontrol et ciktiyi, neye 0 neye 1 demis onemli!
# 12-2 Siniflandirmada ben modelin ciktisina gore kendim belli bir atama yapmak istiyorsam;
y_probs = loj_model.predict_proba(X)
y_pred = [1 if i > 0.5 else 0 for i in y_probs]
#%%
# 12-3 Naive Bayes. Olasilik temelli siniflandirma modelidir.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
#%%
# 12-4 KNN modeli. Tahminler gozlem benzerligine gore yapilir
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)
knn_cv.best_params_
knn = KNeighborsClassifier(11)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 12-5 SVC . Amac iki sinifin ayriminin oldukca iyi olmasi, iki sinifi da kesin sekilde birbirinden ayirmaya calismak
from sklearn.svm import SVC
svc_params = {"C": np.arange(1,10)}

svc = SVC(kernel = "linear") # lineer olmasin istiyorsan rbf yazarak da yapabilirsin

svc_cv_model = GridSearchCV(svc,svc_params,
                            cv = 10,
                            n_jobs = -1,
                            verbose = 2 )

svc_cv_model.fit(X_train, y_train)
svc_cv_model.best_params_
svc_tuned = SVC(kernel = "linear", C = 5).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 12-6 MLP (YSA) .Insan beyninin sinir aglarini taklit ederek ogrenme
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier()
mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100),
                                     (3,5),
                                     (5, 3)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}
mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params,
                         cv = 10,
                         n_jobs = -1,
                         verbose = 2)

mlpc_cv_model.fit(X_train_scaled, y_train)
mlpc_cv_model.best_params_
mlpc_tuned = MLPClassifier(activation = "logistic",
                           alpha = 0.1,
                           hidden_layer_sizes = (100, 100, 100),
                          solver = "adam")
mlpc_tuned.fit(X_train_scaled, y_train)
y_pred = mlpc_tuned.predict(X_test_scaled)
accuracy_score(y_test, y_pred)
#%%
# 12-7 CART . amac verisetindeki karmasik yapilari basit karar yapilarina donusturmek
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }
cart = tree.DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)
cart_cv_model.best_params_
cart = tree.DecisionTreeClassifier(max_depth = 5, min_samples_split = 19)
cart_tuned = cart.fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 12-8 Random Forests. Birden fazla karar agacinin tahminlerini degerlendirip iyi olana agirlik vererek tahmin
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,5,8],
            "n_estimators": [10,500,1000],
            "min_samples_split": [2,5,10]}

rf_cv_model = GridSearchCV(rf_model,
                           rf_params,
                           cv = 10,
                           n_jobs = -1,
                           verbose = 2)
rf_cv_model.fit(X_train, y_train)
rf_cv_model.best_params_
# gorsellestirmek istersen;
plt.plot(rf.feature_importances_)
plt.xticks(np.arange(X.shape[1]), X.columns.tolist(), rotation=90);
rf_tuned = RandomForestClassifier(max_depth = 10,
                                  max_features = 8,
                                  min_samples_split = 10,
                                  n_estimators = 1000)

rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind ="barh", color = "r")

plt.xlabel("Değişken Önem Düzeyleri")
#%%
# 12-9 GBM . Zayif ogrenicileri biraraya getirip guclu bir ogrenici meydana getirme
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier()
gbm_params = {"learning_rate" : [0.001, 0.01, 0.1, 0.05],
             "n_estimators": [100,500,100],
             "max_depth": [3,5,10],
             "min_samples_split": [2,5,10]}
gbm = GradientBoostingClassifier()

gbm_cv = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 2)
gbm_cv.fit(X_train, y_train)
gbm_cv.best_params_
gbm = GradientBoostingClassifier(learning_rate = 0.01,
                                 max_depth = 3,
                                min_samples_split = 5,
                                n_estimators = 500)
gbm_tuned =  gbm.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 12-10 XGBOOST . GBM in hizlandirilmis ve tahmini kuvvetlendirilmis hali
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_samples_split": [2,5,10]}
xgb = XGBClassifier()

xgb_cv_model = GridSearchCV(xgb, xgb_params, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv_model.fit(X_train, y_train)
xgb_cv_model.best_params_
xgb = XGBClassifier(learning_rate = 0.01,
                    max_depth = 6,
                    min_samples_split = 2,
                    n_estimators = 100,
                    subsample = 0.8)
xgb_tuned =  xgb.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 12-11 LightGBM . Derinlemesine ilk arama yapiyor. Xgboost ise genislemesine ilk arama yapiyor.
from lightgbm import LGBMClassifier
lgbm_model = LGBMClassifier()
lgbm_params = {
        'n_estimators': [100, 500, 1000, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5,6],
        'learning_rate': [0.1,0.01,0.02,0.05],
        "min_child_samples": [5,10,20]}
lgbm_cv_model = GridSearchCV(lgbm, lgbm_params,
                             cv = 10,
                             n_jobs = -1,
                             verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm = LGBMClassifier(learning_rate = 0.01,
                       max_depth = 3,
                       subsample = 0.6,
                       n_estimators = 500,
                       min_child_samples = 20)
lgbm_tuned = lgbm.fit(X_train,y_train)
y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 12-12 CatBoost . Cat degiskenlerle otomatik mucadele edebilen bir model
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier()
catb_params = {
    'iterations': [200,500],
    'learning_rate': [0.01,0.05, 0.1],
    'depth': [3,5,8] }
catb = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)
catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_
catb = CatBoostClassifier(iterations = 200,
                          learning_rate = 0.05,
                          depth = 5)

catb_tuned = catb.fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
#%%
# 13 TUM MODELLERIN KARSILASTIRILMASI
modeller = [
    knn_tuned,
    loj_model,
    svc_tuned,
    nb_model,
    mlpc_tuned,
    cart_tuned,
    rf_tuned,
    gbm_tuned,
    catb_tuned,
    lgbm_tuned,
    xgb_tuned

]

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    print("-" * 28)
    print(isimler + ":")
    print("Accuracy: {:.4%}".format(dogruluk)) # Ancak standartlastirmayi unutma. aksi halde MLP skorun dusuk cikar
sonuc = []

sonuclar = pd.DataFrame(columns=["Modeller", "Accuracy"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    sonuc = pd.DataFrame([[isimler, dogruluk * 100]], columns=["Modeller", "Accuracy"])
    sonuclar = sonuclar.append(sonuc)

sns.barplot(x='Accuracy', y='Modeller', data=sonuclar, color="r")
plt.xlabel('Accuracy %')
plt.title('Modellerin Doğruluk Oranları');
#%%
#13-2 Tum modellerin ogrendiklerini birlestirip yeni bir skor elde etmek
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
        
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f}".format(score.mean()))
#%%
# 14- UNSUPERVISED LEARNING (GOZETIMSIZ OGRENME)
# 14-1 K-means ile kendi icinde homojen ama birbirinden farkli kumeler olusturmaya calisilir. Hiyerarsik olmayan kumeleme yontemidir. Rastgele noktalar belirlenir ve bu noktalara en yakin noktalar kumlenmis olur
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4, n_init=10) # n_init baslangicta konulan noktalarin kac defa yeniden konup denenecegini ifade eder
kmeans # n_cluster kume sayisi
k_fit = kmeans.fit(df) # df i fit ediyoruz bu modele
k_fit.n_clusters # kac kume oldugunu soyluyor
k_fit.cluster_centers_ # kumeleri olusturan noktalarin konumlarini ifade ediyor. Ilk satir 1.clusterin merkezi...
k_fit.labels_ # her bir gozlemin hangi kumeye ait oldugunu verir
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = kumeler, s = 50, cmap = "viridis")

merkezler = k_fit.cluster_centers_

plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5); # kumelerin merkezlerini gorsellestirdik
#3 boyutlu gorsellestirmek icin
from mpl_toolkits.mplot3d import Axes3D
#!pip install --upgrade matplotlib
#import mpl_toolkits                  #eger calismazsa bunlari calistir
kmeans = KMeans(n_clusters = 3)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_
merkezler = kmeans.cluster_centers_
plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]);
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=kumeler)
ax.scatter(merkezler[:, 0], merkezler[:, 1], merkezler[:, 2],
           marker='*',
           c='#050505',
           s=1000);
# eger kmeans icin optimum kume sayisini belirlemek istiyorsak;
xcss = []
# wcss = within cluster sum of squares
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
# simdi gorsellestirelim;
number_clusters = range(1,10)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Cluster')
plt.ylabel('Within Cluster Sum of Squares')
#yukaridaki grafikten optimum kume sayisini bulabilirsin!!!
#cok fazla degisken oldugunda once PCA analizi yapip boyut indirgemesi yapip gozlemleyebilirsin
# eger df olusturup gozlemlemek istersen;
pd.DataFrame({"Eyaletler" : df.index, "Kumeler": kumeler})[0:10]
df["kume_no"] = kumeler
df.head()
df["kume_no"] = df["kume_no"] + 1
# simdi Kmeans model tuning yapalim!
#!pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,50)) # optimum kume sayimiz kac olsun? sorusuna yanit ariyoruz. 2 ile 50 arasindaki sayilari denetiyoruz
visualizer.fit(df)
visualizer.poof()  # grafigin yorumu uzun, arastir!
kmeans = KMeans(n_clusters = 4)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_
pd.DataFrame({"Eyaletler" : df.index, "Kumeler": kumeler})[0:10]
#%%
# 14-2 Hiyerarsik kumeleme. Amac kumeleri bibirleine olan benzerliklerine gore alt kumelere ayirmaktir.
from scipy.cluster.hierarchy import linkage
hc_complete = linkage(df, "complete")
hc_average = linkage(df, "average")
hc_single = linkage(df, "single")
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('Hiyerarşik Kümeleme - Dendogram')
plt.xlabel('Indexler')
plt.ylabel('Uzaklık')
dendrogram(
    hc_complete,
    leaf_font_size=10
);
from scipy.cluster.hierarchy import dendrogram

plt.figure(figsize=(15, 10))
plt.title('Hiyerarşik Kümeleme - Dendogram')
plt.xlabel('Indexler')
plt.ylabel('Uzaklık')
dendrogram(
    hc_complete,
    truncate_mode = "lastp",
    p = 4,                          # p ile grafikte gosterilmesini istedigimiz kisimlari seceriz
    show_contracted = True
);
'''optimum kume sayisini nasil buluruz?'''
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 4,
                                  affinity = "euclidean",
                                  linkage = "ward")

cluster.fit_predict(df)
pd.DataFrame({"Eyaletler" : df.index, "Kumeler": cluster.fit_predict(df)})[0:10]
df["kume_no"] = cluster.fit_predict(df)
df.head()
#%%

# 14-3 Anomaly Detection
pip install pycaret # yeni bir environment olustur ve oraya yukle hata almamak icin
#import the dataset from pycaret repository
from pycaret.datasets import get_data
anomaly = get_data('anomaly')

#import anomaly detection module
from pycaret.anomaly import *

#intialize the setup
exp_ano = setup(anomaly)




#%%
# 15 DEEP LEARNING
# 15-1 ANN .ilk deep learning projemizi basit seviyede yapalim
from keras.datasets import mnist # mnist datasetini kullanacagiz
(x_train, y_train),(x_test,y_test) = mnist.load_data()  # verisetini train ve test seti olarak parcaladik

from keras import layers # derin ogrenmenin katmanlaridir layers, her katmanda ogrenme gerceklesir
from keras import models # modelleri filtreleyebilmek icin models kutuphanesini kullanacagiz
network = models.Sequential()
network.add(layers.Dense(512,activation= 'relu', input_shape = (28*28,))) # dense ile her noronu bir onceki norona baglariz. 512 katmandaki noron sayisi, girdilerin hangisi yuksek etkiye sahipse onu aktiflestirir 'relu' metodu, input_shape de x_train.shape yazinca cikan boyut
# simdi ikinci layer ornegini olusturalim
network.add(layers.Dense(10,activation= 'softmax'))
# sinir aglari olusturulduktan sonra modeli yapilandiralim;
network.compile(optimizer= 'rmsprop', loss = 'categorical_crossentropy', metrics='accuracy')
# modeli egitmeye baslamadan once versetini on isleyelim;
x_train = x_train.reshape((60000,28*28))
x_test = x_test.reshape((10000,28*28)) # bu reshape islemlerini kendi verisetine gore ayarlamalisin
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
# simdi etiketleri kategorik olarak kodlayalim;
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# simdi modeli calistirabiliriz;
network.fit(x_train, y_train, epochs=5, batch_size =128) # epoch her bir adimda ogrenme tekrarini ifade eder , batchsize tek seferde tum verisetini egitmek yerine alt kumelere ayirarak egitmeye yarar
# simdi modelin performansini inceleyelim;
network.evaluate(x_test,y_test) # sonuctaki ikinci sayi modelin basarisi %98 gibi.
# TENSORFLOW
import tensorflow as tf
# tensorflow csv ve excel dosyalari icin guzel calismiyor. Kendine ait bir dosya turu var '.npz' Bu degisikligi su sekilde yapabilirsin;
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)
# olusturdugun bu dosyayi tensorflow algoritmasi icin calistirmak istersen su sekilde okutabilirsin;
training_data = np.load('TF_intro.npz')
input_size = 2 # x degiskenlerinin sayisi
output_size = 1 # y target degiskeni sayisi
# derin ogrenmeyle ilgili aldigim ilk kursun icerigi;
# https://github.com/ayyucekizrak/Udemy_DerinOgrenmeyeGiris
# deep learning playground;
# https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.25973&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
# deep learning kaynaklar;
# Deep Learning Tutorial for Beginners --> https://www.kaggle.com/denzpinar/deep-learning-tutorial-for-beginners/edit  (kaggle da kayitli)
# Convolutional Neural Network (CNN) Tutorial --> https://www.kaggle.com/denzpinar/convolutional-neural-network-cnn-tutorial/edit (kaggle da kayitli)
#%%
