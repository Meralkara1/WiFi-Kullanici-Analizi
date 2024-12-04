import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Veriyi yükleyin
veri_seti = pd.read_csv("ibb-wi-fi-noktalar-yllara-gore-kullanc-says.csv", sep=';', encoding="latin1")

print("Sütun Adları:")
print(veri_seti.columns)

# Veri setinin genel durumunu kontrol edin
print("Veri Setinin İlk 5 Satırı:")
print(veri_seti.head())

# Veri türlerini kontrol edin
print("\nVeri Türleri:")
print(veri_seti.dtypes)

# Eksik değerleri kontrol edin
print("\nEksik Değer Sayısı:")
print(veri_seti.isnull().sum())


# Yıllara göre toplam kullanıcı sayısı
yillik_kullanicilar = veri_seti[['2019', '2020', '2021']].astype(int).sum()
print("\nYıllara Göre Toplam Kullanıcı Sayısı:")
print(yillik_kullanicilar)

# Yıllara göre kullanıcı sayısını görselleştirme
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
yillik_kullanicilar.plot(kind="bar", color=["blue", "orange", "green"])
plt.title("Yıllara Göre Kullanıcı Sayısı")
plt.xlabel("Yıllar")
plt.ylabel("Toplam Kullanıcı Sayısı")
plt.xticks(rotation=0)
plt.show()

# En çok kullanılan ilk 10 Wi-Fi noktasını analiz etme
veri_seti['Toplam Kullanıcı'] = veri_seti[['2019', '2020', '2021']].sum(axis=1)
en_cok_kullanilan = veri_seti.nlargest(10, 'Toplam Kullanıcı')
print("\nEn Çok Kullanılan İlk 10 Wi-Fi Noktası:")
print(en_cok_kullanilan[['Konum', 'Toplam Kullanıcı']])

# En çok kullanılan lokasyonları görselleştirme
plt.figure(figsize=(10, 6))
plt.barh(en_cok_kullanilan['Konum'], en_cok_kullanilan['Toplam Kullanıcı'], color='purple')
plt.title("En Çok Kullanılan İlk 10 Wi-Fi Noktası")
plt.xlabel("Toplam Kullanıcı Sayısı")
plt.ylabel("Konum")
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Özellikler ve hedef değişken
X = veri_seti[['2019', '2020']].astype(int)  # Girdi özellikleri
y = veri_seti['2021'].astype(int)  # Hedef değişken

# Veriyi eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Modelin değerlendirilmesi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nLinear Regression Modeli - Mean Squared Error: {mse:.2f}, R2 Score: {r2:.2f}")

# Gerçek ve tahmin edilen değerleri karşılaştırma
tahmin_df = pd.DataFrame({'Gerçek Değerler': y_test, 'Tahmin Edilen Değerler': y_pred})
print("\nGerçek ve Tahmin Edilen Değerler:")
print(tahmin_df.head())


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Modelleri tanımlama
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Modelleri eğitme ve değerlendirme
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} - Mean Squared Error: {mse:.2f}, R2 Score: {r2:.2f}")


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.title("Gerçek ve Tahmin Edilen Değerler")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.grid(True)
plt.show()

from sklearn.model_selection import GridSearchCV

# Hiperparametre ızgarası
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# GridSearchCV ile en iyi parametreleri bulma
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='r2')
grid_search.fit(X_train, y_train)

print(f"En İyi Parametreler: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Tahmin ve değerlendirme
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Random Forest (Optimize Edilmiş) - MSE: {mse:.2f}, R2: {r2:.2f}")


plt.figure(figsize=(8, 6))
sns.heatmap(veri_seti[['2019', '2020', '2021']].corr(), annot=True, cmap='coolwarm')
plt.title("Korelasyon Matrisi")
plt.show()

veri_seti.set_index('Konum', inplace=True)
veri_seti[['2019', '2020', '2021']].T.plot(figsize=(10, 6))
plt.title("Yıllara Göre Kullanıcı Dağılımı")
plt.xlabel("Yıllar")
plt.ylabel("Kullanıcı Sayısı")
plt.grid(True)
plt.show()

################################################################################################

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Yıllara göre toplam kullanıcı sayısını hazırlama
yillik_kullanicilar = veri_seti[['2019', '2020', '2021']].sum()

# ARIMA Modeli
model = ARIMA(yillik_kullanicilar, order=(1, 1, 1))  # (p, d, q) parametreleri
model_fit = model.fit()

# Tahmin yapma
forecast = model_fit.forecast(steps=2)  # 2 yıl için tahmin
print("Gelecek Yıllar için Tahmin:")
print(f"2022: {forecast[0]:.0f}")
print(f"2023: {forecast[1]:.0f}")

# Modelin tahminlerini ve gerçek değerleri görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(yillik_kullanicilar.index, yillik_kullanicilar, label="Gerçek")
plt.plot(['2022', '2023'], forecast, label="Tahmin", linestyle="--", color="red")
plt.title("ARIMA ile Gelecek Yıllar için Kullanıcı Tahmini")
plt.xlabel("Yıllar")
plt.ylabel("Toplam Kullanıcı Sayısı")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


from prophet import Prophet
import pandas as pd

# Prophet için veri setini hazırlama
df = yillik_kullanicilar.reset_index()
df.columns = ['ds', 'y']  # Prophet için tarih ve değer kolonlarını yeniden adlandır

# Prophet Modeli
model = Prophet()
model.fit(df)

# Gelecekteki yılları tahmin etme
future = model.make_future_dataframe(periods=2, freq='Y')  # 2 yıl için tahmin
forecast = model.predict(future)

# Tahmin sonuçlarını görselleştirme
fig = model.plot(forecast)
plt.title("Prophet ile Gelecek Yıllar için Kullanıcı Tahmini")
plt.xlabel("Yıllar")
plt.ylabel("Toplam Kullanıcı Sayısı")
plt.show()

# Tahmin sonuçlarını yazdırma
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


future = model.make_future_dataframe(periods=10, freq='Y')  # 10 yıl için tahmin


from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Yıllık kullanıcı verilerini Prophet formatına uygun hale getirme
yillik_kullanicilar = veri_seti[['2019', '2020', '2021']].sum().reset_index()
yillik_kullanicilar.columns = ['ds', 'y']  # Prophet'in gerektirdiği sütun adları

# Prophet Modeli
model = Prophet()
model.fit(yillik_kullanicilar)

# Gelecekteki yıllar için tahmin yapma
future = model.make_future_dataframe(periods=10, freq='Y')  # 10 yıllık tahmin
forecast = model.predict(future)

# Tahmin sonuçlarını görselleştirme
fig = model.plot(forecast)
plt.title("Prophet ile 10 Yıllık Kullanıcı Tahmini")
plt.xlabel("Yıllar")
plt.ylabel("Toplam Kullanıcı Sayısı")
plt.grid(True)
plt.show()

# Tahmin edilen son yılları yazdırma
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())


# Tahmin bileşenlerini görselleştirme
fig2 = model.plot_components(forecast)
plt.show()
