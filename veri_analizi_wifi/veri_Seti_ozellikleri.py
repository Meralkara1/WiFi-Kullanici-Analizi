import pandas as pd

# Dosyanızı doğru kodlama ile okuyun
veri_seti = pd.read_csv("ibb-wi-fi-noktalar-yllara-gore-kullanc-says.csv", encoding='latin1')

# Veri setinin genel bilgilerini görüntüleyin
print("Veri Setinin İlk 5 Satırı:")
print(veri_seti.head())

print("\nSütun Adları:")
print(veri_seti.columns)

print("\nVeri Türleri:")
print(veri_seti.dtypes)

print("\nEksik Değer Sayısı:")
print(veri_seti.isnull().sum())

print("\nHer Sütundaki Benzersiz Değer Sayısı:")
print(veri_seti.nunique())

print("\nVeri Setinin Temel İstatistikleri:")
print(veri_seti.describe(include="all"))

# Sütun adlarını öğrenin
sutun_adlari = veri_seti.columns
print("Sütun Adları:")
print(sutun_adlari)


print("Sütun Adları:")
print(veri_seti.columns)
