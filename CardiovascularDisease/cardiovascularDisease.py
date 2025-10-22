#----------------------------------------------------------------
#CARDIOVASCULAR DISEASE DATASET İLE SINIFLANDIRMA MODELİ
#----------------------------------------------------------------

#----------------------------------------------------------------
#  Kütüphanelerin Yüklenmesi
#----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")


#----------------------------------------------------------------
# Veri Setini Yükleme ve İlk Analiz (EDA)
#----------------------------------------------------------------
df = pd.read_csv('cardio_train.csv', sep=';')
print("\nVeri seti başarıyla yüklendi.")
print("\nVeri Setinin İlk 5 Satırı:")
print(df.head())

# Veri setinin yapısını kontrol edelim
print("\nVeri Seti Bilgisi:")
df.info()

# Eksik veri kontrolü (Bu veri setinde eksik veri bulunmuyor)
print("\nEksik Değer Sayısı:")
print(df.isnull().sum().sum())

#----------------------------------------------------------------
# Veri Temizleme ve Özellik Mühendisliği
#----------------------------------------------------------------
print("\nVeri temizleme ve özellik mühendisliği adımı başlatılıyor...")

# 1. 'id' sütununu kaldırma
df.drop('id', axis=1, inplace=True)

# 2. 'age' sütununu gün'den 'yıl'a çevirme
df['age'] = (df['age'] / 365).round().astype('int')

# 3. Aykırı (hatalı) kan basıncı değerlerini temizleme
# Sistolik (ap_hi) ve Diastolik (ap_lo) kan basıncı mantık dışı olanları filtreleme
df.drop(df[(df['ap_hi'] > 250) | (df['ap_hi'] < 40)].index, inplace=True)
df.drop(df[(df['ap_lo'] > 200) | (df['ap_lo'] < 40)].index, inplace=True)
df.drop(df[df['ap_hi'] < df['ap_lo']].index, inplace=True)

# 4. Boy ve kilo için aykırı değerleri temizleme (Quantile Yöntemi)
df.drop(df[(df['height'] < df['height'].quantile(0.025)) | (df['height'] > df['height'].quantile(0.975))].index, inplace=True)
df.drop(df[(df['weight'] < df['weight'].quantile(0.025)) | (df['weight'] > df['weight'].quantile(0.975))].index, inplace=True)

# 5. Vücut Kitle İndeksi (BMI) özelliğini oluşturma
# BMI = kilo (kg) / boy (m)^2
df['bmi'] = (df['weight'] / (df['height'] / 100)**2).round(2)

print(f"Veri temizleme sonrası veri seti boyutu: {df.shape}")
print("Yeni 'age' (yıl) ve 'bmi' sütunları eklendi.")

#----------------------------------------------------------------
# Veri Görselleştirme
#----------------------------------------------------------------
print("\nVeri görselleştirme adımı...")

# Hedef değişken dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=df)
plt.title('Hedef Değişken Dağılımı (0: Sağlıklı, 1: Hasta)')
plt.show()

# Sayısal değişkenlerin dağılımı
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
df[numerical_features].hist(bins=30, figsize=(15, 10))
plt.suptitle("Sayısal Değişkenlerin Dağılımı")
plt.show()

# Korelasyon Matrisi
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.show()

#----------------------------------------------------------------
# Veriyi Modellemeye Hazırlama
#----------------------------------------------------------------
# Özellikler (X) ve Hedef (y) olarak ayırma
X = df.drop('cardio', axis=1)
y = df['cardio']

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Sayısal verileri ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nEğitim seti boyutu: {X_train_scaled.shape}")
print(f"Test seti boyutu: {X_test_scaled.shape}")

#----------------------------------------------------------------
# Model Karşılaştırması
#----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = accuracy_score(y_test, y_pred)

print("\nModellerin Doğruluk Skorları (Karşılaştırma):")
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(results_df.sort_values(by='Accuracy', ascending=False))

#----------------------------------------------------------------
# Hiperparametre Optimizasyonu (En iyi model için)
# Logistic Regression genellikle bu tür verilerde iyi ve hızlıdır, onu optimize edelim.
#----------------------------------------------------------------
print("\nLogistic Regression modeli için hiperparametre optimizasyonu (GridSearchCV) başlatılıyor...")
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, verbose=0, n_jobs=-1)
grid_search_lr.fit(X_train_scaled, y_train)

print(f"En iyi Logistic Regression parametreleri: {grid_search_lr.best_params_}")
best_lr_model = grid_search_lr.best_estimator_

#----------------------------------------------------------------
#Final Model Değerlendirmesi
#----------------------------------------------------------------
y_pred_final = best_lr_model.predict(X_test_scaled)
y_pred_proba = best_lr_model.predict_proba(X_test_scaled)[:, 1]

print("\nOptimize Edilmiş Final Model (Logistic Regression) için Değerlendirme:")
print(f"\nDoğruluk (Accuracy): {accuracy_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC Skoru: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_final))

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Final Model Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

#----------------------------------------------------------------
# Özet
#----------------------------------------------------------------
print("\n---ÖZET---")
print("1. Cardiovascular Disease veri seti yüklendi, temizlendi ve analiz edildi.")
print("2. 'age' sütunu yıla çevrildi, BMI özelliği eklendi ve hatalı veriler (kan basıncı, boy, kilo) temizlendi.")
print("3. Logistic Regression ve Random Forest modelleri karşılaştırıldı.")
print("4. Logistic Regression modeli, GridSearchCV ile optimize edildi.")
print(f"5. Final model, test verisi üzerinde {accuracy_score(y_test, y_pred_final):.2%} doğruluk ve {roc_auc_score(y_test, y_pred_proba):.2f} ROC AUC skoru elde etti.")
