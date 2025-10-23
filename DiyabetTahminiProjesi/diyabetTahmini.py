#----------------------------------------------------------------
# DİYABET TAHMİN MODELİ
#----------------------------------------------------------------

# Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("Kütüphaneler başarıyla yüklendi.")

#----------------------------------------------------------------
#  Veri Setini Yükleme ve İlk Analiz (EDA)
#----------------------------------------------------------------
df = pd.read_csv('diabetes.csv')
print("\nVeri seti başarıyla yüklendi.")
print("\nVeri Setinin İlk 5 Satırı:")
print(df.head())

print("\nVeri Seti Bilgisi:")
df.info()

print("\nİstatistiksel Özet:")
print(df.describe())

#----------------------------------------------------------------
#  Veri Temizleme ve Özellik Mühendisliği
#----------------------------------------------------------------


cols_to_clean = ['Glucose', 'Blood_Pressure', 'Skin_Thickness', 'Insulin', 'BMI']


for col in cols_to_clean:
    df[col] = df[col].replace(0, np.nan)


for col in cols_to_clean:
    df[col].fillna(df[col].median(), inplace=True)

#----------------------------------------------------------------
# Veri Görselleştirme
#----------------------------------------------------------------


# Hedef değişken dağılımı
plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df)
plt.title('Hedef Değişken Dağılımı (0: Diyabet Yok, 1: Diyabet Var)')
plt.show()

# Sayısal değişkenlerin dağılımı
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Özelliklerin Dağılımı")
plt.show()

# Korelasyon Matrisi
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.show()

#----------------------------------------------------------------
#  Veriyi Modellemeye Hazırlama
#----------------------------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#----------------------------------------------------------------
# Model Karşılaştırması
#----------------------------------------------------------------
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(),
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
#----------------------------------------------------------------
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, verbose=0, n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)

print(f"En iyi Random Forest parametreleri: {grid_search_rf.best_params_}")
best_rf_model = grid_search_rf.best_estimator_

#----------------------------------------------------------------
# Final Model Değerlendirmesi
#----------------------------------------------------------------
y_pred_final = best_rf_model.predict(X_test_scaled)
y_pred_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]

print("\nOptimize Edilmiş Final Model (Random Forest) için Değerlendirme:")
print(f"\nDoğruluk (Accuracy): {accuracy_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC Skoru: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_final))

# Karışıklık Matrisi
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('Final Model Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

#----------------------------------------------------------------
# Özet
#----------------------------------------------------------------
print("\n--- PROJE ÖZETİ  ---")
print("1. Pima Indians Diabetes veri seti yüklendi ve analiz edildi.")
print("2. 'Glucose', 'BloodPressure' gibi sütunlardaki fizyolojik olarak imkansız olan '0' değerleri NaN olarak kabul edilip medyan ile dolduruldu.")
print("3. KNN ve Random Forest modelleri karşılaştırıldı, Random Forest daha iyi performans gösterdi.")
print("4. Random Forest modeli, GridSearchCV ile optimize edildi.")
print(f"5. Final model, test verisi üzerinde yaklaşık {accuracy_score(y_test, y_pred_final):.2%} doğruluk ve {roc_auc_score(y_test, y_pred_proba):.2f} ROC AUC skoru elde etti.")
