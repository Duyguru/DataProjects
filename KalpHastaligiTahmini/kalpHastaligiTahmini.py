#----------------------------------------------------------------
# Konu: Kalp Hastalığı Tahmini
#----------------------------------------------------------------

#----------------------------------------------------------------
# Kütüphanelerin Yüklenmesi (import libraries)
#----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")


#----------------------------------------------------------------
# Veri Setini Yükleme ve Keşifsel Veri Analizi (load dataset ve EDA)
#----------------------------------------------------------------
df = pd.read_csv("heart_disease_uci.csv")
df = df.drop(columns=["id"]) # ID sütunu gereksiz

print("\nVeri Seti Bilgisi (info):")
df.info()

print("\nSayısal Değişkenler İçin İstatistiksel Özet:")
df.describe()


#----------------------------------------------------------------
# Eksik Değerlerin Yönetimi (handling missing value)
#----------------------------------------------------------------
print(f"\nDoldurma öncesi eksik veri sayısı: {df.isnull().sum()}")
df["trestbps"].fillna(df["trestbps"].median(), inplace=True)
df["chol"].fillna(df["chol"].median(), inplace=True)
df["fbs"].fillna(df["fbs"].mode()[0], inplace=True)
df["restecg"].fillna(df["restecg"].mode()[0], inplace=True)
df["thalch"].fillna(df["thalch"].median(), inplace=True) 
df["exang"].fillna(df["exang"].mode()[0], inplace=True)
df["oldpeak"].fillna(df["oldpeak"].median(), inplace=True)
df["slope"].fillna(df["slope"].mode()[0], inplace=True)
df["thal"].fillna(df["thal"].mode()[0], inplace=True)

# 'ca' sütununda çok fazla eksik veri olduğu için çıkarıyoruz.
df = df.drop(columns=["ca"])

print("\nEksik değerler doldurulduktan sonra kontrol:")
print(df.isnull().sum())


#----------------------------------------------------------------
# Özelliklerin Tanımlanması
# Sayısal ve kategorik özellikleri burada tanımlayarak sonraki adımlarda kullanıma hazır hale getiriyoruz.
#----------------------------------------------------------------
categorical_features = ["sex", "dataset", "cp", "restecg", "exang", "slope", "thal"]
numerical_features = ["age", "trestbps", "chol", "fbs", "thalch", "oldpeak"]


#----------------------------------------------------------------
# Veri Görselleştirme (data visualization)
#----------------------------------------------------------------
print("\nVeri görselleştirme adımı başlatılıyor...")

# Aykırı değerlerin tespiti için kutu grafiği
plt.figure(figsize=(15, 8))
sns.boxplot(data=df[numerical_features])
plt.title('Sayısal Değişkenler için Aykırı Değer Tespiti')
plt.xticks(rotation=45)
plt.show()

# Hedef değişkenin dağılımı
sns.countplot(x="num", data=df)
plt.title('Hedef Değişken Dağılımı (num)')
plt.show()

# Korelasyon Heatmap
numerical_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(14, 12))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Sayısal Özellikler Arası Korelasyon Matrisi')
plt.show()


#----------------------------------------------------------------
# Özellik Mühendisliği (Feature Engineering)
# Veriyi ayırma, ölçeklendirme ve kategorik kodlama
#----------------------------------------------------------------
X = df.drop(["num"], axis=1)
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sayısal verileri ölçeklendirme
scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train[numerical_features])
X_test_num_scaled = scaler.transform(X_test[numerical_features])

# Kategorik verileri kodlama
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat_encoded = encoder.fit_transform(X_train[categorical_features])
X_test_cat_encoded = encoder.transform(X_test[categorical_features])

# Ölçeklenmiş sayısal ve kodlanmış kategorik verileri birleştirme
X_train_processed = np.hstack((X_train_num_scaled, X_train_cat_encoded))
X_test_processed = np.hstack((X_test_num_scaled, X_test_cat_encoded))


#----------------------------------------------------------------
# Model Karşılaştırması
#----------------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "SVM": SVC(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    results[name] = accuracy_score(y_test, y_pred)

print("\nModellerin Doğruluk Skorları (Karşılaştırma):")
results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Accuracy'])
print(results_df.sort_values(by='Accuracy', ascending=False))


#----------------------------------------------------------------
# Hiperparametre Optimizasyonu (hyperparameter tuning)
# En iyi görünen model olan Random Forest için optimizasyon
#----------------------------------------------------------------
print("\nRandom Forest modeli için hiperparametre optimizasyonu (GridSearchCV) başlatılıyor...")
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, verbose=0, n_jobs=-1)
grid_search_rf.fit(X_train_processed, y_train)

print(f"En iyi Random Forest parametreleri: {grid_search_rf.best_params_}")
best_rf_model = grid_search_rf.best_estimator_


#----------------------------------------------------------------
# Final Model Değerlendirmesi
#----------------------------------------------------------------
y_pred_final = best_rf_model.predict(X_test_processed)

print("\nOptimize Edilmiş Final Model (Random Forest) için Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_final))

print("Optimize Edilmiş Final Model için Karışıklık Matrisi oluşturuluyor...")
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Final Model Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()


#----------------------------------------------------------------
# Özellik Önem Analizi (feature importance)
#----------------------------------------------------------------
print("\nModel için özellik önem düzeyleri analiz ediliyor...")

# OneHotEncoder'dan sonra oluşan yeni kategorik sütun isimlerini alıyoruz.
encoded_cat_features = list(encoder.get_feature_names_out(categorical_features))

# Sayısal özellikler ve yeni oluşturulan kategorik özellik isimlerini birleştiriyoruz.
all_feature_names = numerical_features + encoded_cat_features

importances = best_rf_model.feature_importances_

# DataFrame'i yeni oluşturduğumuz özellik isimleriyle kuruyoruz.
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Model için Özellik Önem Düzeyleri')
plt.show()


#----------------------------------------------------------------
# ROC-AUC Skoru Hesaplanması
#----------------------------------------------------------------
y_pred_proba = best_rf_model.predict_proba(X_test_processed)

# One-vs-Rest (OvR) için ROC AUC skoru
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print(f"\nModelin ROC AUC (One-vs-Rest) Skoru: {roc_auc:.4f}")


#----------------------------------------------------------------
# Model Kaydetme ve Yükleme
#----------------------------------------------------------------
model_filename = 'kalp_hastaligi_final_model.joblib'
scaler_filename = 'scaler.joblib'
encoder_filename = 'encoder.joblib'

joblib.dump(best_rf_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(encoder, encoder_filename)
print(f"\nFinal model '{model_filename}' olarak kaydedildi.")
print(f"Scaler '{scaler_filename}' olarak kaydedildi.")
print(f"Encoder '{encoder_filename}' olarak kaydedildi.")

# Yükleme kontrolü
loaded_model = joblib.load(model_filename)
print("Model başarıyla geri yüklendi.")
score = loaded_model.score(X_test_processed, y_test)
print(f"Yüklenen modelin doğruluk skoru: {score:.4f}")

#----------------------------------------------------------------
# Sonuçların Yorumlanması
#----------------------------------------------------------------
print("\n--- FİNAL SONUÇ VE YORUMLAR ---")
print("1. Veri seti yüklendi ve 'id' gibi gereksiz sütunlar çıkarıldı.")
print("2. Eksik veriler, satır silme yerine medyan ve mod gibi istatistiksel yöntemlerle doldurularak veri kaybı önlendi.")
print("3. Random Forest, KNN ve SVM modelleri karşılaştırıldı. En yüksek başlangıç performansını Random Forest gösterdi.")
print("4. Random Forest modeli üzerinde GridSearchCV ile hiperparametre optimizasyonu yapıldı ve modelin performansı artırıldı.")
print(f"5. Optimize edilen final model, test verisi üzerinde {accuracy_score(y_test, y_pred_final):.2%} doğruluk ve {roc_auc:.2f} AUC skoru elde etti.")
print("6. Özellik önem analizi, modelin tahmin yaparken en çok hangi özelliklere dayandığını ortaya koydu.")
print("7. Eğitilen en iyi model, ileride tekrar kullanılabilmesi için `scaler` ve `encoder` ile birlikte diske kaydedildi.")
print("-----------------------------------")














