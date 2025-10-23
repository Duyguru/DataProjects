#----------------------------------------------------------------
# KRONİK BÖBREK HASTALIĞI TAHMİNİ 
#----------------------------------------------------------------

# Kütüphanelerin Yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

import warnings
warnings.filterwarnings("ignore")
print("Kütüphaneler başarıyla yüklendi.")

#----------------------------------------------------------------
# Veri Yükleme, Temizleme ve EDA 
#----------------------------------------------------------------
df = pd.read_csv("kidney_disease.csv")
df.drop("id", axis=1, inplace=True)

# Sütunları yeniden adlandırma
df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
              'red_blood_cells', 'pus_cell', 'pus_cell_clumbs',
              'bacteria', 'blood_glucose_random', 'blood_urea',
              'serum_creatinine', 'sodium', 'potassium', 'hemoglobin',
              'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
              'hypertension', 'diabetes_mellitus', 'coronary_artery_disease',
              'appetite', 'peda_edema', 'aanemia', 'class']

# Veri tiplerini düzeltme
df["packed_cell_volume"] = pd.to_numeric(df["packed_cell_volume"], errors="coerce")
df["white_blood_cell_count"] = pd.to_numeric(df["white_blood_cell_count"], errors="coerce")
df["red_blood_cell_count"] = pd.to_numeric(df["red_blood_cell_count"], errors="coerce")

# Kirli verileri temizleme
df["diabetes_mellitus"].replace(to_replace = {'\tno':"no", '\tyes': "yes", 'yes':"yes"}, inplace=True)
df["coronary_artery_disease"].replace(to_replace = {'\tno':"no"}, inplace=True)
df["class"].replace(to_replace = {'ckd\t':"ckd"},inplace=True)

# Hedef değişkeni 0 ve 1'e çevirme
df["class"] = df["class"].map({"ckd":1, "notckd":0}) # 1: Hasta, 0: Sağlıklı

# Özellikleri kategorik ve numerik olarak ayırma
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
num_cols = [col for col in df.columns if df[col].dtype != "object" and col != "class"]

# Eksik Değerleri Doldurma 
def solve_mv_random_value(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(),feature] = random_sample
for col in num_cols:
    solve_mv_random_value(col)

def solve_mv_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)
for col in cat_cols:
    solve_mv_mode(col)

# Kategorik Verileri Kodlama (Label Encoding)
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

#----------------------------------------------------------------
# Veriyi Modellemeye Hazırlama
#----------------------------------------------------------------
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

# Standardizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nVeri, eğitim ve test olarak ayrıldı ve ölçeklendirildi.")

#----------------------------------------------------------------
# Model Karşılaştırması
#----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
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

print("\nDecision Tree modeli için hiperparametre optimizasyonu (GridSearchCV) başlatılıyor...")
param_grid_dt = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, verbose=0, n_jobs=-1)
grid_search_dt.fit(X_train_scaled, y_train)

print(f"En iyi Decision Tree parametreleri: {grid_search_dt.best_params_}")
best_dt_model = grid_search_dt.best_estimator_

#----------------------------------------------------------------
# Final Model Değerlendirmesi
#----------------------------------------------------------------
y_pred_final = best_dt_model.predict(X_test_scaled)

# ROC-AUC Skoru
y_pred_proba = best_dt_model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n--- Optimize Edilmiş Final Model (Decision Tree) Değerlendirmesi ---")
print(f"Doğruluk (Accuracy): {accuracy_score(y_test, y_pred_final):.4f}")
print(f"ROC AUC Skoru: {roc_auc:.4f}")

print("\nKarışıklık Matrisi (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred_final))

print("\nSınıflandırma Raporu (Classification Report):")
print(classification_report(y_test, y_pred_final))

#----------------------------------------------------------------
# Model Yorumlama 
#----------------------------------------------------------------
# DT visualization - feature importance
feature_names = X.columns
class_names_str = ["notckd","ckd"] # 0: notckd, 1: ckd

plt.figure(figsize=(25,15))
plot_tree(best_dt_model, feature_names=feature_names, class_names=class_names_str, filled=True, rounded=True, fontsize=10, max_depth=3)
plt.title("Optimize Edilmiş Karar Ağacı Yapısı (İlk 3 Seviye)")
plt.show()

feature_importance = pd.DataFrame({"Feature":feature_names, "Importance":best_dt_model.feature_importances_})
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance.sort_values(by="Importance", ascending=False))
plt.title("Özellik Önem Düzeyleri")
plt.show()

#----------------------------------------------------------------
# Model Kaydetme ve Yükleme
#----------------------------------------------------------------
model_filename = 'ckd_final_model.joblib'
scaler_filename = 'ckd_scaler.joblib'

joblib.dump(best_dt_model, model_filename)
joblib.dump(scaler, scaler_filename)
print(f"\nFinal model '{model_filename}' olarak kaydedildi.")
print(f"Scaler '{scaler_filename}' olarak kaydedildi.")

# Yükleme kontrolü
loaded_model = joblib.load(model_filename)
score = loaded_model.score(X_test_scaled, y_test)
print(f"Yüklenen modelin doğruluk skoru: {score:.4f}")










































