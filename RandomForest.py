import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Директория для сохранения изображений
OUT_DIR = 'random_forest_results'
os.makedirs(OUT_DIR, exist_ok=True)

# ========== 1. Загрузка данных и предобработка ==========
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

df = pd.read_csv('adult.data', header=None, names=col_names, na_values=' ?')
df = df.dropna()
df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})

X = df.drop('income', axis=1)
y = df['income']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ========== 2. Базовое дерево решений (для сравнения) ==========
pipe_dt = Pipeline([
    ('prep', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=42))
])
pipe_dt.fit(X_train, y_train)
dt_pred = pipe_dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred, average='weighted')
print(f"Single Decision Tree: Accuracy={dt_acc:.4f}, F1={dt_f1:.4f}")

# ========== 3. Случайный лес: перебор n_estimators ==========
n_list = list(range(50, 101, 10))
rf_acc, rf_f1, oob_scores = [], [], []
best_model, best_n, best_acc = None, None, -np.inf

for n in n_list:
    pipe_rf = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=n,
            max_samples=1.0,        # параметр, аналогичный бэггингу (доля объектов на дерево)
            max_features='sqrt',    # ключевое отличие RF: случайное подмножество признаков
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42
        ))
    ])
    pipe_rf.fit(X_train, y_train)
    pred = pipe_rf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    oob = pipe_rf.named_steps['clf'].oob_score_

    rf_acc.append(acc)
    rf_f1.append(f1)
    oob_scores.append(oob)

    if acc > best_acc:
        best_acc = acc
        best_n = n
        best_model = pipe_rf

    print(f"n={n}: Acc={acc:.4f}, F1={f1:.4f}, OOB={oob:.4f}")

# ========== 4. Визуализация: Accuracy и F1 от числа деревьев ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Accuracy ---
axes[0].plot(n_list, rf_acc, marker='o', linewidth=2, color='#2E86AB', label='Random Forest')
axes[0].axhline(y=dt_acc, color='r', linestyle='--', linewidth=2, label='Single Tree')
for x, yv in zip(n_list, rf_acc):
    axes[0].annotate(f'{yv:.4f}', (x, yv), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
axes[0].annotate(f'{dt_acc:.4f}', (n_list[0], dt_acc),
                 textcoords='offset points', xytext=(0, -14),
                 ha='left', fontsize=8, color='r')
axes[0].set_xlabel('n_estimators', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Accuracy vs Ensemble Size (Random Forest)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# --- F1 ---
axes[1].plot(n_list, rf_f1, marker='s', linewidth=2, color='#F18F01', label='Random Forest')
axes[1].axhline(y=dt_f1, color='r', linestyle='--', linewidth=2, label='Single Tree')
for x, yv in zip(n_list, rf_f1):
    axes[1].annotate(f'{yv:.4f}', (x, yv), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
axes[1].annotate(f'{dt_f1:.4f}', (n_list[0], dt_f1),
                 textcoords='offset points', xytext=(0, -14),
                 ha='left', fontsize=8, color='r')
axes[1].set_xlabel('n_estimators', fontsize=12)
axes[1].set_ylabel('F1-score (weighted)', fontsize=12)
axes[1].set_title('F1-score vs Ensemble Size (Random Forest)', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'rf_accuracy_f1_vs_n.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. OOB + сравнение с одиночным деревом ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(n_list, oob_scores, marker='D', linewidth=2, color='#3A86FF', label='OOB score')
for x, yv in zip(n_list, oob_scores):
    axes[0].annotate(f'{yv:.4f}', (x, yv), textcoords='offset points',
                     xytext=(0, 8), ha='center', fontsize=8)
axes[0].set_xlabel('n_estimators', fontsize=12)
axes[0].set_ylabel('OOB-score', fontsize=12)
axes[0].set_title('Out-of-Bag score (Random Forest)', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

metrics_names = ['Accuracy', 'F1-score']
single_vals = [dt_acc, dt_f1]
best_vals = [rf_acc[n_list.index(best_n)], rf_f1[n_list.index(best_n)]]
x_pos = np.arange(len(metrics_names))
width = 0.35

bars1 = axes[1].bar(x_pos - width/2, single_vals, width,
                    label='Single Tree', color='#D62828', alpha=0.8)
bars2 = axes[1].bar(x_pos + width/2, best_vals, width,
                    label=f'Random Forest (n={best_n})', color='#2E86AB', alpha=0.8)
for bar in bars1:
    axes[1].annotate(f'{bar.get_height():.4f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', fontsize=10)
for bar in bars2:
    axes[1].annotate(f'{bar.get_height():.4f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', fontsize=10)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(metrics_names)
axes[1].set_ylabel('Значение метрики')
axes[1].set_title('Single Tree vs Best Random Forest', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'rf_oob_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. Сохранение лучшей модели ==========
print(f"\n=== Лучшая модель (n={best_n}, Acc={best_acc:.4f}) ===")
artifact = {
    'model': best_model,
    'n_estimators': best_n,
    'test_accuracy': best_acc,
    'test_f1': rf_f1[n_list.index(best_n)],
    'feature_names': list(X.columns),
    'cat_cols': cat_cols,
    'num_cols': num_cols,
}
model_path = os.path.join(OUT_DIR, 'best_rf_model.joblib')
joblib.dump(artifact, model_path)
print(f"Модель сохранена в {model_path}")

# ========== 7. Проверка на первых 10 объектах ==========
print("\n=== Проверка модели на первых 10 объектах ===")
loaded = joblib.load(model_path)
model = loaded['model']
print(f"Загружено: n_estimators={loaded['n_estimators']}, "
      f"test_accuracy={loaded['test_accuracy']:.4f}")

X_sample = X_test.iloc[:10]
y_sample = y_test.iloc[:10]
y_pred = model.predict(X_sample)
y_proba = model.predict_proba(X_sample)

label_map = {0: '<=50K', 1: '>50K'}
df_pred = pd.DataFrame({
    'sample_idx': range(10),
    'true_label': [label_map[v] for v in y_sample.values],
    'predicted': [label_map[v] for v in y_pred],
    'proba_<=50K': y_proba[:, 0].round(4),
    'proba_>50K': y_proba[:, 1].round(4),
    'correct': (y_pred == y_sample.values).astype(int),
})

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
print(df_pred.to_string(index=False))

acc_10 = df_pred['correct'].mean()
print(f"\nТочность на первых 10 объектах: {acc_10:.2f} "
      f"({df_pred['correct'].sum()}/10)")

df_pred.to_csv(os.path.join(OUT_DIR, 'predictions_first_10.csv'), index=False)
print(f"Предсказания сохранены в {os.path.join(OUT_DIR, 'predictions_first_10.csv')}")

# ========== 8. Функция предсказания для новых данных ==========
def predict_custom(df_input, model_path=os.path.join(OUT_DIR, 'best_rf_model.joblib')):
    """
    Предсказание для произвольного DataFrame с той же структурой, что и X.
    Пример:
        df_new = X_test.iloc[[0, 5, 7]]
        predict_custom(df_new)
    """
    art = joblib.load(model_path)
    m = art['model']
    pred = m.predict(df_input)
    proba = m.predict_proba(df_input)
    out = pd.DataFrame({
        'prediction': [label_map[v] for v in pred],
        'proba_<=50K': proba[:, 0].round(4),
        'proba_>50K': proba[:, 1].round(4),
    }, index=df_input.index)
    return out

print("\n=== Предсказание для произвольных объектов (idx 0, 5, 7) ===")
print(predict_custom(X_test.iloc[[0, 5, 7]]).to_string())