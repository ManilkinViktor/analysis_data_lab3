"""
Полная реализация дерева решений без использования scikit-learn.
Поддерживаемые критерии: information gain (entropy), gain ratio, gini index.
Набор данных: Census Income (Adult).
"""

import os
import math
import random
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income'
]

def download_if_needed(path, url):
    if not os.path.exists(path):
        print(f"Скачивание {url} -> {path}")
        urllib.request.urlretrieve(url, path)

def load_raw(train_path='adult.data', test_path='adult.test'):
    download_if_needed(train_path, TRAIN_URL)
    download_if_needed(test_path, TEST_URL)

    df_train = pd.read_csv(train_path, header=None, names=COLUMNS,
                           skipinitialspace=True, na_values='?')
    df_test  = pd.read_csv(test_path,  header=None, names=COLUMNS,
                           skipinitialspace=True, na_values='?', skiprows=1)

    # Целевой признак: убираем точку в тестовом файле (">50K.")
    df_train['income'] = df_train['income'].astype(str).str.strip().str.rstrip('.')
    df_test['income']  = df_test['income'].astype(str).str.strip().str.rstrip('.')

    return df_train, df_test


# ============================================================
# 2. ПРЕДОБРАБОТКА
# ============================================================

class LabelEncoderCustom:
    def __init__(self):
        self.mapping = {}

    def fit(self, values):
        uniq = sorted(set(values))
        self.mapping = {v: i for i, v in enumerate(uniq)}
        return self

    def transform(self, values):
        return np.array([self.mapping.get(v, -1) for v in values], dtype=float)


def preprocess(df_train, df_test):
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Заполнение пропусков модой
    for col in df_all.columns:
        if df_all[col].isna().any():
            mode_val = df_all[col].mode()[0]
            df_all[col] = df_all[col].fillna(mode_val)

    # Целевой признак -> 0/1
    df_all['income'] = df_all['income'].astype(str).str.strip().str.rstrip('.')
    df_all['income'] = df_all['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

    # Кодирование категориальных признаков:
    # проверяем НЕ dtype, а реально ли значения являются строками
    encoders = {}
    for col in df_all.columns:
        if col == 'income':
            continue
        # Пытаемся привести к float; если не получается — это категориальный признак
        try:
            df_all[col] = df_all[col].astype(float)
        except (ValueError, TypeError):
            le = LabelEncoderCustom().fit(df_all[col].astype(str).values)
            df_all[col] = le.transform(df_all[col].astype(str).values)
            encoders[col] = le

    n_train = len(df_train)
    feature_cols = [c for c in df_all.columns if c != 'income']

    X_train = df_all.iloc[:n_train][feature_cols].values.astype(float)
    y_train = df_all.iloc[:n_train]['income'].values.astype(int)
    X_test  = df_all.iloc[n_train:][feature_cols].values.astype(float)
    y_test  = df_all.iloc[n_train:]['income'].values.astype(int)

    return X_train, y_train, X_test, y_test, feature_cols


# ============================================================
# 3. КРИТЕРИИ РАЗБИЕНИЯ
# ============================================================

def entropy(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def gini_index(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)

def information_gain(y, y_left, y_right):
    n, nl, nr = len(y), len(y_left), len(y_right)
    if nl == 0 or nr == 0:
        return 0.0
    return entropy(y) - (nl / n) * entropy(y_left) - (nr / n) * entropy(y_right)

def split_information(y_left, y_right, n):
    si = 0.0
    for part in (len(y_left), len(y_right)):
        p = part / n
        if p > 0:
            si -= p * math.log2(p)
    return si

def gain_ratio(y, y_left, y_right):
    ig = information_gain(y, y_left, y_right)
    si = split_information(y_left, y_right, len(y))
    return ig / si if si > 0 else 0.0

def gini_gain(y, y_left, y_right):
    n, nl, nr = len(y), len(y_left), len(y_right)
    if nl == 0 or nr == 0:
        return 0.0
    return gini_index(y) - (nl / n) * gini_index(y_left) - (nr / n) * gini_index(y_right)


# ============================================================
# 4. УЗЕЛ ДЕРЕВА И ВЫБОР ЛУЧШЕГО РАЗБИЕНИЯ
# ============================================================

class Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value', 'is_leaf')
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 value=None, is_leaf=False):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.is_leaf = is_leaf


def best_split(X, y, criterion, n_thresholds=20):
    """
    Поиск лучшего разбиения. Для каждого признака берётся не более
    n_thresholds квантильных порогов (ускорение на больших данных).
    """
    n_samples, n_features = X.shape
    if n_samples < 2:
        return None, None, -np.inf

    # Функция оценки разбиения
    if criterion == 'entropy':
        scorer = lambda yl, yr: information_gain(y, yl, yr)
    elif criterion == 'gini':
        scorer = lambda yl, yr: gini_gain(y, yl, yr)
    elif criterion == 'gain_ratio':
        scorer = lambda yl, yr: gain_ratio(y, yl, yr)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    best = (None, None, -np.inf)

    for feat in range(n_features):
        col = X[:, feat]
        uniq = np.unique(col)
        if len(uniq) < 2:
            continue

        # Квантильные пороги
        if len(uniq) > n_thresholds:
            qs = np.linspace(0, 100, n_thresholds + 2)[1:-1]
            cand = np.unique(np.percentile(col, qs))
        else:
            cand = (uniq[:-1] + uniq[1:]) / 2.0

        for thr in cand:
            left_mask = col <= thr
            nl = int(left_mask.sum())
            if nl == 0 or nl == n_samples:
                continue
            right_mask = ~left_mask
            score = scorer(y[left_mask], y[right_mask])
            if score > best[2]:
                best = (feat, thr, score)

    return best


# ============================================================
# 5. КЛАСС ДЕРЕВА РЕШЕНИЙ
# ============================================================

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, n_thresholds=20):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_thresholds = n_thresholds
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _build(self, X, y, depth):
        # Остановка
        if (len(y) < self.min_samples_split or
            len(np.unique(y)) == 1 or
            (self.max_depth is not None and depth >= self.max_depth)):
            return Node(value=self._majority(y), is_leaf=True)

        feat, thr, score = best_split(X, y, self.criterion, self.n_thresholds)
        if feat is None or score <= 0:
            return Node(value=self._majority(y), is_leaf=True)

        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return Node(value=self._majority(y), is_leaf=True)

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)
        return Node(feature=feat, threshold=thr, left=left, right=right, is_leaf=False)

    @staticmethod
    def _majority(y):
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value


# ============================================================
# 6. МЕТРИКИ КАЧЕСТВА
# ============================================================

def compute_metrics(y_true, y_pred, pos_label=1):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((y_pred == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((y_pred != pos_label) & (y_true == pos_label)))
    tn = int(np.sum((y_pred != pos_label) & (y_true != pos_label)))

    accuracy  = (tp + tn) / len(y_true) if len(y_true) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ ДЕРЕВА
# ============================================================

def plot_tree(node, feature_names=None, depth=0, max_depth=5,
              x=0.5, y=1.0, dx=0.25, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis('off')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)

    if node is None or depth > max_depth:
        return

    if node.is_leaf:
        label = f"leaf\nclass={node.value}"
        color = 'lightyellow'
    else:
        fname = feature_names[node.feature] if feature_names else f"F{node.feature}"
        label = f"{fname} <= {node.threshold:.2f}"
        color = 'lightblue'

    ax.text(x, y, label, ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle='round', fc=color, ec='gray'))

    if not node.is_leaf:
        ax.plot([x, x - dx], [y - 0.03, y - 0.10], 'k-', lw=0.8)
        ax.plot([x, x + dx], [y - 0.03, y - 0.10], 'k-', lw=0.8)
        plot_tree(node.left,  feature_names, depth + 1, max_depth,
                  x - dx, y - 0.13, dx / 2, ax)
        plot_tree(node.right, feature_names, depth + 1, max_depth,
                  x + dx, y - 0.13, dx / 2, ax)

    if depth == 0:
        plt.tight_layout()
        plt.savefig('tree.png', dpi=120, bbox_inches='tight')
        plt.show()


# ============================================================
# 8. РАЗДЕЛЕНИЕ ВЫБОРКИ (собственный train_test_split)
# ============================================================

def train_test_split_custom(X, y, train_ratio, random_state=42):
    rng = random.Random(random_state)
    n = len(y)
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = int(n * train_ratio)
    tr_idx = indices[:n_train]
    te_idx = indices[n_train:]
    return X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]


# ============================================================
# 9. ЭКСПЕРИМЕНТЫ
# ============================================================

def run_single(criterion, X_train, y_train, X_test, y_test, feature_names):
    print(f"\n=== Единичный эксперимент: criterion={criterion} ===")
    tree = DecisionTree(criterion=criterion, max_depth=12,
                        min_samples_split=20, min_samples_leaf=10,
                        n_thresholds=20)
    print("Построение дерева...")
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    m = compute_metrics(y_test, y_pred)
    print(f"accuracy={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
          f"recall={m['recall']:.4f}  f1={m['f1']:.4f}")
    plot_tree(tree.root, feature_names=feature_names, max_depth=4)
    return m, tree


def run_proportions(criterion, X, y, feature_names, ratios=(0.6, 0.7, 0.8, 0.9)):
    print(f"\n=== Эксперимент с долями: criterion={criterion} ===")
    results = []
    for r in ratios:
        X_tr, y_tr, X_te, y_te = train_test_split_custom(X, y, r, random_state=42)
        tree = DecisionTree(criterion=criterion, max_depth=12,
                            min_samples_split=20, min_samples_leaf=10,
                            n_thresholds=20)
        tree.fit(X_tr, y_tr)
        y_pred = tree.predict(X_te)
        m = compute_metrics(y_te, y_pred)
        m['train_ratio'] = r
        results.append(m)
        print(f"  ratio={int(r*100)}:{int((1-r)*100)}  "
              f"acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
              f"rec={m['recall']:.4f}  f1={m['f1']:.4f}")
    return pd.DataFrame(results)


def plot_metrics(df, criterion):
    plt.figure(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        plt.plot(df['train_ratio'] * 100, df[metric], marker='o', label=metric)
    plt.xlabel('Доля обучающей выборки, %')
    plt.ylabel('Значение метрики')
    plt.title(f'Качество классификации ({criterion})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'metrics_{criterion}.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 10. ТОЧКА ВХОДА
# ============================================================

def main():
    print("Загрузка данных...")
    df_train_raw, df_test_raw = load_raw()
    X_train, y_train, X_test, y_test, feature_names = preprocess(df_train_raw, df_test_raw)
    print(f"Обучающая выборка: {X_train.shape}, тестовая: {X_test.shape}")

    # --- Единичный эксперимент на 100% обучающих данных ---
    for criterion in ['entropy', 'gini', 'gain_ratio']:
        run_single(criterion, X_train, y_train, X_test, y_test, feature_names)

    # --- Эксперимент с долями ---
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    for criterion in ['entropy', 'gini', 'gain_ratio']:
        df = run_proportions(criterion, X_all, y_all, feature_names)
        plot_metrics(df, criterion)


if __name__ == '__main__':
    main()