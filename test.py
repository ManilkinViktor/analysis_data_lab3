"""
Запуск уже сохранённой модели бэггинга.
Использование:
    python predict_saved.py                 # предсказание на первых 10 из adult.data
    python predict_saved.py adult.test      # предсказание на первых 10 из указанного файла
"""
import sys
import pandas as pd
import joblib

MODEL_PATH = 'bagging_results/best_bagging_model.joblib'

col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
             'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

label_map = {0: '<=50K', 1: '>50K'}


def load_data(path, has_header_skip=False):
    """Читает CSV в формате Adult. Если в файле есть строка-заголовок (adult.test), пропускаем её."""
    df = pd.read_csv(path, header=None, names=col_names,
                     na_values=' ?', skipinitialspace=True,
                     skiprows=1 if has_header_skip else 0)
    # В adult.test метки имеют вид ">50K." — убираем точку
    df['income'] = df['income'].astype(str).str.strip().str.rstrip('.')
    df = df.dropna()
    return df


def main():
    # 1. Загрузка модели
    artifact = joblib.load(MODEL_PATH)
    model = artifact['model']
    print(f"Модель загружена: n_estimators={artifact['n_estimators']}, "
          f"test_accuracy={artifact['test_accuracy']:.4f}")

    # 2. Выбор файла данных
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'adult.data'
    has_header = data_path.endswith('adult.test')
    df = load_data(data_path, has_header_skip=has_header)
    print(f"Данные загружены из {data_path}: {len(df)} строк")

    # 3. Первые 10 объектов
    df_sample = df.iloc[:10].copy()
    X_sample = df_sample.drop('income', axis=1)
    y_true = df_sample['income'].map({'<=50K': 0, '>50K': 1}).values

    # 4. Предсказание
    y_pred = model.predict(X_sample)
    y_proba = model.predict_proba(X_sample)

    # 5. Таблица результатов
    result = pd.DataFrame({
        'sample_idx': range(len(df_sample)),
        'true_label': [label_map[v] for v in y_true],
        'predicted': [label_map[v] for v in y_pred],
        'proba_<=50K': y_proba[:, 0].round(4),
        'proba_>50K': y_proba[:, 1].round(4),
        'correct': (y_pred == y_true).astype(int),
    })

    pd.set_option('display.width', 200)
    print("\n=== Предсказание на первых 10 объектах ===")
    print(result.to_string(index=False))

    acc = result['correct'].mean()
    print(f"\nТочность на 10 объектах: {acc:.2f} "
          f"({result['correct'].sum()}/{len(result)})")

    # 6. Сохранение результатов
    out_path = 'saved_model_predictions.csv'
    result.to_csv(out_path, index=False)
    print(f"Результаты сохранены в {out_path}")


if __name__ == '__main__':
    main()