from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

HERE = Path(__file__).resolve().parent

st.set_page_config(page_title='Подбор хороших вин', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(HERE / 'model.joblib')


@st.cache_data
def load_test_data():
    return pd.read_csv(HERE / 'test.csv')


@st.cache_data
def get_test_probabilities(_model, features):
    return _model.predict_proba(features)[:, 1]


def prf(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(int)
    return {
        'precision': metrics.precision_score(y_true, predictions, zero_division=0),
        'recall': metrics.recall_score(y_true, predictions, zero_division=0),
        'f1': metrics.f1_score(y_true, predictions, zero_division=0),
    }


def model_features(model, table):
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    if hasattr(model[0], 'feature_names_in_'):
        return list(model[0].feature_names_in_)
    return [column for column in table.columns if column != 'target']


def validate_features(table, features):
    if table.empty:
        raise ValueError('CSV не содержит строк с данными')
    missing = sorted(set(features).difference(table.columns))
    if missing:
        raise ValueError('Отсутствуют столбцы: ' + ', '.join(missing))
    try:
        raw_X = table[features].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('Все значения признаков должны быть числами') from exc
    if not np.isfinite(raw_X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения')
    if 'is_red' in features and not raw_X['is_red'].isin([0, 1]).all():
        raise ValueError('Столбец is_red должен содержать только 0 (белое) или 1 (красное)')
    return raw_X


try:
    model = load_model()
    test_data = load_test_data()
    settings = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error('Файлы модели ещё не подготовлены. Сначала выполните ноутбук HW3_MMDA_solved.ipynb, '
             'чтобы получить model.joblib, config.json и test.csv в папке wine_app/.')
    st.exception(exc)
    st.stop()

features = model_features(model, test_data)
if 'target' not in test_data.columns or not set(features).issubset(test_data.columns):
    st.error('Неподходящая структура test.csv.')
    st.stop()
X_test = test_data[features]
y_test = test_data['target'].to_numpy(dtype=int)
scores = get_test_probabilities(model, X_test)

st.title('🍷 Подбор вин по качеству')
st.write('Модель оценивает вероятность того, что экспертная оценка вина будет не ниже 7 из 10.')
default_threshold = float(settings['threshold'])
threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0,
                              value=default_threshold, step=0.001,
                              format='%.3f', key='threshold')
st.sidebar.caption('Порог подобран по валидационной части обучающей выборки, а не по тесту.')

one_tab, metrics_tab, batch_tab, coefficients_tab = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетный прогноз', 'Коэффициенты модели'
])

with one_tab:
    st.subheader('Параметры вина')
    columns = st.columns(3)
    record = {}
    for i, name in enumerate(features):
        if name == 'is_red':
            record[name] = int(columns[i % 3].toggle('Красное вино', value=False))
        else:
            low = float(X_test[name].min())
            high = float(X_test[name].max())
            median = float(X_test[name].median())
            record[name] = columns[i % 3].number_input(
                name.replace('_', ' ').capitalize(),
                min_value=low,
                max_value=high,
                value=median,
                step=max((high - low) / 1000, 0.0001),
                format='%.5f',
            )
    wine = pd.DataFrame([record], columns=features)
    probability = float(model.predict_proba(wine)[0, 1])
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    if probability >= threshold:
        st.success('Рекомендовать это вино')
    else:
        st.warning('Не включать в подборку')

with metrics_tab:
    st.subheader('Качество на отложенной тестовой выборке')
    current = prf(y_test, scores, threshold)
    cols = st.columns(3)
    for col, key in zip(cols, ('precision', 'recall', 'f1')):
        col.metric(key.upper(), f'{current[key]:.3f}')

    pred = (scores >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=[0, 1], cmap='Blues', colorbar=False, ax=ax
    )
    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, scores)
    precision_values, recall_values, _ = metrics.precision_recall_curve(y_test, scores)
    fpr_here = float(np.mean((scores >= threshold) & (y_test == 0)) / max(np.mean(y_test == 0), 1e-12))
    tpr_here = current['recall']
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, scores):.3f}')
    ax_roc.scatter([fpr_here], [tpr_here], color='tomato', zorder=3, label=f't = {threshold:.2f}')
    ax_roc.plot([0, 1], [0, 1], '--', color='gray')
    ax_roc.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC-кривая')
    ax_roc.legend()

    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall_values, precision_values,
               label=f'AP = {metrics.average_precision_score(y_test, scores):.3f}')
    ax_pr.scatter([current['recall']], [current['precision']], color='tomato', zorder=3,
                  label=f't = {threshold:.2f}')
    ax_pr.axhline(y_test.mean(), linestyle='--', color='gray', label='Случайная модель')
    ax_pr.set(xlabel='Recall', ylabel='Precision', title='PR-кривая', xlim=(0, 1), ylim=(0, 1.05))
    ax_pr.legend()
    fig_col1, fig_col2 = st.columns(2)
    fig_col1.pyplot(fig_roc)
    fig_col2.pyplot(fig_pr)
    plt.close(fig_roc)
    plt.close(fig_pr)
    st.info('При увеличении порога в подборку попадёт меньше вин: обычно точность растёт, '
            'а полнота падает. При уменьшении порога вина рекомендуются чаще, '
            'но возрастает риск добавить посредственные.')

with batch_tab:
    st.subheader('Загрузить CSV с характеристиками вин')
    st.caption('Требуются те же 12 столбцов, что у обучающей модели; столбец target не нужен.')
    st.download_button('Скачать образец CSV',
                       data=X_test.head(5).to_csv(index=False).encode('utf-8-sig'),
                       file_name='wine_example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV-файл', type='csv')
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f'Не получилось прочитать CSV: {exc}')
        else:
            try:
                raw_X = validate_features(uploaded_df, features)
                proba = model.predict_proba(raw_X)[:, 1]
                result = uploaded_df.copy()
                result['probability_good'] = proba
                result['recommend'] = (proba >= threshold).astype(int)
                st.dataframe(result, use_container_width=True)
                st.download_button('Скачать прогнозы CSV',
                                   data=result.to_csv(index=False).encode('utf-8-sig'),
                                   file_name='wine_predictions.csv', mime='text/csv')
            except (ValueError, TypeError) as exc:
                st.error(f'Проверьте CSV: {exc}')

with coefficients_tab:
    st.subheader('Что выучила модель')
    if not hasattr(model, 'named_steps'):
        st.info('Для графика коэффициентов требуется sklearn-пайплайн.')
    else:
        estimator = model.steps[-1][1]
        if not hasattr(estimator, 'coef_'):
            st.info('У выбранной модели нет линейных коэффициентов.')
        else:
            coef = pd.Series(estimator.coef_[0], index=features).sort_values()
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#cc6677' if value < 0 else '#4477aa' for value in coef]
            coef.plot(kind='barh', ax=ax, color=colors)
            ax.axvline(0, color='black', linewidth=1)
            ax.set(xlabel='Вес после стандартизации', ylabel='', title='Коэффициенты логистической регрессии')
            st.pyplot(fig)
            plt.close(fig)
            st.write('Положительные коэффициенты повышают логит вероятности хорошего вина, '
                     'отрицательные — уменьшают его (при прочих равных).')
            st.write('Сильнее всего за хорошее вино:', ', '.join(coef.tail(3).index[::-1]))
            st.write('Сильнее всего против:', ', '.join(coef.head(3).index))

# Кэширование: ресурс (@st.cache_resource) хранит один объект модели между
# перезапусками скрипта; @st.cache_data хранит результаты чтения таблицы и
# вычисления вероятностей. Без кэша joblib.load читался бы с диска при каждом
# движении слайдера, а predict_proba заново обрабатывал бы весь тестовый набор.
