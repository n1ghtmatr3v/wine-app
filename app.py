import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent
st.set_page_config(page_title='Подбор вина', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test_data():
    data = pd.read_csv(APP_DIR / 'test.csv')
    config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
    return data, config


@st.cache_data
def test_proba(_model, X):
    probabilities = _model.predict_proba(X)
    return probabilities[:, 1]


def prepare_uploaded_data(data, features):
    missing = []
    for feature in features:
        if feature not in data.columns:
            missing.append(feature)
    if missing:
        return None, 'В файле не хватает столбцов: ' + ', '.join(missing)
    if data.empty:
        return None, 'В файле нет строк с винами.'
    X = data[features].apply(pd.to_numeric, errors='coerce')
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        return None, 'Все признаки должны быть числами без пропусков и бесконечностей.'
    if not X['is_red'].isin([0, 1]).all():
        return None, 'is_red должен быть равен 0 (белое) или 1 (красное).'
    return X, None


model = load_model()
test_data, config = load_test_data()
X_test = test_data.drop(columns='target')
y_test = test_data['target']
features = X_test.columns.tolist()
p_test = test_proba(model, X_test)

st.title('🍷 Подбор хороших вин')
st.write('Хорошим считается вино с оценкой экспертов не ниже 7.')
threshold = st.sidebar.slider(
    'Порог отбора', 0.0, 1.0, value=float(config['threshold']),
    step=0.001, format='%.6f', key='threshold')
st.sidebar.caption('Вино отбирается, если его вероятность не меньше порога.')
tabs = st.tabs(['Оценка одного вина', 'Модельные метрики',
                'Пакетное предсказание', 'Что выучила модель'])

with tabs[0]:
    st.subheader('Характеристики вина')
    st.caption('Диапазоны взяты из тестовых данных, начальные значения — медианы.')
    values = {}
    columns = st.columns(3)
    for i in range(len(features)):
        feature = features[i]
        with columns[i % 3]:
            if feature == 'is_red':
                is_red = st.toggle('Красное вино (выключено — белое)',
                                   value=bool(X_test[feature].median()), key=feature)
                values[feature] = int(is_red)
            else:
                step = 0.01
                if feature == 'density':
                    step = 0.00001
                values[feature] = st.number_input(
                    feature.replace('_', ' '), min_value=float(X_test[feature].min()),
                    max_value=float(X_test[feature].max()),
                    value=float(X_test[feature].median()), step=step, format='%.5f', key=feature)
    one_wine = pd.DataFrame([values], columns=features)
    probability = model.predict_proba(one_wine)[0, 1]
    st.metric('Вероятность хорошего вина', format(probability, '.1%'))
    if probability >= threshold:
        st.success('Вино включено в подборку: класс 1.')
    else:
        st.info('Вино не включено в подборку: класс 0.')

with tabs[1]:
    predictions = p_test >= threshold
    predictions = predictions.astype(int)
    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)
    columns = st.columns(3)
    columns[0].metric('Precision', format(precision, '.1%'))
    columns[1].metric('Recall', format(recall, '.1%'))
    columns[2].metric('F1', format(f1, '.3f'))
    st.write('При повышении порога подборка становится меньше: хороших вин находится '
             'меньше, но precision обычно растёт. При снижении порога подборка '
             'расширяется, но в неё может попасть больше посредственных вин.')
    if predictions.sum() == 0:
        st.info('Подборка пустая. Precision и F1 показаны как 0.')

    confusion = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    matrix = metrics.ConfusionMatrixDisplay(confusion, display_labels=['Остальные', 'Хорошие'])
    matrix.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set(xlabel='Предсказанный класс', ylabel='Настоящий класс', title='Матрица ошибок')
    fig.tight_layout()
    st.pyplot(fig, width=500)
    plt.close(fig)

    fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, p_test)
    point_fpr = confusion[0, 1] / confusion[0].sum()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fpr, tpr, label='Модель')
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].scatter(point_fpr, recall, color='red', label='Текущий порог', zorder=3)
    axes[0].set(xlabel='FPR', ylabel='Recall / TPR', title='ROC-кривая')
    axes[1].step(pr_recall, pr_precision, where='post', label='Модель')
    axes[1].axhline(y_test.mean(), linestyle='--', color='gray')
    axes[1].scatter(recall, precision, color='red', label='Текущий порог', zorder=3)
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tabs[2]:
    st.subheader('Предсказания для CSV')
    st.write('Нужны 12 признаков без target. Разделитель — запятая. Порог общий для всех вкладок.')
    example = X_test.head(10).to_csv(index=False)
    st.download_button('Скачать пример CSV', example.encode('utf-8-sig'),
                       file_name='example_wines.csv', mime='text/csv')
    uploaded_file = st.file_uploader('CSV с характеристиками вин', type=['csv'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError):
            st.error('Не удалось прочитать CSV. Проверьте файл, кодировку UTF-8 и разделитель.')
        else:
            X, error = prepare_uploaded_data(data, features)
            if error is not None:
                st.error(error)
            else:
                probabilities = model.predict_proba(X)[:, 1]
                predictions = probabilities >= threshold
                predictions = predictions.astype(int)
                result = X.copy()
                result.insert(0, 'prediction', predictions)
                result.insert(0, 'probability_good', probabilities)
                st.dataframe(result, hide_index=True)
                result_csv = result.to_csv(index=False)
                st.download_button('Скачать предсказания', result_csv.encode('utf-8-sig'),
                                   file_name='wine_predictions.csv', mime='text/csv')

with tabs[3]:
    st.subheader('Коэффициенты модели')
    weights = model.named_steps['logisticregression'].coef_[0]
    coefficients = pd.Series(weights, index=features).sort_values()
    colors = []
    for value in coefficients:
        if value >= 0:
            colors.append('seagreen')
        else:
            colors.append('darkorange')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coefficients.index, coefficients.values, color=colors)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после StandardScaler')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Больше всего повышает оценку качества:', coefficients.idxmax())
    st.write('Больше всего понижает оценку качества:', coefficients.idxmin())
    st.caption('Положительный коэффициент повышает оценку хорошего качества, '
               'отрицательный — понижает, если остальные признаки не меняются.')
