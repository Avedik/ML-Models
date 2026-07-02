import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Импорты sklearn и statsmodels
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- НАСТРОЙКА СИСТЕМЫ (UI) ---
st.set_page_config(page_title="Data Science Portfolio", layout="wide")
st.title("📊 Практические задачи машинного обучения")
st.markdown("Интерактивное портфолио с реализацией классификации, ансамблевой регрессии и прогнозирования временных рядов.")

# Навигация в боковой панели
page = st.sidebar.radio("Выберите модуль:", ["1. Классификация", "2. Регрессия", "3. Временные ряды"])

# --- ОПТИМИЗАЦИЯ СИСТЕМЫ: КЭШИРОВАНИЕ ДАННЫХ ---
# Кэшируем генерацию данных, чтобы они не пересоздавались при каждом движении слайдера
@st.cache_data
def get_classification_data():
    X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=3.8, center_box=(-6.0, 6.0), random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)

@st.cache_data
def get_regression_data():
    n_samples, n_features = 1000, 10
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    coefficients = np.random.randn(n_features)
    y = X.dot(coefficients) + np.random.randn(n_samples)
    return train_test_split(X, y, test_size=0.2, random_state=42), X

@st.cache_data
def get_timeseries_data():
    np.random.seed(42)
    time = np.arange(1000)
    y_ts = np.sin(0.1 * time) + (0.05 * time) + np.random.normal(0, 1, 1000)
    train_size = int(len(y_ts) * 0.8)
    return time, y_ts[:train_size], y_ts[train_size:]


# ==========================================
# МОДУЛЬ 1: КЛАССИФИКАЦИЯ
# ==========================================
if page == "1. Классификация":
    st.header("Задача 1: Бинарная классификация и поиск гиперпараметров")
    
    X_train, X_test, y_train, y_test = get_classification_data()
    
    # Интерактивный UI для управления гиперпараметрами
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Настройки модели")
        model_choice = st.selectbox("Алгоритм", ["Random Forest", "Logistic Regression"])
        
        if model_choice == "Random Forest":
            n_est = st.slider("Количество деревьев (n_estimators)", 10, 200, 50, step=10)
            model = RandomForestClassifier(n_estimators=n_est, random_state=42)
        else:
            c_val = st.select_slider("Параметр регуляризации (C)", options=[0.01, 0.1, 1, 10, 100], value=1)
            model = LogisticRegression(C=c_val, solver='liblinear', random_state=42)

with col2:
    # Обучение модели
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    gini = 2 * auc - 1

    st.metric("Коэффициент Джини (Gini)", f"{gini:.4f}")

    # -------------------------------
    # Построение границы решений
    # -------------------------------
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = pipeline.predict(grid)
    zz = zz.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Фон — области классификации
    ax.contourf(
        xx,
        yy,
        zz,
        alpha=0.30,
        cmap="coolwarm"
    )

    # Точки тестовой выборки
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        edgecolors="black",
        s=45
    )

    ax.set_title(
        f"{model_choice} (n_estimators={n_est})"
        if model_choice == "Random Forest"
        else f"{model_choice} (C={c_val})"
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    st.pyplot(fig)

# ==========================================
# МОДУЛЬ 2: РЕГРЕССИЯ
# ==========================================
elif page == "2. Регрессия":
    st.header("Задача 2: Ансамблевая регрессия и системный анализ остатков")
    
    (X_train, X_test, y_train, y_test), X_full = get_regression_data()
    
    bagging_model = BaggingRegressor(estimator=LinearRegression(), n_estimators=50, random_state=42)
    bagging_model.fit(X_train, y_train)
    y_pred = bagging_model.predict(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="R-squared (R²)", value=f"{r2_score(y_test, y_pred):.4f}")
    with col2:
        st.metric(label="Mean Squared Error (MSE)", value=f"{mean_squared_error(y_test, y_pred):.4f}")
    
    st.subheader("Диагностика модели: Фактические vs Предсказанные значения")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    ax.set_xlabel("Фактические значения")
    ax.set_ylabel("Предсказанные значения")
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    st.subheader("Проверка на мультиколлинеарность (VIF)")
    if st.checkbox("Показать таблицу VIF"):
        X_df = pd.DataFrame(X_full, columns=[f"Feature_{i+1}" for i in range(X_full.shape[1])])
        vif_data = pd.DataFrame({
            "Признак": X_df.columns,
            "VIF": [variance_inflation_factor(X_df.values, i) for i in range(X_full.shape[1])]
        })
        st.dataframe(vif_data.T)

# ==========================================
# МОДУЛЬ 3: ВРЕМЕННЫЕ РЯДЫ
# ==========================================
elif page == "3. Временные ряды":
    st.header("Задача 3: Анализ и прогнозирование сигнала (Тренд + Цикл + Шум)")
    
    time, train, test = get_timeseries_data()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Настройки прогноза")
        lags = st.number_input("Количество лагов (AR модель)", min_value=1, max_value=50, value=10)
    
    with col2:
        # AR
        ar_model = AutoReg(train, lags=lags).fit()
        ar_pred = ar_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
        
        # ETS
        ets_model = ExponentialSmoothing(train, trend="add", seasonal=None).fit()
        ets_pred = ets_model.forecast(steps=len(test))
        
        st.write(f"**MAE (AR):** {mean_absolute_error(test, ar_pred):.4f} | **MAE (ETS):** {mean_absolute_error(test, ets_pred):.4f}")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time[len(train):], test, label="Факт (Тест)", color="black", linewidth=2)
        ax.plot(time[len(train):], ar_pred, label=f"Прогноз AR (lags={lags})", color="red", linestyle="--")
        ax.plot(time[len(train):], ets_pred, label="Прогноз ETS", color="blue", linestyle="-.")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
