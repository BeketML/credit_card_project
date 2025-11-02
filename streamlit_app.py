import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Конфигурация страницы
st.set_page_config(
    page_title="Сегментация клиентов банка",
    page_icon="🏦",
    layout="wide"
)

# Заголовок приложения
st.title("🏦 Сегментация клиентов банка")
st.markdown("""
Это приложение использует модель машинного обучения для классификации клиентов банка на сегменты 
на основе их транзакционного поведения. Введите данные клиента ниже, чтобы определить его сегмент.
""")

# Кеширование загрузки моделей
@st.cache_resource
def load_models():
    """Загружает сохраненные модели"""
    models_path = Path('models')
    scaler = joblib.load(models_path / 'scaler.joblib')
    pca = joblib.load(models_path / 'pca_model.joblib')
    kmeans = joblib.load(models_path / 'kmeans_model.joblib')
    return scaler, pca, kmeans

@st.cache_data
def load_data_and_analyze_clusters():
    """Загружает данные и анализирует кластеры для создания описаний"""
    # Загрузка данных
    df = pd.read_csv('data/CC GENERAL.csv')
    
    # Предобработка (как в notebook)
    df = df.drop(['CUST_ID'], axis=1)
    df = df.dropna(subset=['CREDIT_LIMIT'])
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median())
    
    # Логарифмическое преобразование
    cols_log = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 
                'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY', 
                'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS', 
                'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
    
    df_log = df.copy()
    for col in cols_log:
        df_log[col] = np.log(1 + df_log[col])
    
    # Загрузка моделей и предсказание
    scaler, pca, kmeans = load_models()
    X_scaled = scaler.transform(df_log)
    X_red = pca.transform(X_scaled)
    clusters = kmeans.predict(X_red)
    
    # Добавление меток кластеров
    df['cluster_id'] = clusters
    
    # Анализ кластеров (используем исходные данные без логарифмирования)
    cluster_analysis = {}
    
    for cluster_id in range(2):
        cluster_data = df[df['cluster_id'] == cluster_id]
        cluster_means = cluster_data.mean()
        
        # Создание описания кластера
        description = create_cluster_description(cluster_id, cluster_means)
        
        cluster_analysis[cluster_id] = {
            'means': cluster_means,
            'description': description,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100
        }
    
    return cluster_analysis

def create_cluster_description(cluster_id, means):
    """Создает текстовое описание кластера на основе средних значений"""
    
    # Анализ характеристик
    high_balance = means['BALANCE'] > 1000
    high_purchases = means['PURCHASES'] > 1000
    high_frequency = means['PURCHASES_FREQUENCY'] > 0.5
    uses_cash_advance = means['CASH_ADVANCE'] > 500
    high_credit_limit = means['CREDIT_LIMIT'] > 5000
    pays_full = means['PRC_FULL_PAYMENT'] > 0.3
    
    # Определение типа клиента
    if cluster_id == 0:
        if high_purchases and high_frequency:
            if pays_full:
                return {
                    'name': 'Активные платежеспособные клиенты',
                    'description': '''
                    Этот сегмент характеризуется высоким уровнем активности и ответственным подходом к платежам.
                    
                    **Основные характеристики:**
                    - Высокая частота покупок ({:.2%})
                    - Большой объем покупок (${:.2f})
                    - Регулярные полные платежи ({:.2%})
                    - Активное использование кредитного лимита
                    
                    **Рекомендации:**
                    - Предлагать премиальные продукты и программы лояльности
                    - Рекомендовать кэшбэк-программы
                    - Рассматривать увеличение кредитного лимита
                    '''.format(
                        means['PURCHASES_FREQUENCY'],
                        means['PURCHASES'],
                        means['PRC_FULL_PAYMENT']
                    ),
                    'key_features': ['Высокая активность', 'Ответственные платежи', 'Высокий оборот']
                }
            else:
                return {
                    'name': 'Активные клиенты с рассрочкой',
                    'description': '''
                    Клиенты этого сегмента активно используют карту, но предпочитают оплачивать в рассрочку.
                    
                    **Основные характеристики:**
                    - Высокая частота покупок ({:.2%})
                    - Большой объем покупок (${:.2f})
                    - Использование рассрочки
                    - Средний уровень баланса (${:.2f})
                    
                    **Рекомендации:**
                    - Предлагать выгодные программы рассрочки
                    - Информировать о преимуществах полного погашения
                    - Программы накопления баллов
                    '''.format(
                        means['PURCHASES_FREQUENCY'],
                        means['PURCHASES'],
                        means['BALANCE']
                    ),
                    'key_features': ['Высокая активность', 'Использование рассрочки', 'Регулярные покупки']
                }
        else:
            return {
                'name': 'Умеренные пользователи',
                'description': '''
                Клиенты с умеренным использованием карты и средним уровнем активности.
                
                **Основные характеристики:**
                - Средняя частота покупок ({:.2%})
                - Умеренный объем покупок (${:.2f})
                - Стабильное использование кредитного лимита
                - Регулярные платежи
                
                **Рекомендации:**
                - Программы стимулирования активности
                - Предложения по увеличению использования
                - Информация о новых продуктах
                '''.format(
                    means['PURCHASES_FREQUENCY'],
                    means['PURCHASES']
                ),
                'key_features': ['Умеренная активность', 'Стабильное использование', 'Регулярные платежи']
            }
    else:  # cluster_id == 1
        if uses_cash_advance and not high_purchases:
            return {
                'name': 'Клиенты с денежными авансами',
                'description': '''
                Сегмент клиентов, активно использующих функцию получения наличных.
                
                **Основные характеристики:**
                - Высокое использование денежных авансов (${:.2f})
                - Низкая частота покупок ({:.2%})
                - Высокая частота получения наличных ({:.2%})
                - Средний/высокий кредитный лимит (${:.2f})
                
                **Рекомендации:**
                - Мониторинг рисков
                - Предложения по снижению использования наличных
                    - Образовательные материалы о преимуществах безналичных платежей
                '''.format(
                    means['CASH_ADVANCE'],
                    means['PURCHASES_FREQUENCY'],
                    means['CASH_ADVANCE_FREQUENCY'],
                    means['CREDIT_LIMIT']
                ),
                'key_features': ['Денежные авансы', 'Низкая активность покупок', 'Требует внимания']
            }
        elif not high_purchases and not high_frequency:
            return {
                'name': 'Малоактивные клиенты',
                'description': '''
                Клиенты с низкой активностью использования карты.
                
                **Основные характеристики:**
                - Низкая частота покупок ({:.2%})
                - Небольшой объем покупок (${:.2f})
                - Низкий баланс (${:.2f})
                - Минимальное использование кредитного лимита
                
                **Рекомендации:**
                - Программы активации клиентов
                - Специальные предложения для возврата активности
                - Анализ причин неиспользования карты
                - Рассмотрение закрытия неактивных счетов
                '''.format(
                    means['PURCHASES_FREQUENCY'],
                    means['PURCHASES'],
                    means['BALANCE']
                ),
                'key_features': ['Низкая активность', 'Минимальное использование', 'Требует активации']
            }
        else:
            return {
                'name': 'Клиенты с особым поведением',
                'description': '''
                Сегмент с уникальными паттернами использования карты.
                
                **Основные характеристики:**
                - Специфические модели использования
                - Уникальное сочетание параметров
                - Требует индивидуального подхода
                
                **Рекомендации:**
                - Персонализированный анализ
                - Индивидуальные предложения
                '''.format(),
                'key_features': ['Особое поведение', 'Требует анализа']
            }

# Загрузка моделей
try:
    scaler, pca, kmeans = load_models()
    cluster_analysis = load_data_and_analyze_clusters()
except Exception as e:
    st.error(f"Ошибка при загрузке моделей: {e}")
    st.stop()

# Боковая панель с вводом данных
st.sidebar.header("📊 Ввод данных клиента")

# Группировка полей по категориям
st.sidebar.subheader("💰 Баланс и кредитный лимит")
balance = st.sidebar.number_input("BALANCE (Баланс)", min_value=0.0, value=1000.0, step=100.0)
balance_frequency = st.sidebar.slider("BALANCE_FREQUENCY (Частота обновления баланса)", 0.0, 1.0, 0.5, 0.01)
credit_limit = st.sidebar.number_input("CREDIT_LIMIT (Кредитный лимит)", min_value=0.0, value=3000.0, step=500.0)

st.sidebar.subheader("🛒 Покупки")
purchases = st.sidebar.number_input("PURCHASES (Общая сумма покупок)", min_value=0.0, value=500.0, step=50.0)
oneoff_purchases = st.sidebar.number_input("ONEOFF_PURCHASES (Разовая покупка)", min_value=0.0, value=100.0, step=50.0)
installments_purchases = st.sidebar.number_input("INSTALLMENTS_PURCHASES (Покупки в рассрочку)", min_value=0.0, value=200.0, step=50.0)
purchases_frequency = st.sidebar.slider("PURCHASES_FREQUENCY (Частота покупок)", 0.0, 1.0, 0.5, 0.01)
oneoff_purchases_frequency = st.sidebar.slider("ONEOFF_PURCHASES_FREQUENCY (Частота разовых покупок)", 0.0, 1.0, 0.3, 0.01)
purchases_installments_frequency = st.sidebar.slider("PURCHASES_INSTALLMENTS_FREQUENCY (Частота покупок в рассрочку)", 0.0, 1.0, 0.3, 0.01)
purchases_trx = st.sidebar.number_input("PURCHASES_TRX (Количество транзакций покупок)", min_value=0, value=10, step=1)

st.sidebar.subheader("💵 Денежные авансы")
cash_advance = st.sidebar.number_input("CASH_ADVANCE (Сумма денежных авансов)", min_value=0.0, value=0.0, step=100.0)
cash_advance_frequency = st.sidebar.slider("CASH_ADVANCE_FREQUENCY (Частота получения наличных)", 0.0, 1.0, 0.0, 0.01)
cash_advance_trx = st.sidebar.number_input("CASH_ADVANCE_TRX (Количество транзакций получения наличных)", min_value=0, value=0, step=1)

st.sidebar.subheader("💳 Платежи")
payments = st.sidebar.number_input("PAYMENTS (Сумма платежей)", min_value=0.0, value=500.0, step=50.0)
minimum_payments = st.sidebar.number_input("MINIMUM_PAYMENTS (Минимальные платежи)", min_value=0.0, value=100.0, step=50.0)
prc_full_payment = st.sidebar.slider("PRC_FULL_PAYMENT (Процент полных платежей)", 0.0, 1.0, 0.3, 0.01)

st.sidebar.subheader("📅 Прочее")
tenure = st.sidebar.number_input("TENURE (Срок пользования картой в месяцах)", min_value=0, value=12, step=1)

# Кнопка предсказания
if st.sidebar.button("🔍 Определить сегмент", type="primary"):
    # Создание DataFrame с введенными данными
    input_data = {
        'BALANCE': balance,
        'BALANCE_FREQUENCY': balance_frequency,
        'PURCHASES': purchases,
        'ONEOFF_PURCHASES': oneoff_purchases,
        'INSTALLMENTS_PURCHASES': installments_purchases,
        'CASH_ADVANCE': cash_advance,
        'PURCHASES_FREQUENCY': purchases_frequency,
        'ONEOFF_PURCHASES_FREQUENCY': oneoff_purchases_frequency,
        'PURCHASES_INSTALLMENTS_FREQUENCY': purchases_installments_frequency,
        'CASH_ADVANCE_FREQUENCY': cash_advance_frequency,
        'CASH_ADVANCE_TRX': cash_advance_trx,
        'PURCHASES_TRX': purchases_trx,
        'CREDIT_LIMIT': credit_limit,
        'PAYMENTS': payments,
        'MINIMUM_PAYMENTS': minimum_payments,
        'PRC_FULL_PAYMENT': prc_full_payment,
        'TENURE': tenure
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Применение логарифмического преобразования
    cols_log = ['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
                'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
                'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
                'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']
    
    df_processed = df_input.copy()
    for col in cols_log:
        df_processed[col] = np.log(1 + df_processed[col])
    
    # Применение pipeline: Scaler -> PCA -> KMeans
    X_scaled = scaler.transform(df_processed)
    X_red = pca.transform(X_scaled)
    cluster = kmeans.predict(X_red)[0]
    
    # Вывод результата
    st.success(f"✅ Клиент отнесен к сегменту: **Кластер {cluster}**")
    
    # Информация о кластере
    cluster_info = cluster_analysis[cluster]
    
    st.header(f"📋 Описание сегмента: {cluster_info['description']['name']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Размер сегмента", f"{cluster_info['size']:,} клиентов")
        st.metric("Доля в базе", f"{cluster_info['percentage']:.1f}%")
    
    with col2:
        st.markdown("**Ключевые особенности:**")
        for feature in cluster_info['description']['key_features']:
            st.markdown(f"- {feature}")
    
    st.markdown("### 📝 Детальное описание")
    st.markdown(cluster_info['description']['description'])
    
    # Показ средних значений для этого кластера
    st.markdown("### 📊 Средние значения характеристик сегмента")
    
    means_df = cluster_info['means'].drop('cluster_id')
    
    # Отображение ключевых метрик
    key_metrics = ['BALANCE', 'PURCHASES', 'PURCHASES_FREQUENCY', 'CREDIT_LIMIT', 
                  'PAYMENTS', 'CASH_ADVANCE', 'PRC_FULL_PAYMENT']
    
    metrics_cols = st.columns(len(key_metrics))
    for i, metric in enumerate(key_metrics):
        with metrics_cols[i]:
            st.metric(metric, f"${means_df[metric]:,.2f}" if 'BALANCE' in metric or 'PURCHASES' in metric or 'PAYMENTS' in metric or 'CREDIT_LIMIT' in metric or 'CASH_ADVANCE' in metric or 'MINIMUM_PAYMENTS' in metric else f"{means_df[metric]:.2%}" if 'FREQUENCY' in metric or 'PAYMENT' in metric else f"{means_df[metric]:.0f}")
    
    # Таблица всех характеристик
    with st.expander("📈 Все характеристики сегмента"):
        st.dataframe(means_df.to_frame('Среднее значение').style.format('{:.2f}'))

# Информация о всех сегментах
st.header("📊 Обзор всех сегментов")
st.markdown("Ниже представлена информация о всех сегментах клиентов:")

for cluster_id in range(2):
    with st.expander(f"Кластер {cluster_id}: {cluster_analysis[cluster_id]['description']['name']}"):
        st.markdown(cluster_analysis[cluster_id]['description']['description'])
        st.metric("Размер", f"{cluster_analysis[cluster_id]['size']:,} клиентов ({cluster_analysis[cluster_id]['percentage']:.1f}%)")

