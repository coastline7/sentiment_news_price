import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import nltk

# Загрузка компонентов NLTK
nltk.download('vader_lexicon')

# Импортируем модуль streamlit как st
import streamlit as st
# Импортируем модуль cli из web под именем stcli
from streamlit.web import cli as stcli
# Импортируем модуль sys
import sys
# Из модуля streamlit импортируем модуль runtime
from streamlit import runtime

# Проверяем существование runtime
runtime.exists()

# Определяем функцию main()
def main():
    # Добавляем информацию в левый верхний угол
    st.markdown(
        """
        <div style="position: fixed; top: 0; left: 0; background-color: rgba(255,255,255,0.9); 
                    padding: 10px; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); z-index: 1000;">
            <h4 style="margin: 0;">Выпускная квалификационная работа</h4>
            <p style="margin: 0;">Web-приложения для анализа валютного рынка</p>
            <p style="margin: 0;">УБВТ2102 Оганнисян А.С.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Основной заголовок приложения
    st.title('Рекомендации по акциям')

    # Словарь сопоставления названий компаний и их тикеров
    tickers_alphabet = {
        'Apple': 'AAPL',
        'AMD': 'AMD',
        'Amazon': 'AMZN',
        'Google': 'GOOGL',
        'Intel': 'INTC',
        'Microsoft': 'MSFT',
        'Netflix': 'NFLX',
        'NVIDIA': 'NVDA',
        'Tesla': 'TSLA'
    }

    # Словарь для кодирования тикеров в целочисленные значения
    company_to_company_encoded = {
        'Apple': 0,
        'AMD': 1,
        'Amazon': 2,
        'Google': 3,
        'Intel': 4,
        'Microsoft': 5,
        'Netflix': 6,
        'NVIDIA': 7,
        'Tesla': 8
    }

    # Функция для получения цены закрытия
    def get_closing_price(ticker, date):
        end_date = date + pd.Timedelta(days=1)
        start_date = date - pd.Timedelta(days=7)
        try:
            stock_data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')
            if not stock_data.empty:
                return stock_data['Close'].iloc[-1]
            return None
        except Exception as e:
            st.error(f"Error downloading stock data for {ticker}: {e}")
            return None

    # Функция для получения новостей
    def fetch_news(company_name, date):
        date_str = date.strftime('%Y-%m-%d')
        end_date_str = (date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        api_key = '18ee4bbff68345b691ddf8223bfc0269'
        url = f'https://newsapi.org/v2/everything?q={company_name}&from={date_str}&to={end_date_str}&sortBy=relevancy&apiKey={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return [article['description'] for article in articles if article['description']]
        else:
            st.error(f"Failed to fetch news: {response.text}")
            return []

    # Функция анализа сентимента новостей
    def analyze_sentiment(news):
        sid = SentimentIntensityAnalyzer()
        if not news:
            return 1
        scores = [sid.polarity_scores(article)['compound'] for article in news]
        average_score = np.mean(scores)
        if average_score > 0.05:
            return 2
        elif average_score < -0.05:
            return 0
        return 1

    # Функция получения торговой рекомендации
    def get_decision(date, company_name):
        ticker = tickers_alphabet[company_name]
        news = fetch_news(company_name, date)
        sentiment = analyze_sentiment(news)
        closing_price = get_closing_price(ticker, date)
        if closing_price is None:
            return "Не удалось получить данные о цене закрытия."
        loaded_model = keras.models.load_model('optimal_model.h5')
        code = company_to_company_encoded[company_name]
        year = date.year
        month = date.month
        day_of_week = date.weekday()
        hour = date.hour
        minute = date.minute
        day_hour = hour + day_of_week * 24
        input_data = pd.DataFrame({
            'Price': [closing_price],
            'Year': [year],
            'Month': [month],
            'DayOfWeek': [day_of_week],
            'Hour': [hour],
            'Minute': [minute],
            'DayHour': [day_hour],
            'Company_encoded': [code],
            'Sentiment_encoded': [sentiment]
        })
        # Преобразуем DataFrame в numpy-массив с нужным типом
        input_data = np.array(input_data, dtype=np.float32)
        # Добавляем новое измерение, чтобы получить форму (1, 9, 1)
        input_data = np.expand_dims(input_data, axis=2)
        prediction = loaded_model.predict(input_data)
        decision_classes = ['Купить', 'Держать', 'Продать']
        return decision_classes[np.argmax(prediction)]

    # Элементы интерфейса Streamlit
    company = st.selectbox('Выберите компанию:', list(tickers_alphabet.keys()))
    date = st.date_input('Выберите дату:', min_value=datetime.date.today() - datetime.timedelta(days=365), max_value=datetime.date.today())

    if st.button('Получить рекомендацию'):
        decision = get_decision(datetime.datetime.combine(date, datetime.datetime.min.time()), company)
        st.write(f"Рекомендация для {company} на {date}: {decision}")

# Если скрипт запускается напрямую
if __name__ == '__main__':
    # Если runtime существует
    if runtime.exists():
        # Вызываем функцию main()
        main()
    # Если runtime не существует
    else:
        # Устанавливаем аргументы командной строки
        sys.argv = ["streamlit", "run", sys.argv[0]]
        # Выходим из программы с помощью функции main() из модуля stcli
        sys.exit(stcli.main())
