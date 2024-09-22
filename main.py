import streamlit as st
from datetime import date


import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

#time line of stocks
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#create the title for the web app
st.title("Stock Prediction")

#allow the user to choose a stock to predict from the list of stocks below
stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select a stock for prediction: ", stocks)

#add a slider to allow user to look n number of years into the future
n_years = st.slider("Years of prediction", 1, 10)
period = n_years*365

#load stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY) #pandas df
    data.reset_index(inplace = True)
    return data

#load data using selected stock
data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
data_load_state.text("Load Data... DONE!")

#show tail of data
st.subheader('Raw Data')
st.write(data.tail())

#create a plot
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting using prophet
df_train = data[['Date', 'Close']] #grab training data
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'}) #reformat data into dictionary head for prophet to use

m = Prophet() #create model
m.fit(df_train)
future = m.make_future_dataframe(periods=period) #in days
forecast = m.predict(future)

#Forecast data
st.subheader('Forecast Data')
st.write(forecast.tail())

#plot forecast data
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)