import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from pandas.plotting import register_matplotlib_converters
import seaborn as sns

# Inicializando configurações para plotagem de gráficos
register_matplotlib_converters()

# Carregando os dados do IPEA
@st.cache_data
def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/lightscream/bases_pos_tech/master/fase_4/ipea_database.csv', decimal=',', sep=';')
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df.set_index('data', inplace=True)
    df.sort_index(inplace=True)
    return df

df = load_data()
df_series = df.asfreq('D').fillna(method='ffill')

# Definindo funções para calcular KPIs
def calculate_kpis(data):
    media = data.mean()
    mediana = data.median()
    desvio_padrao = data.std()
    maximo = data.max()
    minimo = data.min()
    return media, mediana, desvio_padrao, maximo, minimo

# Widgets para seleção de período
st.sidebar.subheader('Selecione o Período')
start_date = st.sidebar.date_input('Data Inicial', df_series.index.min())
end_date = st.sidebar.date_input('Data Final', df_series.index.max())

# Convertendo as datas para datetime64[ns]
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filtrando os dados históricos de acordo com o período selecionado
filtered_data = df_series.loc[start_date:end_date]

# Calculando KPIs dos dados históricos
media, mediana, desvio_padrao, maximo, minimo = calculate_kpis(filtered_data['values'])

# Exibindo KPIs dos dados históricos
st.sidebar.subheader('KPIs - Dados Históricos (em USD)')
st.sidebar.write("Média:", media)
st.sidebar.write("Mediana:", mediana)
st.sidebar.write("Desvio Padrão:", desvio_padrao)
st.sidebar.write("Máximo:", maximo)
st.sidebar.write("Mínimo:", minimo)
st.sidebar.subheader('Insights:')
st.sidebar.write("Março 2020 - Início da pandemia mundial - COVID")
st.sidebar.write("Outubro 2022 - OPEP diminui a produção de barril do petróleo para conter a queda dos preços e recuperar após baixa ocorrida pela pandemia do COVID.")
st.sidebar.write("Junho 2023 - Rússia inicia conflito no Leste Europeu.")
st.sidebar.write("Setembro 2023 - Cresce a demanda na China para petróleo.")
st.sidebar.subheader('Referências:')
st.sidebar.write("https://www.worldbank.org/en/research/commodity-markets")

# Título do Dashboard
st.header('Dashboard de Variação do preço do Petróleo')
# Visualização dos dados históricos
st.subheader('Visualização dos Dados Históricos')
# Gráfico histograma e boxplot
fig, axs = plt.subplots(1, 2, figsize=(20, 6))
histogram = sns.histplot(data=filtered_data, x='values', ax=axs[0])
histogram.set_title('Histograma da Frequência de preços (em USD)')
histogram.set_xlabel('Preço')
histogram.set_ylabel('Frequência')

boxplot = sns.boxplot(data=filtered_data, y='values', ax=axs[1])
boxplot.set_title('média de preços (em USD)')
boxplot.set_xlabel('')
boxplot.set_ylabel('Preço')

plt.tight_layout()
st.pyplot(fig)

# Gráfico de linha
fig0, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(data=filtered_data, x=filtered_data.index, y='values', ax=ax).set_title('Preço ao Longo do Tempo (em USD)')
plt.xlabel('Período', fontsize=14)  # Adiciona título ao eixo x
plt.ylabel('Preço', fontsize=14)  # Adiciona título ao eixo y
plt.tight_layout()
st.pyplot(fig0)


# Preparação dos dados para o modelo Prophet
df_prophet = filtered_data.reset_index().rename(columns={'data': 'ds', 'values': 'y'})
df_prophet['y'].fillna(method='ffill', inplace=True)

# Criação e ajuste do modelo Prophet
modelo_prophet = Prophet()
modelo_prophet.fit(df_prophet)

# Previsões futuras com o modelo Prophet
data_final = pd.to_datetime("2024-05-31")
dias_ate_final = (data_final - df_prophet['ds'].max()).days
futuro_prophet = modelo_prophet.make_future_dataframe(periods=dias_ate_final)
previsao_prophet = modelo_prophet.predict(futuro_prophet)

# Filtrando as previsões do Prophet de acordo com o período selecionado
filtered_predictions = previsao_prophet[(previsao_prophet['ds'] >= start_date) & (previsao_prophet['ds'] <= end_date)]

# Gráfico da previsão do modelo Prophet
st.subheader('Previsão do Preço - Prophet')
fig1 = modelo_prophet.plot(filtered_predictions, plot_cap=True, xlabel='Período', ylabel='Preço')
plt.title('Previsão do Preço - Prophet (Selecionado)')
plt.legend(['Preço histórico', 'Tendência', 'Predição'])
st.pyplot(fig1)