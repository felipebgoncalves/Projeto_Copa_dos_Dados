import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


def page1():
    st.sidebar.markdown("## Predição dos Jogos da Copa")

    image = Image.open("logo.png")

    st.image(image)

    st.title("Copa dos Dados")
    st.text("Algoritmo de Machine Learning capaz de prever o ganhador da copa do mundo de 2022")

    df_selecoes = pd.read_csv('Selecoes2022.csv')
    df_selecoes.sort_values(by='Selecoes', ascending=True)

    todas_selecoes = sorted(df_selecoes['Selecoes'].unique())
    selecao_da_casa = st.selectbox('Primeira Seleção (Seleção da Casa)', todas_selecoes)

    selecao_visitante = df_selecoes[df_selecoes['Selecoes'] != selecao_da_casa]
    selecionar_selecao_visitante = st.selectbox('Segunda Seleção (Seleção Visitante)', selecao_visitante)

    svm_model = joblib.load('model.pkl')

    nome_time = {'France': 0, 'Mexico': 1, 'USA': 2, 'Belgium': 3, 'Yugoslavia': 4, 'Brazil': 5, 'Romania': 6,
                 'Peru': 7,
                 'Argentina': 8, 'Chile': 9, 'Bolivia': 10, 'Paraguay': 11, 'Uruguay': 12, 'Austria': 13, 'Hungary': 14,
                 'Egypt': 15, 'Switzerland': 16, 'Netherlands': 17, 'Sweden': 18, 'Germany': 19, 'Spain': 20,
                 'Italy': 21,
                 'Czechoslovakia': 22, 'Dutch East Indies': 23, 'Cuba': 24, 'Norway': 25, 'Poland': 26, 'England': 27,
                 'Scotland': 28, 'Turkey': 29, 'Korea Republic': 30, 'Soviet Union': 31, 'Wales': 32,
                 'Northern Ireland': 33, 'Colombia': 34, 'Bulgaria': 35, 'Korea DPR': 36, 'Portugal': 37, 'Israel': 38,
                 'Morocco': 39, 'El Salvador': 40, 'German DR': 41, 'Australia': 42, 'Zaire': 43, 'Haiti': 44,
                 'Tunisia': 45, 'Iran': 46, 'Cameroon': 47, 'New Zealand': 48, 'Algeria': 49, 'Honduras': 50,
                 'Kuwait': 51,
                 'Canada': 52, 'Iraq': 53, 'Denmark': 54, 'United Arab Emirates': 55, 'Costa Rica': 56,
                 'Republic of Ireland': 57, 'Saudi Arabia': 58, 'Russia': 59, 'Greece': 60, 'Nigeria': 61,
                 'South Africa': 62, 'Japan': 63, 'Jamaica': 64, 'Croatia': 65, 'Senegal': 66, 'Slovenia': 67,
                 'Ecuador': 68, 'China PR': 69, 'Trinidad and Tobago': 70, "Côte d'Ivoire": 71,
                 'Serbia and Montenegro': 72,
                 'Angola': 73, 'Czech Republic': 74, 'Ghana': 75, 'Togo': 76, 'Ukraine': 77, 'Serbia': 78,
                 'Slovakia': 79,
                 'Bosnia and Herzegovina': 80, 'Iceland': 81, 'Panama': 82}

    df_campeoes = pd.read_csv("Campeoes.csv")
    campeoes = df_campeoes['Vencedor'].value_counts()

    def predicao(timeA, timeB):
        idA = nome_time[timeA]
        idB = nome_time[timeB]

        campeaoA = campeoes.get(timeA) if campeoes.get(timeA) is not None else 0
        campeaoB = campeoes.get(timeB) if campeoes.get(timeB) is not None else 0

        x = np.array([idA, idB, campeaoA, campeaoB]).astype('float64')
        x = np.reshape(x, (1, -1))
        _y = svm_model.predict_proba(x)[0]

        # text = ('Chance de ' + timeA + ' vencer ' + timeB + ' é {}\nChance de ' + timeB + ' Vencer ' + timeA +
        #         ' é {}\nChance de ' + timeA + ' e ' + timeB + ' empatar é {} ').format(_y[1] * 100, _y[2] * 100,
        #                                                                                _y[0] * 100)

        text = f'Chance de {timeA} vencer {timeB} é {(_y[1] * 100):.2f}%\n' \
               f'Chance de {timeB} Vencer {timeA} é {(_y[2] * 100):.2f}%\n' \
               f'Chance de {timeA} e {timeB} empatar é {(_y[0]%.2 * 100):.2f}%'

        return _y[0], text

    prob1, text1 = predicao(selecao_da_casa, selecionar_selecao_visitante)

    if st.button('Realizar Predição do Jogo'):
        st.text(text1)


def page2():
    # Adicionando a logo
    st.sidebar.markdown("## Análise Exploratória dos Dados")

    # Adicionando a logo
    image = Image.open("logo.png")
    st.image(image)

    st.title("Análise Exploratória dos Dados")

    st.header("Entendimento do problema")
    st.write("Construir um algorítimo de Machine Learning capaz de predizer o vencedor de cada jogo da "
             "Copa do Mundo 2022")

    st.header("Coleta dos dados")
    st.write("Foi disponibilizado conjuntos de dados no formato Excel para que fosse construido o Banco de Dados "
             "por meio do MongoDB, a fim de alimenta-lo com os dados e poder realizar as consultas das tabelas "
             "para as futuras análises para o projeto.")

    st.header("Tabela: Jogos Copas do Mundo")

    # df jogos copas do mundo
    df_jogos = pd.read_csv("Jogos Copas do Mundo.csv", encoding="cp1252 ")
    st.dataframe(df_jogos)

    st.header("Tabela: Jogadores Copas do Mundo")

    # df jogadores copas do mundo
    df_jogadores = pd.read_csv("Jogadores.csv", encoding="cp1252 ")
    st.dataframe(df_jogadores)

    st.header("Tabela: Campeões Copas do Mundo")

    # df campeões copas do mundo
    df_campeoes = pd.read_csv("Campeoes.csv", encoding="cp1252 ")
    st.dataframe(df_campeoes)

    # Perguntas respondidas com os dado
    st.header("Insights da base de dados")

    st.subheader("1 - Quem são os maiores vencedores?")

    df_campeoes['Vencedor'].replace('Germany FR', 'Germany', inplace=True)

    descending_order = df_campeoes["Vencedor"].value_counts().sort_values(ascending=False).index

    fig = plt.figure(figsize=(12, 6))
    plt.title("Seleções vencedoras nas Copas do Mundo")
    sns.countplot(data=df_campeoes, x="Vencedor", order=descending_order)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("2 - Seleções que mais ficaram em 2º lugar nas Copas do Mundo")

    df_campeoes['Segundo'].replace('Germany FR', 'Germany', inplace=True)

    descending_order = df_campeoes["Segundo"].value_counts().sort_values(ascending=False).index

    fig = plt.figure(figsize=(12, 6))
    plt.title("Seleções que mais ficaram em 2º lugar nas Copas do Mundo")
    sns.countplot(data=df_campeoes, x="Segundo", order=descending_order)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Alterando o tipo da coluna de string para numérica
    df_jogos['Publico'] = df_jogos['Publico'].apply(pd.to_numeric)

    st.subheader("3 - Qual a média de Público nas Copas do Mundo")

    fig = plt.figure(figsize=(12, 6))
    plt.title("Média público nas Copas do mundo", color="black")
    sns.boxplot(x=df_jogos["Ano"], y=df_jogos["Publico"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 4 - Média de Gols nas Copas do Mundo
    df_jogos["TotalGols"] = df_jogos["GolsTimeDaCasa"] + df_jogos["GolsTimeVisitante"]

    st.subheader("4 - Média de Gols nas Copas do Mundo")

    fig = plt.figure(figsize=(12, 6))
    plt.title("Média de Gols", color="black")
    sns.boxplot(x=df_jogos["Ano"], y=df_jogos["TotalGols"])
    plt.xticks(rotation=45)
    st.pyplot(fig)


# Chamando as funções de cada página
page_names_to_funcs = {
    "Predição dos Jogos Copa do Mundo": page1,
    "Análise Exploratória dos Dados": page2,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
