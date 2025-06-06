import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clustering Jugadores Euroliga", layout="wide")

st.title("🏀 Análisis de Clustering de Jugadores - Euroliga")

# Cargar el CSV automáticamente
@st.cache_data
def load_data():
    return pd.read_csv("fiba_europe_stats_completo.csv")

df = load_data()
st.success("Datos cargados exitosamente.")
st.write("Vista previa de los datos:")
st.dataframe(df.head())

# Seleccionar variables numéricas para clustering
st.subheader("🔢 Selección de variables numéricas para clustering")

# Lista de columnas que pueden ser útiles para clustering
possible_features = [
    'PTS', 'FG%', '3P%', 'FT%', 'REB', 'AST_x', 'STL', 'BLK', 'TOV', 'TS%', 'eFG%', 
    'TRB%', 'AST%', 'TOV%', 'USG%', 'PER', 'PPS', 'ORtg', 'DRtg', 'eDiff', 'FIC'
]

numeric_cols = [col for col in possible_features if col in df.columns]
selected_features = st.multiselect("Selecciona variables:", numeric_cols, default=['PTS', 'REB', 'AST_x', 'PER'])

if selected_features:
    X = df[selected_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elegir número de clusters
    k = st.slider("Selecciona número de clusters (KMeans)", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = -1
    df.loc[X.index, 'Cluster'] = clusters

    st.subheader("📋 Resultados del Clustering")
    st.dataframe(df[['Player', 'Team_prom'] + selected_features + ['Cluster']])

    # Visualización
    if len(selected_features) >= 2:
        st.subheader("📈 Visualización de Clusters")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=X[selected_features[0]],
            y=X[selected_features[1]],
            hue=clusters,
            palette="Set2",
            ax=ax
        )
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_title("Clusters de jugadores")
        st.pyplot(fig)
else:
    st.warning("Selecciona al menos una variable para aplicar clustering.")
