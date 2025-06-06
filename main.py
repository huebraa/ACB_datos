import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Clustering Euroliga", layout="wide")

st.title("🏀 Clustering de Jugadores - Euroliga")

@st.cache_data
def load_data():
    return pd.read_csv("fiba_europe_stats_completo.csv")

df = load_data()
st.success("Datos cargados correctamente.")

# Mostrar preview de los datos
st.subheader("📊 Vista previa del dataset")
st.dataframe(df.head())

# Filtrar automáticamente columnas numéricas útiles
excluded_cols = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Pos', 
                 'Team_x', 'Team_y', 'Team_completo']
numeric_df = df.drop(columns=[col for col in excluded_cols if col in df.columns], errors='ignore')
numeric_cols = numeric_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

st.subheader("🔧 Selección de variables para clustering")
selected_features = st.multiselect("Selecciona columnas numéricas para usar en el clustering:", 
                                   numeric_cols, 
                                   default=['PTS', 'REB', 'AST_x', 'PER'])

if len(selected_features) >= 2:
    X = df[selected_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    k = st.slider("Selecciona número de clusters (KMeans)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = -1
    df.loc[X.index, 'Cluster'] = clusters

    st.subheader("📋 Resultados")
    st.dataframe(df[['Player', 'Team_prom'] + selected_features + ['Cluster']])

    # Visualización
    st.subheader("📈 Visualización de Clusters")
    if len(selected_features) == 2:
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=X[selected_features[0]], 
            y=X[selected_features[1]], 
            hue=clusters, 
            palette="Set2", 
            ax=ax
        )
        ax.set_title("Clusters 2D")
        st.pyplot(fig)

    elif len(selected_features) >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[selected_features[0]], X[selected_features[1]], X[selected_features[2]],
                   c=clusters, cmap="Set2", s=60)
        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_zlabel(selected_features[2])
        ax.set_title("Clusters 3D")
        st.pyplot(fig)

else:
    st.warning("Selecciona al menos 2 variables para aplicar clustering.")
