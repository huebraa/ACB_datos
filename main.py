import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cargar datos
df = pd.read_csv('fiba_europe_stats_completo.csv')

st.title('Análisis de Clusters - Jugadoras FIBA Europa')

# Mostrar primeras filas
if st.checkbox("Mostrar datos crudos"):
    st.dataframe(df.head())

# Identificar columnas numéricas útiles para clustering
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_x', 'Team_y', 'Team_completo', 'Pos']
columnas_numericas = df.select_dtypes(include='number').columns
columnas_utiles = [col for col in columnas_numericas if col not in columnas_excluir]

# Selector de variables
variables = st.multiselect("Selecciona 2 o 3 variables para el clustering:", columnas_utiles, default=columnas_utiles[:3])

if len(variables) not in [2, 3]:
    st.warning("Por favor selecciona exactamente 2 o 3 variables.")
    st.stop()

# Limpiar y escalar datos
X = df[variables].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
k = st.slider("Número de clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

# Asociar los clusters al dataframe
df_clustered = df.loc[X.index].copy()
df_clustered['Cluster'] = clusters

st.success("Clustering realizado correctamente.")

# Mostrar resultados
st.subheader("Jugadoras por Cluster")
st.dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + variables + ['Cluster']].sort_values('Cluster'))

# Visualización
st.subheader("Visualización del Clustering")

fig = plt.figure()
if len(variables) == 2:
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', s=50)
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
else:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=clusters, cmap='viridis', s=50)
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_zlabel(variables[2])

st.pyplot(fig)
