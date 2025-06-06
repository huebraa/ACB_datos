import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

# Cargar datos
df = pd.read_csv('fiba_europe_stats_completo.csv')

st.title('Análisis de Clusters - Jugadoras FIBA Europa (2D PCA)')

# Mostrar datos
if st.checkbox("Mostrar datos crudos"):
    st.dataframe(df.head())

# Columnas útiles para clustering
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_x', 'Team_y', 'Team_completo', 'Pos']
columnas_numericas = df.select_dtypes(include='number').columns
columnas_utiles = [col for col in columnas_numericas if col not in columnas_excluir]

# Selección de variables
variables = st.multiselect("Selecciona las variables que quieras usar para el clustering:", columnas_utiles, default=columnas_utiles[:4])

if len(variables) < 2:
    st.warning("Selecciona al menos 2 variables.")
    st.stop()

# Limpiar y escalar datos
X = df[variables].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
k = st.slider("Número de clusters", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

# Reducir a 2D con PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DataFrame con clusters
df_clustered = df.loc[X.index].copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

# Mostrar resultados
st.subheader("Jugadoras por Cluster")
st.dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + variables + ['Cluster']].sort_values('Cluster'))

# Visualización 2D interactiva con Plotly
st.subheader("Visualización 2D con PCA y Clusters")

fig = px.scatter(
    df_clustered, x='PCA1', y='PCA2',
    color='Cluster',
    hover_data=['Player', 'Team_completo', 'Pos'],
    color_continuous_scale=px.colors.qualitative.Set2,
    title="Clustering de Jugadoras - PCA 2D",
    labels={'PCA1': 'Componente Principal 1', 'PCA2': 'Componente Principal 2'}
)

fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color='DarkSlateGrey')))
fig.update_layout(
    template='simple_white',
    legend_title_text='Cluster',
    legend=dict(
        itemsizing='constant',
        bgcolor='rgba(0,0,0,0)'
    ),
    margin=dict(l=40, r=40, t=40, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

st.plotly_chart(fig, use_container_width=True)
