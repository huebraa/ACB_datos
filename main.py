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

# Aquí crearías tu gráfico:
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

# Aquí agregas el update_layout para ajustar diseño y tamaño:
fig.update_layout(
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
    ),
    width=None,     # No fijo, para que ocupe todo el ancho
    height=600      # Altura fija
)

# Finalmente muestras la figura usando todo el ancho:
st.plotly_chart(fig, use_container_width=True)

st.subheader("🌳 Dendrograma jerárquico global (mejorado para lectura)")

df_filtrado = df_clustered.copy()
columnas_seleccionadas = variables  # Usamos las variables que seleccionaste

if len(df_filtrado) > 2:
    X_all = df_filtrado[columnas_seleccionadas].values
    labels_all = df_filtrado['Player'].values
    linkage_matrix_global = linkage(X_all, method='ward')

    fig, ax = plt.subplots(figsize=(20, 7))
    dendrogram(
        linkage_matrix_global,
        labels=labels_all,
        orientation='top',
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color='grey',
        ax=ax
    )
    ax.set_title("🌳 Dendrograma jerárquico global (todos los jugadores)", fontsize=16)
    st.pyplot(fig)
else:
    st.info("⚠️ Muy pocos jugadores para dendrograma global.")

# Dendrogramas por cluster
unique_clusters = sorted(df_filtrado['Cluster'].unique())
colores = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))

for idx, cluster_id in enumerate(unique_clusters):
    subset = df_filtrado[df_filtrado['Cluster'] == cluster_id]
    if len(subset) <= 2:
        continue
    X_cluster = subset[columnas_seleccionadas].values
    linkage_matrix = linkage(X_cluster, method='ward')

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linkage_matrix,
        labels=subset['Player'].values,
        orientation='top',
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0,
        above_threshold_color=to_hex(colores[idx]),
        ax=ax
    )
    ax.set_title(f"🌳 Dendrograma - Cluster {cluster_id}", fontsize=14)
    st.pyplot(fig)

# Radar charts
st.subheader("📊 Radar charts por Cluster")

for cluster_id in unique_clusters:
    subset = df_filtrado[df_filtrado['Cluster'] == cluster_id]
    if len(subset) <= 2:
        continue
    means = subset[columnas_seleccionadas].mean()
    scaler = MinMaxScaler((0, 100))
    normalized = pd.Series(scaler.fit_transform(means.values.reshape(-1, 1)).flatten(), index=means.index)

    labels_radar = normalized.index.tolist()
    values = normalized.values.tolist()
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color=to_hex(colores[cluster_id % len(colores)]), alpha=0.25)
    ax.plot(angles, values, color=to_hex(colores[cluster_id % len(colores)]), linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=10)
    ax.set_title(f"Radar (0-100) - Cluster {cluster_id}", fontsize=14)
    st.pyplot(fig)

# 🧬 Detección de jugadores atípicos
st.subheader("📏 Jugadores más alejados del centroide por cluster")

for cluster_id in unique_clusters:
    subset = df_filtrado[df_filtrado['Cluster'] == cluster_id]
    if len(subset) <= 2:
        continue

    X_subset = subset[columnas_seleccionadas].values
    centroid = X_subset.mean(axis=0)
    distances = np.linalg.norm(X_subset - centroid, axis=1)

    subset = subset.copy()
    subset['DistanciaCentroide'] = distances

    media = distances.mean()
    std = distances.std()
    umbral = media + 1.5 * std

    top5 = subset.sort_values(by='DistanciaCentroide', ascending=False).head(5)

    st.write(f"🔹 **Cluster {cluster_id} - Jugadores más diferentes:**")
    for _, row in top5.iterrows():
        alerta = "⚠️ DIFERENTE" if row['DistanciaCentroide'] > umbral else ""
        st.write(f"- {row['Player']:<25} Distancia: {row['DistanciaCentroide']:.2f} {alerta}")
