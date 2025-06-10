# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px
import seaborn as sns
from scipy.stats import percentileofscore

# --- CONFIGURACION INICIAL ---
st.set_page_config(layout="wide", page_title="Perfiles Jugador")

@st.cache_data(show_spinner=False)
def cargar_y_preparar_datos(path):
    df = pd.read_csv(path)
    # Limpieza especial
    for col in ['Ast/TO', 'Stl/TO']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

    columnas_porcentaje = ['FG%', '3P%', 'FT%', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%']
    
    for col in columnas_porcentaje:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Solo convertir a float, no dividir

    # Convertir columnas numéricas a float32 para ahorrar memoria
    columnas_numericas = df.select_dtypes(include='number').columns
    for col in columnas_numericas:
        df[col] = df[col].astype('float32')

    return df

@st.cache_data(show_spinner=False)
def preparar_datos_cluster(df, variables):
    df_cluster = df[variables].copy()
    for col in variables:
        df_cluster[col] = pd.to_numeric(df_cluster[col], errors='coerce')
    df_cluster = df_cluster.dropna()
    return df_cluster

@st.cache_data(show_spinner=False)
def clustering_pca(X, n_clusters):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    return clusters, X_pca

df = cargar_y_preparar_datos("estadisticas_acb_2025.csv")

# Excluir columnas no numéricas ni relevantes para clustering
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_x', 'Team_y', 'Team_completo', 'Pos']
columnas_numericas = df.select_dtypes(include='number').columns
columnas_utiles = [col for col in columnas_numericas if col not in columnas_excluir]

st.sidebar.title("Configuración")
variables = st.sidebar.multiselect("Variables para clustering:", columnas_utiles, default=columnas_utiles[:4])
k = st.sidebar.slider("Número de clusters", 2, 10, 3)

mostrar_radar = st.sidebar.checkbox("Mostrar Radar Charts", True)
mostrar_dendros = st.sidebar.checkbox("Mostrar Dendrogramas", True)
mostrar_similares = st.sidebar.checkbox("Mostrar Jugadores Similares", True)
mostrar_corr = st.sidebar.checkbox("Mostrar Correlaciones", True)

if len(variables) < 2:
    st.error("Selecciona al menos 2 variables para continuar.")
    st.stop()

X = preparar_datos_cluster(df, variables)
if X.empty:
    st.error("No hay datos válidos para clustering.")
    st.stop()

clusters, X_pca = clustering_pca(X, k)

# Construir dataframe con solo columnas necesarias para reducir memoria
df_clustered = df.loc[X.index, ['Player', 'Team_completo', 'Pos'] + variables].copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

# --- VISUALIZACIONES EN TABS ---
tabs = st.tabs([
    "📊 Clusters",
    "🌳 Dendrogramas",
    "📈 Radar",
    "📏 Diferentes",
    "🎯 Similares",
    "🔥 Correlaciones",
    "📝 Scouting Report"
])

# --- TAB 1: Clusters ---
tabs[0].subheader("Jugadores por Cluster")
tabs[0].dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + variables + ['Cluster']])

fig = px.scatter(
    df_clustered,
    x='PCA1',
    y='PCA2',
    color=df_clustered['Cluster'].astype(str),
    hover_data=['Player', 'Team_completo', 'Pos'],
    title="PCA 2D - Clustering de Jugadores",
    color_discrete_sequence=px.colors.qualitative.Set1,
)
fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
fig.update_layout(legend_title_text='Cluster')
tabs[0].plotly_chart(fig, use_container_width=True)

# --- TAB 2: Dendrogramas ---
if mostrar_dendros:
    df_filt = df_clustered.copy()
    if len(df_filt) > 2:
        X_all = df_filt[variables].values
        labels_all = df_filt['Player'].values
        linkage_matrix = linkage(X_all, method='ward')
        fig, ax = plt.subplots(figsize=(20, 7))
        dendrogram(linkage_matrix, labels=labels_all, leaf_rotation=90, leaf_font_size=10)
        tabs[1].pyplot(fig)

# --- TAB 3: Radar Charts ---
if mostrar_radar:
    tabs[2].subheader("Radar Charts por Cluster")
    colores = plt.cm.viridis(np.linspace(0, 1, k))

    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        subset = df_clustered[df_clustered['Cluster'] == cluster_id]
        if len(subset) < 2:
            continue

        means = subset[variables].mean()
        scaler_radar = MinMaxScaler((0, 100))
        normalized = pd.Series(
            scaler_radar.fit_transform(means.values.reshape(-1, 1)).flatten(),
            index=means.index
        )
        if len(normalized) < 2:
            continue

        labels_radar = normalized.index.tolist()
        values = normalized.values.tolist()
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()

        # Cerrar el círculo
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        color = to_hex(colores[cluster_id % len(colores)])
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar)
        ax.set_title(f"Radar (0-100) - Cluster {cluster_id}")
        ax.set_yticklabels([])
        tabs[2].pyplot(fig)

# --- TAB 4: Jugadores diferentes ---
tabs[3].subheader("Jugadores más alejados del centroide")
for cluster_id in sorted(df_clustered['Cluster'].unique()):
    subset = df_clustered[df_clustered['Cluster'] == cluster_id]
    if len(subset) <= 2:
        continue
    X_subset = subset[variables].values
    centroid = X_subset.mean(axis=0)
    distances = np.linalg.norm(X_subset - centroid, axis=1)
    subset = subset.copy()
    subset['DistanciaCentroide'] = distances
    top5 = subset.sort_values(by='DistanciaCentroide', ascending=False).head(5)
    tabs[3].write(f"**Cluster {cluster_id}**")
    tabs[3].dataframe(top5[['Player', 'DistanciaCentroide']])

# --- TAB 5: Jugadores similares ---
if mostrar_similares:
    jugador = tabs[4].selectbox("Selecciona un jugador", df_clustered['Player'].unique())
    if tabs[4].button("Buscar similares"):
        X_sim = StandardScaler().fit_transform(df_clustered[variables])
        df_sim = pd.DataFrame(X_sim, index=df_clustered['Player'], columns=variables)
        jugador_vector = df_sim.loc[jugador].values
        df_sim['Distancia'] = df_sim.apply(lambda row: np.linalg.norm(row.values - jugador_vector), axis=1)
        similares = df_sim.sort_values(by='Distancia').iloc[1:6]
        tabs[4].dataframe(similares[['Distancia']])

# --- TAB 6: Correlaciones ---
if mostrar_corr:
    corr_matrix = df_clustered[variables].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    tabs[5].pyplot(fig)

def generar_texto_scouting(fortalezas, debilidades, percentiles):
    texto = ""
    if fortalezas:
        fortalezas_text = []
        for v in fortalezas:
            fortalezas_text.append(f"{v} (percentil {int(percentiles[v])})")
        texto += "🟢 **Fortalezas:** Destaca en " + ", ".join(fortalezas_text) + ".\n\n"
    if debilidades:
        debilidades_text = []
        for v in debilidades:
            debilidades_text.append(f"{v} (percentil {int(percentiles[v])})")
        texto += "🔴 **Debilidades:** Puede mejorar en " + ", ".join(debilidades_text) + ".\n\n"
    if not fortalezas and not debilidades:
        texto += "Perfil equilibrado, sin variables particularmente altas o bajas.\n\n"
    return texto

# --- TAB 7: Scouting Report ---
jugadora = tabs[6].selectbox("Selecciona una jugadora", df_clustered['Player'].unique(), key="scouting_player")
fila = df_clustered[df_clustered['Player'] == jugadora].iloc[0]
percentiles = {var: percentileofscore(df_clustered[var].dropna(), fila[var]) for var in variables}

fortalezas = [var for var, pct in percentiles.items() if pct >= 75]
debilidades = [var for var, pct in percentiles.items() if pct <= 25]

valores_normalizados = MinMaxScaler((0, 100)).fit_transform(df_clustered[variables]).T
valores_dict = dict(zip(df_clustered['Player'], valores_normalizados.T))
valores_radar = valores_dict[jugadora].tolist()
valores_radar += valores_radar[:1]

labels = variables
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.plot(angles, valores_radar, linewidth=2, label=jugadora)
ax.fill(angles, valores_radar, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title(f"Radar de {jugadora}")

tabs[6].pyplot(fig)
tabs[6].markdown("_Valores normalizados (0-100) para comparación entre variables._")

texto = f"**Informe de {jugadora}**\n\n" + generar_texto_scouting(fortalezas, debilidades, percentiles)
tabs[6].markdown(texto)
