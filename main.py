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
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURACION INICIAL ---
st.set_page_config(layout="wide", page_title="Cluster Jugadoras FIBA Europa")

# --- SIDEBAR ---
st.sidebar.title("Configuración")
df = pd.read_csv("fiba_europe_stats_completo.csv")

# LIMPIEZA ESPECIAL
for col in ['Ast/TO', 'Stl/TO']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

columnas_porcentaje = ['3P%', 'ORB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%']
for col in columnas_porcentaje:
    if col in df.columns:
        df[col] = df[col].astype(str).str.rstrip('%').replace('', np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce') / 100

columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_x', 'Team_y', 'Team_completo', 'Pos']
columnas_numericas = df.select_dtypes(include='number').columns
columnas_utiles = [col for col in columnas_numericas if col not in columnas_excluir]

variables = st.sidebar.multiselect("Variables para clustering:", columnas_utiles, default=columnas_utiles[:4])
k = st.sidebar.slider("Número de clusters", 2, 10, 3)

mostrar_radar = st.sidebar.checkbox("Mostrar Radar Charts", True)
mostrar_dendros = st.sidebar.checkbox("Mostrar Dendrogramas", True)
mostrar_similares = st.sidebar.checkbox("Mostrar Jugadores Similares", True)
mostrar_corr = st.sidebar.checkbox("Mostrar Correlaciones", True)

# --- PROCESAMIENTO DE DATOS ---
if len(variables) < 2:
    st.error("Selecciona al menos 2 variables para continuar.")
    st.stop()

for col in variables:
    df[col] = pd.to_numeric(df[col], errors='coerce')

X = df[variables].replace([np.inf, -np.inf], np.nan).dropna()
if X.empty:
    st.error("No hay datos válidos para clustering.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
clusters = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clustered = df.loc[X.index].copy()
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
tabs[0].subheader("Jugadoras por Cluster")
tabs[0].dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + variables + ['Cluster']])

fig = px.scatter(
    df_clustered, x='PCA1', y='PCA2', color='Cluster',
    hover_data=['Player', 'Team_completo', 'Pos'],
    title="PCA 2D - Clustering de Jugadoras"
)
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
            continue  # Evitar radar charts inválidos

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

# --- TAB 7: Scouting Report ---
from matplotlib.colors import to_rgba

jugadora = tabs[6].selectbox("Selecciona una jugadora", df_clustered['Player'].unique(), key="scouting_player")
fila = df_clustered[df_clustered['Player'] == jugadora].iloc[0]

# Calcular percentiles
percentiles = {var: percentileofscore(df_clustered[var].dropna(), fila[var]) for var in variables}
fortalezas = [var for var, pct in percentiles.items() if pct >= 75]
debilidades = [var for var, pct in percentiles.items() if pct <= 25]

# Normalizar valores para radar 0-100 usando MinMaxScaler
scaler_radar = MinMaxScaler(feature_range=(0, 100))
valores_array = df_clustered[variables].values
scaler_radar.fit(valores_array)
valores_norm = scaler_radar.transform([fila[variables].values])[0]
valores_radar = valores_norm.tolist()
valores_radar += valores_radar[:1]  # cerrar círculo

num_vars = len(variables)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
color = "#1f77b4"
rgba_fill = to_rgba(color, alpha=0.25)
rgba_line = to_rgba(color, alpha=0.8)

# Dibujar círculos guía sutiles (25, 50, 75)
for r in [25, 50, 75]:
    ax.plot(np.linspace(0, 2*np.pi, 100), [r]*100, 
            color='gray', linestyle='--', linewidth=0.5, alpha=0.2)

# Plot radar minimalista
ax.plot(angles, valores_radar, color=rgba_line, linewidth=2)
ax.fill(angles, valores_radar, color=rgba_fill)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(variables, fontsize=11, fontweight='bold', color='black')

ax.spines['polar'].set_visible(False)
ax.grid(False)
ax.set_ylim(0, 100)
ax.set_title(f"Radar de {jugadora}", fontsize=14, fontweight='bold', y=1.1)

tabs[6].pyplot(fig)

# Informe de texto
texto = f"**Informe de {jugadora}**\n\n"
if fortalezas:
    texto += "🟢 **Fortalezas**: " + ", ".join(f"{v} (pctl {int(percentiles[v])})" for v in fortalezas) + "\n"
if debilidades:
    texto += "🔴 **Debilidades**: " + ", ".join(f"{v} (pctl {int(percentiles[v])})" for v in debilidades) + "\n"
if not fortalezas and not debilidades:
    texto += "Perfil equilibrado, sin variables particularmente altas o bajas."

tabs[6].markdown(texto)





