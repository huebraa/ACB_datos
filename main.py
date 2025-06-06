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

# Cargar datos
df = pd.read_csv('fiba_europe_stats_completo.csv')

# --- LIMPIEZA ESPECIAL PARA 'Ast/TO' y 'Stl/TO' ---
for col in ['Ast/TO', 'Stl/TO']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
# --- FIN LIMPIEZA ---

st.title('Análisis de Clusters - Jugadoras FIBA Europa (2D PCA)')

# Columnas útiles para clustering
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_x', 'Team_y', 'Team_completo', 'Pos']
columnas_numericas = df.select_dtypes(include='number').columns
columnas_utiles = [col for col in columnas_numericas if col not in columnas_excluir]

# Layout columnas
col_izq, col_der = st.columns([1, 3])

with col_izq:
    st.header("Configuración")

    mostrar_datos = st.checkbox("Mostrar datos crudos")
    variables = st.multiselect("Variables para clustering:", columnas_utiles, default=columnas_utiles[:4])

    if len(variables) < 2:
        st.warning("Selecciona al menos 2 variables.")
        st.stop()

    k = st.slider("Número de clusters", 2, 10, 3)

    jugador_seleccionado = st.selectbox("Selecciona un jugador para recomendar similares:", sorted(df['Player'].unique()))

    btn_similares = st.button("Recomendar jugadores similares")

    mostrar_dendrograma_global = st.checkbox("Mostrar dendrograma global", value=True)
    mostrar_dendrogramas_por_cluster = st.checkbox("Mostrar dendrogramas por cluster", value=True)
    mostrar_radar_charts = st.checkbox("Mostrar radar charts por cluster", value=False)
    mostrar_atipicos = st.checkbox("Mostrar jugadores atípicos por cluster", value=False)
    mostrar_mapa_calor = st.checkbox("Mostrar mapa de calor de correlaciones", value=True)

with col_der:
    if mostrar_datos:
        st.subheader("Datos crudos")
        st.dataframe(df.head())

    # Limpiar columnas con '%'
    columnas_porcentaje = ['3P%', 'ORB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%']
    for col in columnas_porcentaje:
        if col in df.columns:
            df[col] = df[col].astype(str).str.rstrip('%').replace('', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce') / 100

    # Convertir variables seleccionadas a numérico
    for col in variables:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Preparar DataFrame para clustering
    X = df[variables].copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    if X.empty:
        st.error("No hay datos válidos para escalar después de limpiar NaN/Inf. Revisa las columnas seleccionadas.")
        st.stop()

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)

    # PCA 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_clustered = df.loc[X.index].copy()
    df_clustered['Cluster'] = clusters
    df_clustered['PCA1'] = X_pca[:, 0]
    df_clustered['PCA2'] = X_pca[:, 1]

    st.subheader("Jugadoras por Cluster")
    st.dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + variables + ['Cluster']].sort_values('Cluster'))

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
        legend=dict(itemsizing='constant', bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=40, r=40, t=40, b=40),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dendrograma global
    if mostrar_dendrograma_global:
        st.subheader("🌳 Dendrograma jerárquico global")
        if len(df_clustered) > 2:
            X_all = df_clustered[variables].values
            labels_all = df_clustered['Player'].values
            linkage_matrix_global = linkage(X_all, method='ward')

            fig, ax = plt.subplots(figsize=(20, 7))
            dendrogram(linkage_matrix_global, labels=labels_all, orientation='top', leaf_rotation=90, leaf_font_size=10,
                       color_threshold=0, above_threshold_color='grey', ax=ax)
            ax.set_title("🌳 Dendrograma jerárquico global (todos los jugadores)", fontsize=16)
            st.pyplot(fig)
        else:
            st.info("⚠️ Muy pocos jugadores para dendrograma global.")

    # Dendrogramas por cluster
    if mostrar_dendrogramas_por_cluster:
        st.subheader("🌳 Dendrogramas por cluster")
        unique_clusters = sorted(df_clustered['Cluster'].unique())
        colores = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        for idx, cluster_id in enumerate(unique_clusters):
            subset = df_clustered[df_clustered['Cluster'] == cluster_id]
            if len(subset) <= 2:
                continue
            X_cluster = subset[variables].values
            linkage_matrix = linkage(X_cluster, method='ward')

            fig, ax = plt.subplots(figsize=(10, 5))
            dendrogram(linkage_matrix, labels=subset['Player'].values, orientation='top', leaf_rotation=90,
                       leaf_font_size=8, color_threshold=0, above_threshold_color=to_hex(colores[idx]), ax=ax)
            ax.set_title(f"🌳 Dendrograma - Cluster {cluster_id}", fontsize=14)
            st.pyplot(fig)

    # Radar charts por cluster
    if mostrar_radar_charts:
        st.subheader("📊 Radar charts por cluster")
        unique_clusters = sorted(df_clustered['Cluster'].unique())
        colores = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
        for idx, cluster_id in enumerate(unique_clusters):
            subset = df_clustered[df_clustered['Cluster'] == cluster_id]
            if len(subset) <= 2:
                continue
            means = subset[variables].mean()
            scaler_radar = MinMaxScaler((0, 100))
            normalized = pd.Series(scaler_radar.fit_transform(means.values.reshape(-1, 1)).flatten(), index=means.index)

            labels_radar = normalized.index.tolist()
            values = normalized.values.tolist()
            angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax.fill(angles, values, color=to_hex(colores[idx]), alpha=0.25)
            ax.plot(angles, values, color=to_hex(colores[idx]), linewidth=2)
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels_radar, fontsize=10)
            ax.set_title(f"Radar (0-100) - Cluster {cluster_id}", fontsize=14)
            st.pyplot(fig)

    # Jugadores más alejados del centroide
    if mostrar_atipicos:
        st.subheader("📏 Jugadores más alejados del centroide por cluster")
        unique_clusters = sorted(df_clustered['Cluster'].unique())
        for cluster_id in unique_clusters:
            subset = df_clustered[df_clustered['Cluster'] == cluster_id]
            if len(subset) <= 2:
                continue

            X_subset = subset[variables].values
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

    # Jugadores similares
    st.subheader("🎯 Buscar jugadores similares")
    if btn_similares:
        X_sim = df_clustered[variables]
        scaler_sim = StandardScaler()
        X_scaled_sim = scaler_sim.fit_transform(X_sim)
        df_scaled_sim = pd.DataFrame(X_scaled_sim, columns=variables, index=df_clustered['Player'])
    
        if jugador_seleccionado not in df_scaled_sim.index:
            st.error("Jugador no válido o datos incompletos")
        else:
            jugador_vector = df_scaled_sim.loc[jugador_seleccionado].values.reshape(1, -1)
            # Calcular distancias vectorizadas a todos los jugadores
            distancias = np.linalg.norm(df_scaled_sim.values - jugador_vector, axis=1)
            df_scaled_sim['Distancia'] = distancias
    
            similares = df_scaled_sim.sort_values(by='Distancia').iloc[1:11]  # Los 10 más similares, excluyendo el mismo
            st.write(f"Jugadores más similares a **{jugador_seleccionado}**:")
            st.dataframe(similares[['Distancia']])


    # Mapa de calor de correlaciones
    if mostrar_mapa_calor:
        st.subheader("Mapa de calor de correlaciones entre variables")
        if len(variables) >= 2:
            corr_matrix = df_clustered[variables].corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
            plt.title('Mapa de calor de correlaciones entre variables')

            st.pyplot(fig)
        else:
            st.info("Selecciona al menos 2 variables para mostrar el mapa de calor.")
