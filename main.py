import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import plotly.graph_objects as go

from matplotlib.colors import to_hex
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
from scipy.stats import percentileofscore


st.set_page_config(layout="wide", page_title="Perfiles Jugadores")

# --- Función para cargar datos ---
@st.cache_data(show_spinner=False)
def cargar_datos(path):
    df = pd.read_csv(path)
    for col in ['Ast/TO', 'Stl/TO']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

    cols_pct = ['FG%', '3P%', 'FT%', 'TS%', 'eFG%', 'ORB%', 'DRB%', 'TRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%']
    for col in cols_pct:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Diccionario de rutas de archivos
ligas_dict = {
    "Liga ACB": "datos/estadisticas_acb_2025.csv",
    "Primera FEB": "datos/estadisticas_primera_feb_2025.csv"
}

st.sidebar.title("Configuración")

# Selección de ligas para cargar
ligas_seleccionadas = st.sidebar.multiselect(
    "Selecciona una o más ligas",
    options=list(ligas_dict.keys()),
    default=list(ligas_dict.keys())
)

if not ligas_seleccionadas:
    st.warning("Selecciona al menos una liga para continuar.")
    st.stop()

# Cargar datos de ligas seleccionadas
dfs_ligas = []
for liga in ligas_seleccionadas:
    df_liga = cargar_datos(ligas_dict[liga])
    df_liga["Liga"] = liga
    dfs_ligas.append(df_liga)

df = pd.concat(dfs_ligas, ignore_index=True)

# --- FILTROS ---
if "posiciones" not in st.session_state:
    st.session_state.posiciones = []
if "equipos" not in st.session_state:
    st.session_state.equipos = []
if "minutos" not in st.session_state:
    st.session_state.minutos = (int(df['MIN'].min()), int(df['MIN'].max()))

posiciones = st.sidebar.multiselect(
    "Filtrar por posición",
    sorted(df['Pos'].dropna().unique()),
    default=st.session_state.posiciones,
    key="posiciones"
)

equipos = st.sidebar.multiselect(
    "Filtrar por equipo",
    sorted(df['Team_completo'].dropna().unique()),
    default=st.session_state.equipos,
    key="equipos"
)

min_min = int(df['MIN'].min())
max_min = int(df['MIN'].max()) + 1

minutos_seleccionados = st.sidebar.slider(
    "Filtrar por minutos jugados (MIN)",
    min_value=min_min,
    max_value=max_min,
    value=st.session_state["minutos"],
    key="minutos"
)

def aplicar_filtros(df, posiciones, equipos, minutos):
    df_filt = df.copy()
    if posiciones:
        df_filt = df_filt[df_filt['Pos'].isin(posiciones)]
    if equipos:
        df_filt = df_filt[df_filt['Team_completo'].isin(equipos)]
    if minutos:
        df_filt = df_filt[(df_filt['MIN'] >= minutos[0]) & (df_filt['MIN'] <= minutos[1])]
    return df_filt

df_filtrado = aplicar_filtros(df, posiciones, equipos, minutos_seleccionados)

# --- Variables para clustering ---
columnas_excluir = ['#_prom', 'Player', 'Team_prom', '#_adv', 'Team_adv', 'Team_completo', 'Pos']
columnas_numericas = df_filtrado.select_dtypes(include='number').columns
variables = [c for c in columnas_numericas if c not in columnas_excluir]

vars_predeterminadas = ['3PA', '3P%', 'ORB%', 'TRB%', 'AST%', 'TOV%', 'BLK%', 'STL%', 'USG%', 'Ast/TO', 'Stl/TO', 'FT/FGA']
vars_predeterminadas = [v for v in vars_predeterminadas if v in variables]

st.sidebar.markdown("### Variables recomendadas para clustering:")
st.sidebar.markdown(", ".join(vars_predeterminadas))

vars_seleccionadas = st.sidebar.multiselect(
    "Variables para clustering",
    options=variables,
    default=st.session_state.get("vars_seleccionadas", vars_predeterminadas),
    key="vars_seleccionadas"
)

if len(vars_seleccionadas) < 2:
    st.error("Selecciona al menos 2 variables.")
    st.stop()

k = st.sidebar.slider(
    "Número de clusters",
    2, 10,
    value=st.session_state.get("k", 3),
    key="k"
)

mostrar_radar = st.sidebar.checkbox("Mostrar Radar Charts", True, key="mostrar_radar")
mostrar_dendros = st.sidebar.checkbox("Mostrar Dendrogramas", True, key="mostrar_dendros")
mostrar_similares = st.sidebar.checkbox("Mostrar Jugadores Similares", True, key="mostrar_similares")
mostrar_corr = st.sidebar.checkbox("Mostrar Correlaciones", True, key="mostrar_corr")

# --- Preprocesamiento ---
@st.cache_data(show_spinner=False)
def preprocesar(df_local, variables_local):
    df_local = df_local.dropna(subset=variables_local)
    scaler_local = StandardScaler()
    X_scaled = scaler_local.fit_transform(df_local[variables_local])
    return df_local, X_scaled, scaler_local

df_clustered, X_scaled, scaler = preprocesar(df_filtrado, vars_seleccionadas)

@st.cache_data(show_spinner=False)
def aplicar_kmeans(X_scaled_local, k_local):
    kmeans_local = KMeans(n_clusters=k_local, random_state=42, n_init='auto')
    clusters_local = kmeans_local.fit_predict(X_scaled_local)
    return clusters_local, kmeans_local

clusters, kmeans = aplicar_kmeans(X_scaled, k)

@st.cache_data(show_spinner=False)
def aplicar_pca(X_scaled_local):
    pca_local = PCA(n_components=2)
    X_pca_local = pca_local.fit_transform(X_scaled_local)
    return X_pca_local, pca_local

X_pca, pca = aplicar_pca(X_scaled)

df_clustered = df_clustered.reset_index(drop=True)
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = X_pca[:, 0]
df_clustered['PCA2'] = X_pca[:, 1]

def describir_cluster_avanzado(df_total, cluster_id, vars_seleccionadas, umbral=1.0):
    cluster_data = df_total[df_total['Cluster'] == cluster_id]
    if cluster_data.empty:
        return {"etiquetas": "Cluster vacío", "arquetipo_principal": None, "prototipos": []}

    global_mean = df_total[vars_seleccionadas].mean()
    global_std = df_total[vars_seleccionadas].std()
    centroid = cluster_data[vars_seleccionadas].mean()
    z_scores = ((centroid - global_mean) / global_std).sort_values(ascending=False)

    etiquetas = []

    # Etiquetas ofensivas
    if z_scores.get('AST%', 0) > umbral:
        etiquetas.append(("Playmaker", z_scores['AST%']))
    if z_scores.get('USG%', 0) > umbral and z_scores.get('FG%', 0) > 0:
        etiquetas.append(("Finalizador", z_scores['USG%']))
    if z_scores.get('3P%', 0) > umbral or z_scores.get('3PA', 0) > umbral:
        etiquetas.append(("Tirador", max(z_scores.get('3P%', 0), z_scores.get('3PA', 0))))
    if z_scores.get('FT/FGA', 0) > umbral and z_scores.get('USG%', 0) > 0:
        etiquetas.append(("Slasher", z_scores['FT/FGA']))

    # Etiquetas defensivas
    if z_scores.get('BLK%', 0) > umbral:
        etiquetas.append(("Protector del aro", z_scores['BLK%']))
    if z_scores.get('STL%', 0) > umbral:
        etiquetas.append(("Ladrón", z_scores['STL%']))
    if z_scores.get('STL%', 0) > umbral and z_scores.get('3P%', 0) > 0:
        etiquetas.append(("3&D", z_scores['STL%'] + z_scores.get('3P%', 0)))

    # Rebounding
    if z_scores.get('TRB%', 0) > umbral:
        etiquetas.append(("Reboteador", z_scores['TRB%']))
    if z_scores.get('ORB%', 0) > umbral and z_scores.get('DRB%', 0) > umbral:
        etiquetas.append(("Dominante en rebote", z_scores['ORB%'] + z_scores['DRB%']))
    elif z_scores.get('ORB%', 0) > umbral:
        etiquetas.append(("Reboteador ofensivo", z_scores['ORB%']))
    elif z_scores.get('DRB%', 0) > umbral:
        etiquetas.append(("Reboteador defensivo", z_scores['DRB%']))

    # Eficiencia
    if z_scores.get('Ast/TO', 0) > umbral:
        etiquetas.append(("Creador eficiente", z_scores['Ast/TO']))
    if z_scores.get('TOV%', 0) < -umbral:
        etiquetas.append(("Cuida el balón", -z_scores['TOV%']))

    # Si no hay nada destacado
    if not etiquetas:
        etiquetas = [("Perfil mixto", 0)]

    # Ordenar por impacto
    etiquetas.sort(key=lambda x: x[1], reverse=True)
    etiquetas_finales = [e[0] for e in etiquetas[:3]]

    # Arquetipo principal = más destacado
    arquetipo_principal = etiquetas_finales[0]

    # Diccionario de prototipos
    arquetipos_prototipos = {
        "Playmaker": ["Tyrese Haliburton", "Ricky Rubio"],
        "Finalizador": ["Zach LaVine", "Anthony Edwards"],
        "Tirador": ["Klay Thompson", "Buddy Hield"],
        "Slasher": ["DeMar DeRozan", "RJ Barrett"],
        "Protector del aro": ["Jaren Jackson Jr.", "Walker Kessler"],
        "Ladrón": ["Matisse Thybulle", "Alex Caruso"],
        "3&D": ["OG Anunoby", "Dorian Finney-Smith"],
        "Reboteador": ["Clint Capela", "Andre Drummond"],
        "Dominante en rebote": ["Steven Adams", "Domantas Sabonis"],
        "Reboteador ofensivo": ["Mitchell Robinson", "Kenneth Faried"],
        "Reboteador defensivo": ["Brook Lopez", "Rudy Gobert"],
        "Creador eficiente": ["Chris Paul", "Monte Morris"],
        "Cuida el balón": ["Malcolm Brogdon", "Tyus Jones"],
        "Perfil mixto": ["Bruce Brown", "Josh Hart"]
    }

    prototipos = arquetipos_prototipos.get(arquetipo_principal, [])

    return {
        "etiquetas": etiquetas_finales,
        "arquetipo_principal": arquetipo_principal,
        "prototipos": prototipos
    }



# --- Visualizaciones y tabs ---
tabs = st.tabs([
    "📊 Clusters",
    "🌳 Dendrogramas",
    "📈 Radar",
    "📏 Diferentes",
    "🎯 Similares",
    "🔥 Correlaciones",
    "📝 Scouting Report"
])

# TAB 1: Clusters
with tabs[0]:
    st.subheader("Jugadores por Cluster")
    st.dataframe(df_clustered[['Player', 'Team_completo', 'Pos'] + vars_seleccionadas])

    st.subheader("Perfil promedio por Cluster")
    resumen = df_clustered.groupby('Cluster')[vars_seleccionadas].mean().round(2)
    resultados = [
    etiquetar_y_prototipar_cluster(df_clustered, cluster_id, vars_seleccionadas)
    for cluster_id in resumen.index]
    resumen['Etiqueta'] = [describir_cluster_avanzado(df_clustered, cluster_id, vars_seleccionadas, umbral=1.0) for cluster_id in resumen.index]
    resumen['Arquetipo'] = [r['arquetipo_principal'] for r in resultados]
    resumen['Prototipos'] = [", ".join(r['prototipos']) for r in resultados]

    df_clustered['ClusterEtiqueta'] = df_clustered['Cluster'].map(resumen['Etiqueta'])
    df_clustered['PerfilJugador'] = df_clustered['Jugador'] + " — " + df_clustered['ClusterEtiqueta']



    st.dataframe(resumen)

    fig = px.scatter(
        df_clustered,
        x='PCA1', y='PCA2',
        color=df_clustered['Cluster'].astype(str),
        hover_data=['Player', 'Team_completo', 'Pos'],
        title="PCA 2D - Clustering de Jugadores",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(legend_title_text='Cluster')
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: Dendrogramas
with tabs[1]:
    if mostrar_dendros:
        clusters_unicos = sorted(df_clustered['Cluster'].unique())
        cluster_sel = st.selectbox(
            "Filtrar dendrograma por cluster",
            options=[-1] + clusters_unicos,
            format_func=lambda x: "Todos" if x == -1 else f"Cluster {x}",
            key="cluster_dendro"
        )

        if cluster_sel == -1:
            df_dendro = df_clustered
        else:
            df_dendro = df_clustered[df_clustered['Cluster'] == cluster_sel]

        if len(df_dendro) > 2:
            linkage_matrix = linkage(df_dendro[vars_seleccionadas], method='ward')
            fig = ff.create_dendrogram(
                df_dendro[vars_seleccionadas],
                labels=df_dendro['Player'].values,
                linkagefun=lambda x: linkage_matrix
            )
            fig.update_layout(width=1000, height=600)
            st.plotly_chart(fig)
        else:
            st.info("Pocos datos para dendrograma en este cluster.")

# TAB 3: Radar
with tabs[2]:
    if mostrar_radar:
        st.subheader("Radar Charts por Cluster")
        colores = plt.cm.viridis(np.linspace(0, 1, k))

        scaler_radar = MinMaxScaler((0, 100))
        for cluster_id in sorted(df_clustered['Cluster'].unique()):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            if len(cluster_data) < 2:
                continue
            mean_vals = cluster_data[vars_seleccionadas].mean().values.reshape(-1, 1)
            normalized = scaler_radar.fit_transform(mean_vals).flatten()

            labels = vars_seleccionadas
            values = list(normalized) + [normalized[0]]
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
            ax.plot(angles, values, color=to_hex(colores[cluster_id]), linewidth=2)
            ax.fill(angles, values, color=to_hex(colores[cluster_id]), alpha=0.25)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            ax.set_yticklabels([])
            ax.set_title(f"Radar (0-100) - Cluster {cluster_id}")
            st.pyplot(fig)

# TAB 4: Más alejados del centroide
with tabs[3]:
    st.subheader("Jugadores más alejados del centroide")
    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
        if len(cluster_data) <= 2:
            st.info(f"Pocos jugadores en cluster {cluster_id} para mostrar.")
            continue
        centroid = cluster_data[vars_seleccionadas].mean().values
        distances = np.linalg.norm(cluster_data[vars_seleccionadas] - centroid, axis=1)
        cluster_data = cluster_data.copy()
        cluster_data['DistanciaCentroide'] = distances
        top5 = cluster_data.sort_values(by='DistanciaCentroide', ascending=False).head(5)
        st.write(f"**Cluster {cluster_id}**")
        st.dataframe(top5[['Player', 'Pos', 'Team_completo', 'DistanciaCentroide'] + vars_seleccionadas])

# TAB 5: Jugadores similares
with tabs[4]:
    if mostrar_similares:
        st.subheader("Buscar jugadores similares")
        jugador = st.selectbox("Selecciona un jugador", df_clustered['Player'].sort_values().unique())
        jugador_data = df_clustered[df_clustered['Player'] == jugador][vars_seleccionadas]
        if jugador_data.empty:
            st.warning("Jugador no encontrado.")
        else:
            jugador_vals = jugador_data.values[0].reshape(1, -1)
            jugador_vals_scaled = scaler.transform(jugador_vals)
            variables_scaled = scaler.transform(df_clustered[vars_seleccionadas])
        
            df_clustered['DistSim'] = np.linalg.norm(variables_scaled - jugador_vals_scaled, axis=1)
            similares = df_clustered[df_clustered['Player'] != jugador].sort_values('DistSim').head(10)
            st.dataframe(similares[['Player', 'Pos', 'Team_completo', 'Cluster', 'DistSim'] + vars_seleccionadas])

# TAB 6: Correlaciones
with tabs[5]:
    if mostrar_corr:
        st.subheader("Mapa de calor de correlaciones")
        corr = df_clustered[vars_seleccionadas].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
# TAB 7: Scouting Report
with tabs[6]:


    vars_perfil = vars_seleccionadas  # variables ya definidas en tu código
    vars_rendimiento = ['ORtg', 'DRtg', 'eDiff', 'FIC', 'PER', 'OWS', 'DWS', 'WS']
    vars_todas = vars_perfil + vars_rendimiento

    def mostrar_scouting_dos_columnas(fila_1, df_posicion, vars_seleccionadas):
        percentiles = {var: percentileofscore(df_posicion[var].dropna(), fila_1[var]) for var in vars_seleccionadas}
        fortalezas = [(var, int(pct)) for var, pct in percentiles.items() if pct >= 75]
        debilidades = [(var, int(pct)) for var, pct in percentiles.items() if pct <= 25]

        st.markdown("### 🏀 Informe de Fortalezas y Debilidades")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 Fortalezas")
            if fortalezas:
                for var, pct in fortalezas:
                    st.markdown(f"**{var}** — Percentil {pct}")
                    st.progress(pct / 100)
            else:
                st.markdown("✅ Perfil equilibrado sin áreas sobresalientes.")

        with col2:
            st.markdown("#### 🔴 Debilidades")
            if debilidades:
                for var, pct in debilidades:
                    st.markdown(f"**{var}** — Percentil {pct}")
                    st.progress(pct / 100)
            else:
                st.markdown("🟢 Sin áreas críticas de mejora detectadas.")

    st.subheader("🔍 Scouting individual y comparativo")

    col1, col2 = st.columns(2)
    with col1:
        jugadora_1 = st.selectbox("Selecciona una jugadora principal", df_clustered['Player'].unique(), key="scouting_1")
    with col2:
        jugadora_2 = st.selectbox(
            "Comparar con otra jugadora (opcional)",
            ["Promedio de su posición", "Promedio de su cluster"] + list(df_clustered['Player'].unique()),
            key="scouting_2"
        )

    fila_1 = df_clustered[df_clustered['Player'] == jugadora_1].iloc[0]
    posicion = fila_1['Pos']
    cluster = fila_1['Cluster']
    df_posicion = df_clustered[df_clustered['Pos'] == posicion]

    scaler = MinMaxScaler((0, 100))
    normalizados = scaler.fit_transform(df_clustered[vars_todas])
    df_norm = pd.DataFrame(normalizados, columns=vars_todas)
    df_norm['Player'] = df_clustered['Player'].values

    valores_1 = df_norm[df_norm['Player'] == jugadora_1][vars_todas].values.flatten().tolist()

    if jugadora_2 == "Promedio de su posición":
        df_pos = df_clustered[df_clustered['Pos'] == posicion]
        valores_2 = scaler.transform(df_pos[vars_todas]).mean(axis=0).tolist()
        nombre_2 = f"Promedio {posicion}"
        color_2 = "gray"
        dash_2 = 'dash'
    elif jugadora_2 == "Promedio de su cluster":
        df_clu = df_clustered[df_clustered['Cluster'] == cluster]
        valores_2 = scaler.transform(df_clu[vars_todas]).mean(axis=0).tolist()
        nombre_2 = f"Promedio Cluster {cluster}"
        color_2 = "orange"
        dash_2 = 'dot'
    else:
        valores_2 = df_norm[df_norm['Player'] == jugadora_2][vars_todas].values.flatten().tolist()
        nombre_2 = jugadora_2
        color_2 = "firebrick"
        dash_2 = 'solid'

    # Cerrar el radar
    valores_1 += valores_1[:1]
    valores_2 += valores_2[:1]
    labels = vars_todas + vars_todas[:1]

    # Crear radar con Plotly
    fig = go.Figure()

    # Bloque perfil
    fig.add_trace(go.Scatterpolar(
        r=valores_1[:len(vars_perfil)+1],
        theta=labels[:len(vars_perfil)+1],
        fill='toself',
        name=jugadora_1 + " - Perfil",
        line=dict(color='#006699', width=3),
        fillcolor='rgba(0,102,153,0.3)',
        hoverinfo='all'
    ))

    # Bloque rendimiento
    fig.add_trace(go.Scatterpolar(
        r=valores_1[len(vars_perfil):],
        theta=labels[len(vars_perfil):],
        fill='toself',
        name=jugadora_1 + " - Rendimiento",
        line=dict(color='#ffa500', width=3),
        fillcolor='rgba(255,165,0,0.3)',
        hoverinfo='all'
    ))

    # Comparativa jugador 2 - perfil
    fig.add_trace(go.Scatterpolar(
        r=valores_2[:len(vars_perfil)+1],
        theta=labels[:len(vars_perfil)+1],
        fill='toself',
        name=nombre_2 + " - Perfil",
        line=dict(color=color_2, width=2, dash=dash_2),
        fillcolor='rgba(128,128,128,0.2)' if dash_2 == 'dash' else 'rgba(255,165,0,0.15)',
        hoverinfo='all'
    ))

    # Comparativa jugador 2 - rendimiento
    fig.add_trace(go.Scatterpolar(
        r=valores_2[len(vars_perfil):],
        theta=labels[len(vars_perfil):],
        fill='toself',
        name=nombre_2 + " - Rendimiento",
        line=dict(color=color_2, width=2, dash=dash_2),
        fillcolor='rgba(255,165,0,0.15)',
        hoverinfo='all'
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="#f9f9f9",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"],
                gridcolor="lightgray",
                gridwidth=1
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color="black", family="Arial Black"),
            )
        ),
        legend=dict(
            title="Leyenda",
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            x=1.1,
            y=1
        ),
        margin=dict(t=50, b=50, l=50, r=150),
        title=f"{jugadora_1} vs {nombre_2} - Perfil y Rendimiento",
        title_font_size=18
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("_El radar está dividido en dos bloques: **Perfil** (azul) y **Rendimiento** (naranja)._")
    st.markdown("_Valores normalizados de 0 a 100._")

    mostrar_scouting_dos_columnas(fila_1, df_posicion, vars_perfil)

