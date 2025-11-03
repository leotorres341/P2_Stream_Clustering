import os
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
from io import BytesIO
import requests

# =====================================
# CONFIGURACIÓN
# =====================================
# Usar ruta local: los CSV deben estar en la misma carpeta que app.py
base_path = "."
umap_file = os.path.join(base_path, "umap_dbscan_resultados.csv")
movie_file = os.path.join(base_path, "MovieGenre.csv")

st.set_page_config(page_title="Visualizador de Películas UMAP+DBSCAN", layout="wide")
st.title("🎬 Sistema de Recomendación Basado en Clustering Visual (UMAP + DBSCAN)")

# =====================================
# CARGAR DATOS
# =====================================
@st.cache_data
def load_data():
    umap_df = pd.read_csv(umap_file)
    movie_df = pd.read_csv(movie_file, encoding="latin-1")

    if "Cluster_DBSCAN" not in umap_df.columns:
        umap_df.rename(columns={"cluster_dbscan": "Cluster_DBSCAN"}, inplace=True)

    return umap_df, movie_df

umap_df, movie_df = load_data()
id_col, genre_col = "imdbId", "Genre"

# =====================================
# FILTROS
# =====================================
st.sidebar.header("🎚️ Filtros")
genres = sorted(umap_df[genre_col].dropna().unique().tolist())
sel_genre = st.sidebar.selectbox("Género:", ["Todos"] + genres)

clusters = sorted(umap_df["Cluster_DBSCAN"].unique().tolist())
sel_cluster = st.sidebar.selectbox("Cluster:", ["Todos"] + [str(c) for c in clusters])

df_filtered = umap_df.copy()
if sel_genre != "Todos":
    df_filtered = df_filtered[df_filtered[genre_col] == sel_genre]
if sel_cluster != "Todos":
    df_filtered = df_filtered[df_filtered["Cluster_DBSCAN"] == int(sel_cluster)]

# =====================================
# VISUALIZACIÓN 2D
# =====================================
st.subheader("📊 Distribución 2D de películas (UMAP + DBSCAN)")

fig = px.scatter(
    df_filtered,
    x="umap_x",
    y="umap_y",
    color=df_filtered["Cluster_DBSCAN"].astype(str),
    hover_data=[genre_col],
    title="Espacio bidimensional de características visuales",
    color_discrete_sequence=px.colors.qualitative.Set2,
)
st.plotly_chart(fig, use_container_width=True)

# =====================================
# PELÍCULAS REPRESENTATIVAS DE CADA CLUSTER
# =====================================
st.subheader("🎞️ Películas representativas de cada cluster")

for cl in sorted(df_filtered["Cluster_DBSCAN"].unique()):
    st.markdown(f"#### Cluster {cl}")
    cluster_movies = df_filtered[df_filtered["Cluster_DBSCAN"] == cl]
    if len(cluster_movies) == 0:
        st.write("Sin películas en este cluster.")
        continue

    sample_movies = cluster_movies.sample(min(6, len(cluster_movies)))
    cols = st.columns(3)
    for i, (_, row) in enumerate(sample_movies.iterrows()):
        try:
            poster_url = movie_df.loc[movie_df[id_col] == row[id_col], "Poster"].values[0]
            img = Image.open(BytesIO(requests.get(poster_url, timeout=5).content))
            with cols[i % 3]:
                st.image(img, caption=f"{row[genre_col]}", use_column_width=True)
        except Exception:
            continue

# =====================================
# RECOMENDADOR
# =====================================
st.subheader("🔍 Buscar películas por similitud visual")
selected_id = st.selectbox("Selecciona una película (imdbId):", umap_df[id_col].unique())
ref = umap_df.loc[umap_df[id_col] == selected_id].iloc[0]
cluster_ref = ref["Cluster_DBSCAN"]

st.write(f"Películas similares (mismo cluster = {cluster_ref}):")
similar = umap_df[umap_df["Cluster_DBSCAN"] == cluster_ref].sample(min(6, len(umap_df)), replace=True)

cols = st.columns(3)
for i, (_, row) in enumerate(similar.iterrows()):
    try:
        poster_url = movie_df.loc[movie_df[id_col] == row[id_col], "Poster"].values[0]
        img = Image.open(BytesIO(requests.get(poster_url, timeout=5).content))
        with cols[i % 3]:
            st.image(img, caption=f"{row[genre_col]}", use_column_width=True)
    except Exception:
        continue

# =====================================
# SUBIR IMAGEN
# =====================================
st.subheader("📤 O sube una imagen para encontrar pósters similares")
uploaded = st.file_uploader("Sube una imagen de póster (opcional)", type=["jpg", "png", "jpeg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Imagen subida", width=250)
    st.info(
        "En esta versión demostrativa, la búsqueda por imagen no está implementada, "
        "pero aquí podrías extraer features visuales y comparar con los clusters existentes."
    )