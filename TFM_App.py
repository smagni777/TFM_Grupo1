# ============================================================
# TUI DESTINATION EXPLORER
# Desafío 3 - TFM Grupo 1
#
# Exploración turística basada en:
# - Recursos turísticos
# - Restaurantes
# - POI
# - Comentarios
# - Barrios
# - Información geoespacial
#
# OBJETIVO:
# Transformar el dashboard territorial en una herramienta
# sencilla para explorar un destino a partir de un lugar
# conocido por el usuario.
# ============================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import html
import numpy as np

from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="TUI Destination Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. ESTILO
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fa;
    }

    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    h1, h2, h3 {
        color: #17365D;
    }

    .hero {
        background: #ffffff;
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #e5e9f0;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 750;
        color: #17365D;
        margin-bottom: 5px;
    }

    .hero-subtitle {
        font-size: 16px;
        color: #6b7280;
    }

    .card {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e9f0;
        margin-bottom: 12px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 700;
        color: #17365D;
        margin-bottom: 7px;
    }

    .muted {
        color: #6b7280;
        font-size: 13px;
    }

    .recommendation {
        background: #ffffff;
        border-left: 5px solid #1967D2;
        padding: 16px;
        border-radius: 9px;
        margin-bottom: 12px;
    }

    .insight {
        background: #ffffff;
        border-left: 5px solid #2e8b57;
        padding: 18px;
        border-radius: 9px;
        margin-bottom: 12px;
    }

    .warning {
        background: #fff8e6;
        border-left: 5px solid #e0a800;
        padding: 16px;
        border-radius: 9px;
    }

    .kpi {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e9f0;
        min-height: 110px;
    }

    .kpi-title {
        color: #6b7280;
        font-size: 13px;
        font-weight: 600;
    }

    .kpi-value {
        color: #17365D;
        font-size: 28px;
        font-weight: 750;
    }

    .kpi-desc {
        color: #7b8491;
        font-size: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "Oro"


# ============================================================
# 4. FUNCIONES DE CARGA
# ============================================================

@st.cache_data
def cargar_parquet(nombre):

    ruta = DATA_DIR / nombre

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    return pd.read_parquet(ruta)


@st.cache_data
def cargar_datos():

    df_barrios = cargar_parquet(
        "Barrios.parquet"
    )

    df_restaurantes = cargar_parquet(
        "Restaurantes.parquet"
    )

    df_poi = cargar_parquet(
        "POI.parquet"
    )

    df_recursos = cargar_parquet(
        "Recursos_resenables.parquet"
    )

    df_comentarios = cargar_parquet(
        "Comentarios.parquet"
    )

    geo_path = DATA_DIR / "Barrios.geojson"

    if geo_path.exists():
        gdf_barrios = gpd.read_file(geo_path)
    else:
        gdf_barrios = gpd.GeoDataFrame()

    return (
        df_barrios,
        df_restaurantes,
        df_poi,
        df_recursos,
        df_comentarios,
        gdf_barrios
    )


# ============================================================
# 5. CARGAR DATOS
# ============================================================

try:

    (
        df_barrios,
        df_restaurantes,
        df_poi,
        df_recursos,
        df_comentarios,
        gdf_barrios

    ) = cargar_datos()

except Exception as e:

    st.error(
        "❌ No se pudieron cargar los datos."
    )

    st.code(str(e))

    st.info(
        """
        La aplicación espera esta estructura:

        TFM_Grupo1/
        ├── TFM_App_mapa_versión_final.py
        └── data/
            └── Oro/
                ├── Barrios.parquet
                ├── Restaurantes.parquet
                ├── POI.parquet
                ├── Recursos_resenables.parquet
                ├── Comentarios.parquet
                └── Barrios.geojson
        """
    )

    st.stop()


# ============================================================
# 6. LIMPIEZA DE NOMBRES DE COLUMNAS
# ============================================================

def limpiar_columnas(df):

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


df_barrios = limpiar_columnas(df_barrios)
df_restaurantes = limpiar_columnas(df_restaurantes)
df_poi = limpiar_columnas(df_poi)
df_recursos = limpiar_columnas(df_recursos)
df_comentarios = limpiar_columnas(df_comentarios)


# ============================================================
# 7. FUNCIONES AUXILIARES
# ============================================================

def normalizar_0_1(serie):

    serie = pd.to_numeric(
        serie,
        errors="coerce"
    )

    minimo = serie.min()
    maximo = serie.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(
            0.5,
            index=serie.index
        )

    if minimo == maximo:
        return pd.Series(
            0.5,
            index=serie.index
        )

    return (
        (serie - minimo)
        / (maximo - minimo)
    )


def texto_seguro(valor, defecto="No disponible"):

    if pd.isna(valor):
        return defecto

    texto = str(valor).strip()

    if texto.lower() in [
        "",
        "nan",
        "none",
        "null",
        "<na>"
    ]:
        return defecto

    return html.escape(texto)


def numero_seguro(valor):

    try:

        valor = float(valor)

        if np.isnan(valor):
            return None

        return valor

    except:

        return None


def distancia_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    try:

        R = 6371.0

        lat1 = radians(float(lat1))
        lon1 = radians(float(lon1))

        lat2 = radians(float(lat2))
        lon2 = radians(float(lon2))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2) ** 2
            +
            cos(lat1)
            * cos(lat2)
            * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(
            sqrt(a),
            sqrt(1 - a)
        )

        return R * c

    except:

        return np.nan


def calcular_distancias(
    df,
    lat,
    lon
):

    df = df.copy()

    if "lat" not in df.columns:
        df["distancia_km"] = np.nan
        return df

    if "lon" not in df.columns:
        df["distancia_km"] = np.nan
        return df

    df["distancia_km"] = df.apply(
        lambda row:
        distancia_km(
            lat,
            lon,
            row["lat"],
            row["lon"]
        ),
        axis=1
    )

    return df


def buscar_columna(
    df,
    candidatos
):

    columnas = {
        str(c).lower().strip(): c
        for c in df.columns
    }

    for candidato in candidatos:

        if candidato.lower() in columnas:

            return columnas[
                candidato.lower()
            ]

    for candidato in candidatos:

        for col_lower, col_original in columnas.items():

            if candidato.lower() in col_lower:

                return col_original

    return None


# ============================================================
# 8. NORMALIZACIÓN GEOESPACIAL
# ============================================================

def preparar_geo(df):

    df = df.copy()

    lat_col = buscar_columna(
        df,
        [
            "lat",
            "latitude",
            "latitud"
        ]
    )

    lon_col = buscar_columna(
        df,
        [
            "lon",
            "lng",
            "longitude",
            "longitud"
        ]
    )

    nombre_col = buscar_columna(
        df,
        [
            "nombre",
            "name",
            "titulo",
            "title"
        ]
    )

    categoria_col = buscar_columna(
        df,
        [
            "categoria",
            "category",
            "tipo",
            "type",
            "subcategoria"
        ]
    )

    rating_col = buscar_columna(
        df,
        [
            "rating",
            "valoracion",
            "valoración",
            "puntuacion",
            "puntuación",
            "estrellas"
        ]
    )

    if lat_col:
        df["lat"] = pd.to_numeric(
            df[lat_col],
            errors="coerce"
        )

    else:
        df["lat"] = np.nan


    if lon_col:
        df["lon"] = pd.to_numeric(
            df[lon_col],
            errors="coerce"
        )

    else:
        df["lon"] = np.nan


    if nombre_col:
        df["nombre_std"] = (
            df[nombre_col]
            .fillna("Sin nombre")
            .astype(str)
        )

    else:
        df["nombre_std"] = "Sin nombre"


    if categoria_col:

        df["categoria_std"] = (
            df[categoria_col]
            .fillna("Sin categoría")
            .astype(str)
        )

    else:

        df["categoria_std"] = "Sin categoría"


    if rating_col:

        df["rating_std"] = pd.to_numeric(
            df[rating_col],
            errors="coerce"
        )

    else:

        df["rating_std"] = np.nan


    return df


df_restaurantes = preparar_geo(
    df_restaurantes
)

df_poi = preparar_geo(
    df_poi
)

df_recursos = preparar_geo(
    df_recursos
)


# ============================================================
# 9. TERRAZA
# ============================================================

col_terraza = buscar_columna(
    df_restaurantes,
    [
        "tiene_terraza",
        "terraza",
        "terrace",
        "outdoor"
    ]
)

if col_terraza:

    df_restaurantes["terraza_std"] = (
        df_restaurantes[col_terraza]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "si",
                "sí",
                "yes"
            ]
        )
    )

else:

    df_restaurantes["terraza_std"] = False


# ============================================================
# 10. CABECERA
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🌍 TUI Destination Explorer
        </div>

        <div class="hero-subtitle">
            Descubre un destino a partir de cualquier lugar
            que conozcas.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## 🌍 TUI Destination Explorer"
)

st.sidebar.caption(
    "Exploración turística georreferenciada"
)

st.sidebar.markdown("---")


# ============================================================
# 12. BUSCADOR DE DESTINO
# ============================================================

st.sidebar.markdown(
    "### 🔎 1. ¿Dónde quieres ir?"
)

nombres = sorted(
    df_recursos[
        "nombre_std"
    ]
    .dropna()
    .astype(str)
    .unique()
)


busqueda = st.sidebar.text_input(
    "Busca un lugar",
    placeholder="Ej.: Museo del Prado"
)


if busqueda.strip():

    candidatos = [
        nombre
        for nombre in nombres
        if busqueda.lower()
        in nombre.lower()
    ]

else:

    candidatos = nombres


if candidatos:

    lugar_seleccionado = st.sidebar.selectbox(
        "Selecciona el lugar",
        candidatos
    )

else:

    st.sidebar.warning(
        "No se ha encontrado ese lugar."
    )

    lugar_seleccionado = None


# ============================================================
# 13. RADIO
# ============================================================

st.sidebar.markdown(
    "### 📍 2. Radio de exploración"
)

radio = st.sidebar.select_slider(
    "Distancia máxima",
    options=[
        0.5,
        1.0,
        2.0,
        5.0
    ],
    value=1.0,
    format_func=lambda x:
    f"{x:g} km"
)


# ============================================================
# 14. FILTROS
# ============================================================

st.sidebar.markdown(
    "### 🍽️ 3. Preferencias"
)

solo_terraza = st.sidebar.checkbox(
    "🍷 Restaurantes con terraza"
)


rating_min = st.sidebar.slider(
    "⭐ Valoración mínima",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)


# ============================================================
# 15. RESET
# ============================================================

if st.sidebar.button(
    "🔄 Restablecer filtros"
):

    st.rerun()


# ============================================================
# 16. IDENTIFICAR LUGAR
# ============================================================

if lugar_seleccionado is None:

    st.info(
        "👈 Utiliza el buscador para comenzar."
    )

    st.markdown(
        """
        ### ¿Cómo funciona?

        **1. Elige un lugar**

        Por ejemplo: un museo, monumento,
        atracción o recurso turístico.

        **2. Define el radio**

        El sistema analiza la oferta disponible
        alrededor del punto seleccionado.

        **3. Descubre el destino**

        Consulta restaurantes, actividades,
        valoraciones, distancias y oportunidades.

        **4. Explora el mapa**

        Toda la información queda integrada
        geográficamente.
        """
    )

    st.stop()


# ============================================================
# 17. PUNTO CENTRAL
# ============================================================

seleccion_df = df_recursos[
    df_recursos["nombre_std"].astype(str).str.lower()
    ==
    lugar_seleccionado.lower()
].copy()


if seleccion_df.empty:

    st.error(
        "No se pudo localizar geográficamente "
        "el recurso seleccionado."
    )

    st.stop()


poi_central = seleccion_df.iloc[0]


lat_central = poi_central["lat"]
lon_central = poi_central["lon"]


if pd.isna(lat_central) or pd.isna(lon_central):

    st.error(
        "El lugar seleccionado no tiene coordenadas válidas."
    )

    st.stop()


categoria_central = (
    poi_central["categoria_std"]
)


rating_central = numero_seguro(
    poi_central["rating_std"]
)


# ============================================================
# 18. RESTAURANTES CERCANOS
# ============================================================

rest_cercanos = calcular_distancias(
    df_restaurantes,
    lat_central,
    lon_central
)


rest_cercanos = rest_cercanos[
    rest_cercanos["distancia_km"]
    <= radio
].copy()


# Rating

rest_cercanos = rest_cercanos[
    rest_cercanos["rating_std"].fillna(0)
    >= rating_min
]


# Terraza

if solo_terraza:

    rest_cercanos = rest_cercanos[
        rest_cercanos["terraza_std"]
    ]


# ============================================================
# 19. SCORE DE RECOMENDACIÓN
# ============================================================

if not rest_cercanos.empty:

    rating_norm = (
        rest_cercanos["rating_std"]
        .fillna(0)
        / 5
    )

    proximidad_norm = (
        1
        /
        (
            1
            +
            rest_cercanos[
                "distancia_km"
            ]
        )
    )

    rest_cercanos["score_recomendacion"] = (
        0.70 * rating_norm
        +
        0.30 * proximidad_norm
    )

    rest_cercanos = (
        rest_cercanos
        .sort_values(
            "score_recomendacion",
            ascending=False
        )
    )


# ============================================================
# 20. ACTIVIDADES CERCANAS
# ============================================================

actividades = calcular_distancias(
    df_recursos,
    lat_central,
    lon_central
)


actividades = actividades[
    actividades["distancia_km"]
    <= radio
].copy()


# Excluir el lugar principal

actividades = actividades[
    actividades["nombre_std"].astype(str).str.lower()
    !=
    lugar_seleccionado.lower()
]


# ============================================================
# 21. DIVERSIDAD DE ACTIVIDADES
# ============================================================

actividades_seleccionadas = []

categorias_vistas = set()


for _, fila in actividades.sort_values(
    "distancia_km"
).iterrows():

    categoria = str(
        fila["categoria_std"]
    )

    if categoria not in categorias_vistas:

        actividades_seleccionadas.append(
            fila
        )

        categorias_vistas.add(
            categoria
        )

    if len(actividades_seleccionadas) >= 6:
        break


if actividades_seleccionadas:

    actividades_recomendadas = pd.DataFrame(
        actividades_seleccionadas
    )

else:

    actividades_recomendadas = pd.DataFrame(
        columns=actividades.columns
    )


# ============================================================
# 22. INFORMACIÓN PRINCIPAL
# ============================================================

st.markdown(
    f"""
    <div class="card">

        <div class="card-title">
            📍 {texto_seguro(lugar_seleccionado)}
        </div>

        <div class="muted">
            {texto_seguro(categoria_central)}
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 23. KPIs
# ============================================================

k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "📍 Lugar",
        "Seleccionado"
    )


with k2:

    st.metric(
        "🍽️ Restaurantes",
        len(rest_cercanos)
    )


with k3:

    st.metric(
        "🎭 Actividades",
        len(actividades_recomendadas)
    )


with k4:

    if rating_central is not None:

        valor_rating = (
            f"{rating_central:.1f} ⭐"
        )

    else:

        valor_rating = "N/D"

    st.metric(
        "⭐ Valoración",
        valor_rating
    )


# ============================================================
# 24. TABS
# ============================================================

tab_mapa, tab_comer, tab_descubrir, tab_inteligencia = st.tabs(
    [
        "🗺️ Mapa",
        "🍽️ Comer",
        "🎭 Descubrir",
        "💡 Inteligencia TUI"
    ]
)


# ============================================================
# 25. MAPA
# ============================================================

with tab_mapa:

    st.markdown(
        "## 🗺️ Explora el entorno"
    )

    st.caption(
        f"Oferta disponible en un radio de {radio:g} km."
    )


    mapa = folium.Map(
        location=[
            lat_central,
            lon_central
        ],
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True
    )


    # --------------------------------------------------------
    # RADIO
    # --------------------------------------------------------

    folium.Circle(
        location=[
            lat_central,
            lon_central
        ],
        radius=radio * 1000,
        color="#1967D2",
        fill=True,
        fill_opacity=0.08,
        tooltip=f"Radio: {radio:g} km"
    ).add_to(mapa)


    # --------------------------------------------------------
    # LUGAR CENTRAL
    # --------------------------------------------------------

    popup_central = f"""
    <b>{texto_seguro(lugar_seleccionado)}</b><br>
    Categoría: {texto_seguro(categoria_central)}<br>
    Valoración:
    {
        f"{rating_central:.1f} ⭐"
        if rating_central is not None
        else "No disponible"
    }
    """


    folium.Marker(
        [
            lat_central,
            lon_central
        ],
        popup=folium.Popup(
            popup_central,
            max_width=320
        ),
        tooltip="📍 Lugar seleccionado",
        icon=folium.Icon(
            color="red",
            icon="star"
        )
    ).add_to(mapa)


    # --------------------------------------------------------
    # RESTAURANTES
    # --------------------------------------------------------

    cluster_rest = MarkerCluster(
        name="🍽️ Restaurantes"
    ).add_to(mapa)


    for _, fila in rest_cercanos.head(80).iterrows():

        nombre = texto_seguro(
            fila["nombre_std"],
            "Restaurante"
        )

        rating = numero_seguro(
            fila["rating_std"]
        )

        distancia = fila[
            "distancia_km"
        ]


        rating_texto = (
            f"{rating:.1f} ⭐"
            if rating is not None
            else "No disponible"
        )


        terraza_texto = (
            "Sí"
            if fila["terraza_std"]
            else "No / no disponible"
        )


        popup = f"""
        <b>🍽️ {nombre}</b><br>
        Valoración: {rating_texto}<br>
        Distancia: {distancia:.2f} km<br>
        Terraza: {terraza_texto}
        """


        folium.Marker(
            [
                fila["lat"],
                fila["lon"]
            ],
            tooltip=f"🍽️ {nombre}",
            popup=folium.Popup(
                popup,
                max_width=320
            ),
            icon=folium.Icon(
                color="orange",
                icon="cutlery"
            )
        ).add_to(cluster_rest)


    # --------------------------------------------------------
    # ACTIVIDADES
    # --------------------------------------------------------

    cluster_act = MarkerCluster(
        name="🎭 Actividades"
    ).add_to(mapa)


    for _, fila in actividades_recomendadas.iterrows():

        nombre = texto_seguro(
            fila["nombre_std"],
            "Actividad"
        )

        categoria = texto_seguro(
            fila["categoria_std"],
            "Recurso turístico"
        )

        distancia = fila[
            "distancia_km"
        ]


        popup = f"""
        <b>🎭 {nombre}</b><br>
        Categoría: {categoria}<br>
        Distancia: {distancia:.2f} km
        """


        folium.Marker(
            [
                fila["lat"],
                fila["lon"]
            ],
            tooltip=f"🎭 {nombre}",
            popup=folium.Popup(
                popup,
                max_width=320
            ),
            icon=folium.Icon(
                color="blue",
                icon="camera"
            )
        ).add_to(cluster_act)


    folium.LayerControl(
        collapsed=False
    ).add_to(mapa)


    st_folium(
        mapa,
        width=None,
        height=650,
        returned_objects=[]
    )


# ============================================================
# 26. COMER
# ============================================================

with tab_comer:

    st.markdown(
        "## 🍽️ Dónde comer"
    )

    st.caption(
        "Las recomendaciones combinan proximidad y valoración."
    )


    if rest_cercanos.empty:

        st.warning(
            "No se encontraron restaurantes con "
            "los filtros seleccionados."
        )

    else:

        top_restaurantes = (
            rest_cercanos
            .head(6)
        )


        for _, fila in top_restaurantes.iterrows():

            nombre = texto_seguro(
                fila["nombre_std"],
                "Restaurante"
            )

            rating = numero_seguro(
                fila["rating_std"]
            )

            distancia = fila[
                "distancia_km"
            ]


            rating_texto = (
                f"{rating:.1f} ⭐"
                if rating is not None
                else "No disponible"
            )


            terraza_texto = (
                "Sí"
                if fila["terraza_std"]
                else "No disponible"
            )


            st.markdown(
                f"""
                <div class="recommendation">

                    <div class="card-title">
                        🍽️ {nombre}
                    </div>

                    <b>⭐ Valoración:</b>
                    {rating_texto}

                    &nbsp;&nbsp;|&nbsp;&nbsp;

                    <b>📏 Distancia:</b>
                    {distancia:.2f} km

                    &nbsp;&nbsp;|&nbsp;&nbsp;

                    <b>🍷 Terraza:</b>
                    {terraza_texto}

                    <br><br>

                    <span class="muted">
                    Recomendado por la combinación de
                    proximidad y valoración disponible.
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 27. DESCUBRIR
# ============================================================

with tab_descubrir:

    st.markdown(
        "## 🎭 Completa tu experiencia"
    )

    st.caption(
        "Actividades diferentes al recurso principal "
        "para favorecer una experiencia más diversificada."
    )


    if actividades_recomendadas.empty:

        st.warning(
            "No se encontraron recursos complementarios "
            "en el radio seleccionado."
        )

    else:

        for _, fila in actividades_recomendadas.iterrows():

            nombre = texto_seguro(
                fila["nombre_std"],
                "Actividad"
            )

            categoria = texto_seguro(
                fila["categoria_std"],
                "Recurso turístico"
            )

            distancia = fila[
                "distancia_km"
            ]

            rating = numero_seguro(
                fila["rating_std"]
            )


            rating_texto = (
                f"{rating:.1f} ⭐"
                if rating is not None
                else "No disponible"
            )


            st.markdown(
                f"""
                <div class="recommendation">

                    <div class="card-title">
                        🎭 {nombre}
                    </div>

                    <b>Categoría:</b>
                    {categoria}

                    &nbsp;&nbsp;|&nbsp;&nbsp;

                    <b>📏 Distancia:</b>
                    {distancia:.2f} km

                    &nbsp;&nbsp;|&nbsp;&nbsp;

                    <b>⭐ Valoración:</b>
                    {rating_texto}

                    <br><br>

                    <span class="muted">
                    Recurso próximo al punto seleccionado
                    que puede complementar la visita.
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 28. INTELIGENCIA TUI
# ============================================================

with tab_inteligencia:

    st.markdown(
        "## 💡 Inteligencia TUI"
    )

    st.caption(
        "Lectura territorial derivada de la oferta "
        "identificada alrededor del recurso."
    )


    total_oferta = (
        len(rest_cercanos)
        +
        len(actividades_recomendadas)
    )


    categorias_actividad = 0

    if (
        not actividades_recomendadas.empty
        and "categoria_std"
        in actividades_recomendadas.columns
    ):

        categorias_actividad = (
            actividades_recomendadas[
                "categoria_std"
            ]
            .nunique()
        )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Oferta identificada",
            total_oferta
        )


    with c2:

        st.metric(
            "Categorías complementarias",
            categorias_actividad
        )


    with c3:

        if not rest_cercanos.empty:

            distancia_media = (
                rest_cercanos[
                    "distancia_km"
                ]
                .mean()
            )

        else:

            distancia_media = 0


        st.metric(
            "Distancia media restaurantes",
            f"{distancia_media:.2f} km"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # LECTURA AUTOMÁTICA
    # --------------------------------------------------------

    if total_oferta == 0:

        st.markdown(
            """
            <div class="warning">

            <b>Oferta limitada identificada</b>

            <br><br>

            El conjunto de datos no muestra
            suficiente oferta complementaria
            alrededor del recurso seleccionado.

            </div>
            """,
            unsafe_allow_html=True
        )


    elif total_oferta < 10:

        st.markdown(
            """
            <div class="insight">

            <b>Oferta complementaria moderada</b>

            <br><br>

            El entorno dispone de una oferta
            complementaria relativamente limitada.
            Puede ser interesante estudiar recursos
            adicionales para aumentar la diversidad
            de la experiencia.

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        st.markdown(
            """
            <div class="insight">

            <b>Entorno con elevada oferta complementaria</b>

            <br><br>

            El recurso seleccionado se encuentra
            integrado en un entorno con múltiples
            alternativas de restauración y actividades.

            Esto facilita el diseño de experiencias
            turísticas combinadas.

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DIVERSIDAD
    # --------------------------------------------------------

    if categorias_actividad >= 3:

        st.markdown(
            """
            <div class="insight">

            <b>🎯 Diversificación de la experiencia</b>

            <br><br>

            Se identifican al menos tres categorías
            diferentes de actividades complementarias.

            Esto puede favorecer una experiencia
            menos dependiente del recurso principal.

            </div>
            """,
            unsafe_allow_html=True
        )


    elif categorias_actividad > 0:

        st.markdown(
            """
            <div class="warning">

            <b>🎯 Diversificación moderada</b>

            <br><br>

            Existe oferta complementaria,
            aunque concentrada en pocas categorías.

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # LIMITACIÓN METODOLÓGICA
    # --------------------------------------------------------

    st.markdown("---")

    st.caption(
        """
        Las recomendaciones se generan exclusivamente a partir
        de la información disponible en los datasets del proyecto.
        La proximidad y la valoración se utilizan como criterios
        de priorización. No se afirma causalidad ni se interpreta
        automáticamente una mayor concentración de oferta como
        una oportunidad económica.
        """
    )


# ============================================================
# 29. DIAGNÓSTICO DE DATOS
# ============================================================

with st.expander(
    "🔧 Información técnica del prototipo"
):

    st.write(
        "Directorio de datos:"
    )

    st.code(
        str(DATA_DIR)
    )


    st.write(
        "Registros cargados:"
    )


    st.write(
        {
            "Barrios": len(df_barrios),
            "Restaurantes": len(df_restaurantes),
            "POI": len(df_poi),
            "Recursos reseñables": len(df_recursos),
            "Comentarios": len(df_comentarios)
        }
    )


    st.write(
        "Columnas de Recursos:"
    )

    st.write(
        df_recursos.columns.tolist()
    )


    st.write(
        "Columnas de Restaurantes:"
    )

    st.write(
        df_restaurantes.columns.tolist()
    )


# ============================================================
# 30. PIE
# ============================================================

st.markdown("---")

st.caption(
    "TUI Destination Explorer · Desafío 3 · "
    "TFM Grupo 1 · Prototipo académico"
)