# ============================================================
# TUI DESTINATION EXPLORER
# Desafío 3 - TUI
# Máster Data Science, Big Data & Business Analytics
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium

from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from math import radians, sin, cos, sqrt, atan2


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

st.markdown("""
<style>

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 1.5rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 650;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

.card {
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e5e5e5;
    background-color: #ffffff;
    margin-bottom: 0.8rem;
}

.metric-card {
    padding: 1rem;
    border-radius: 12px;
    background-color: #f7f8fa;
    border: 1px solid #e6e6e6;
    text-align: center;
}

.small-text {
    color: #666;
    font-size: 0.9rem;
}

.recommendation {
    padding: 0.8rem;
    border-left: 4px solid #2c7be5;
    background-color: #f7faff;
    margin-bottom: 0.7rem;
    border-radius: 6px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def cargar_parquet(ruta):
    """
    Carga un parquet.
    Si existe algún problema, devuelve DataFrame vacío.
    """
    try:
        return pd.read_parquet(ruta)
    except Exception as e:
        st.warning(f"No se pudo cargar {ruta}: {e}")
        return pd.DataFrame()


def encontrar_columna(df, candidatos):
    """
    Busca automáticamente una columna entre diferentes nombres posibles.
    """
    if df.empty:
        return None

    columnas = {str(c).lower().strip(): c for c in df.columns}

    # Coincidencia exacta
    for candidato in candidatos:
        if candidato.lower() in columnas:
            return columnas[candidato.lower()]

    # Coincidencia parcial
    for candidato in candidatos:
        for col_lower, col_original in columnas.items():
            if candidato.lower() in col_lower:
                return col_original

    return None


def normalizar_0_1(serie):
    """
    Normalización Min-Max.
    """
    serie = pd.to_numeric(serie, errors="coerce")

    if serie.dropna().empty:
        return pd.Series(np.nan, index=serie.index)

    minimo = serie.min()
    maximo = serie.max()

    if maximo == minimo:
        return pd.Series(0.5, index=serie.index)

    return (serie - minimo) / (maximo - minimo)


def haversine(lat1, lon1, lat2, lon2):
    """
    Distancia aproximada entre dos puntos en kilómetros.
    """

    R = 6371.0

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def calcular_distancias(df, lat, lon, lat_col, lon_col):

    df = df.copy()

    if lat_col is None or lon_col is None:
        df["distancia_km"] = np.nan
        return df

    df["distancia_km"] = df.apply(
        lambda row: haversine(
            lat,
            lon,
            row[lat_col],
            row[lon_col]
        )
        if pd.notna(row[lat_col]) and pd.notna(row[lon_col])
        else np.nan,
        axis=1
    )

    return df


def mostrar_valor(valor, sufijo=""):

    if pd.isna(valor):
        return "No disponible"

    try:
        return f"{float(valor):.1f}{sufijo}"
    except:
        return str(valor)


# ============================================================
# 4. CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():

    restaurantes = cargar_parquet("Restaurantes.parquet")
    recursos = cargar_parquet("Recursos_resenables.parquet")
    comentarios = cargar_parquet("Comentarios.parquet")

    # Datos opcionales del proyecto anterior
    try:
        barrios = cargar_parquet("Barrios.parquet")
    except:
        barrios = pd.DataFrame()

    return restaurantes, recursos, comentarios, barrios


df_restaurantes, df_recursos, df_comentarios, df_barrios = cargar_datos()


# ============================================================
# 5. IDENTIFICAR COLUMNAS
# ============================================================

def identificar_columnas(df):

    return {
        "nombre": encontrar_columna(
            df,
            [
                "nombre",
                "name",
                "nombre_poi",
                "poi",
                "titulo",
                "title"
            ]
        ),

        "lat": encontrar_columna(
            df,
            [
                "latitud",
                "latitude",
                "lat",
                "y"
            ]
        ),

        "lon": encontrar_columna(
            df,
            [
                "longitud",
                "longitude",
                "lon",
                "lng",
                "x"
            ]
        ),

        "categoria": encontrar_columna(
            df,
            [
                "categoria",
                "category",
                "tipo",
                "type",
                "subcategoria"
            ]
        ),

        "rating": encontrar_columna(
            df,
            [
                "rating",
                "valoracion",
                "valoración",
                "puntuacion",
                "puntuación",
                "stars",
                "estrellas"
            ]
        ),

        "terraza": encontrar_columna(
            df,
            [
                "terraza",
                "terrace",
                "outdoor",
                "exterior"
            ]
        ),

        "precio": encontrar_columna(
            df,
            [
                "precio",
                "price",
                "price_level",
                "nivel_precio"
            ]
        )
    }


cols_rest = identificar_columnas(df_restaurantes)
cols_rec = identificar_columnas(df_recursos)


# ============================================================
# 6. NORMALIZACIÓN DE DATOS
# ============================================================

def preparar_dataframe(df, cols):

    if df.empty:
        return df

    df = df.copy()

    if cols["lat"]:
        df[cols["lat"]] = pd.to_numeric(
            df[cols["lat"]],
            errors="coerce"
        )

    if cols["lon"]:
        df[cols["lon"]] = pd.to_numeric(
            df[cols["lon"]],
            errors="coerce"
        )

    if cols["rating"]:
        df[cols["rating"]] = pd.to_numeric(
            df[cols["rating"]],
            errors="coerce"
        )

    return df


df_restaurantes = preparar_dataframe(
    df_restaurantes,
    cols_rest
)

df_recursos = preparar_dataframe(
    df_recursos,
    cols_rec
)


# ============================================================
# 7. CABECERA
# ============================================================

st.markdown(
    '<div class="main-title">🌍 TUI Destination Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Descubre un destino a partir de cualquier lugar que conozcas.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.title("Explorar destino")

st.sidebar.markdown(
    "Busca un lugar de interés y descubre qué puedes hacer "
    "a su alrededor."
)


# ============================================================
# 9. PREPARAR BUSCADOR DE POI
# ============================================================

nombre_rec = cols_rec["nombre"]

if nombre_rec:

    lista_poi = (
        df_recursos[nombre_rec]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

else:
    lista_poi = []


# ============================================================
# 10. BUSCADOR
# ============================================================

if lista_poi:

    busqueda = st.sidebar.text_input(
        "🔎 ¿Dónde quieres ir?",
        placeholder="Ej.: Museo del Prado"
    )

    resultados_busqueda = []

    if busqueda:

        resultados_busqueda = [
            x for x in lista_poi
            if busqueda.lower() in x.lower()
        ]

    if resultados_busqueda:

        seleccion = st.sidebar.selectbox(
            "Selecciona el lugar",
            resultados_busqueda
        )

    else:

        seleccion = st.sidebar.selectbox(
            "Selecciona un lugar",
            lista_poi
        )

else:

    seleccion = None

    st.sidebar.warning(
        "No se encontró una columna de nombre reconocible "
        "en Recursos_resenables.parquet."
    )


# ============================================================
# 11. RADIO
# ============================================================

radio = st.sidebar.select_slider(
    "📍 Radio de exploración",
    options=[
        0.5,
        1.0,
        2.0,
        5.0
    ],
    value=1.0,
    format_func=lambda x: f"{x:g} km"
)


# ============================================================
# 12. FILTROS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("Preferencias")

solo_terraza = st.sidebar.checkbox(
    "🍷 Solo restaurantes con terraza"
)

rating_min = st.sidebar.slider(
    "⭐ Valoración mínima",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5
)


# ============================================================
# 13. IDENTIFICAR POI SELECCIONADO
# ============================================================

poi_seleccionado = None

if seleccion and nombre_rec:

    coincidencias = df_recursos[
        df_recursos[nombre_rec].astype(str).str.lower()
        == seleccion.lower()
    ]

    if not coincidencias.empty:
        poi_seleccionado = coincidencias.iloc[0]


# ============================================================
# 14. SI NO HAY POI
# ============================================================

if poi_seleccionado is None:

    st.info(
        "👈 Selecciona un lugar en el panel izquierdo "
        "para comenzar la exploración."
    )

    st.markdown("### ¿Cómo funciona?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
            <h3>1️⃣ Elige un lugar</h3>
            <p>Puede ser un museo, monumento,
            atracción o recurso turístico.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
            <h3>2️⃣ Define el radio</h3>
            <p>Explora qué existe alrededor
            del punto seleccionado.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
            <h3>3️⃣ Descubre</h3>
            <p>Alojamiento, gastronomía
            y actividades complementarias.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# ============================================================
# 15. COORDENADAS DEL POI
# ============================================================

lat_rec = cols_rec["lat"]
lon_rec = cols_rec["lon"]

if lat_rec is None or lon_rec is None:

    st.error(
        "El recurso seleccionado no tiene columnas reconocibles "
        "de latitud/longitud."
    )

    st.stop()


lat_poi = poi_seleccionado[lat_rec]
lon_poi = poi_seleccionado[lon_rec]


if pd.isna(lat_poi) or pd.isna(lon_poi):

    st.error(
        "El lugar seleccionado no dispone de coordenadas válidas."
    )

    st.stop()


# ============================================================
# 16. INFORMACIÓN DEL POI
# ============================================================

categoria_poi = "Recurso turístico"

if cols_rec["categoria"]:

    categoria_poi = str(
        poi_seleccionado[cols_rec["categoria"]]
    )


rating_poi = None

if cols_rec["rating"]:

    rating_poi = poi_seleccionado[cols_rec["rating"]]


st.markdown(
    f"""
    <div class="card">
        <h2>📍 {seleccion}</h2>
        <p>
        <b>Categoría:</b> {categoria_poi}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <b>Valoración:</b> {mostrar_valor(rating_poi, " ⭐")}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 17. DISTANCIAS
# ============================================================

df_rest_cerca = calcular_distancias(
    df_restaurantes,
    lat_poi,
    lon_poi,
    cols_rest["lat"],
    cols_rest["lon"]
)

df_rec_cerca = calcular_distancias(
    df_recursos,
    lat_poi,
    lon_poi,
    cols_rec["lat"],
    cols_rec["lon"]
)


# ============================================================
# 18. FILTRAR RADIO
# ============================================================

df_rest_cerca = df_rest_cerca[
    df_rest_cerca["distancia_km"] <= radio
].copy()


df_rec_cerca = df_rec_cerca[
    df_rec_cerca["distancia_km"] <= radio
].copy()


# ============================================================
# 19. FILTROS RESTAURANTES
# ============================================================

if cols_rest["rating"]:

    df_rest_cerca = df_rest_cerca[
        (
            pd.to_numeric(
                df_rest_cerca[cols_rest["rating"]],
                errors="coerce"
            )
            >= rating_min
        )
    ]


if solo_terraza and cols_rest["terraza"]:

    valores_terraza = (
        df_rest_cerca[cols_rest["terraza"]]
        .astype(str)
        .str.lower()
    )

    df_rest_cerca = df_rest_cerca[
        valores_terraza.isin(
            [
                "true",
                "1",
                "sí",
                "si",
                "yes",
                "true "
            ]
        )
    ]

elif solo_terraza and not cols_rest["terraza"]:

    st.sidebar.warning(
        "El dataset de restaurantes no contiene "
        "un campo reconocible de terraza."
    )


# ============================================================
# 20. ORDENAR RESTAURANTES
# ============================================================

if cols_rest["rating"]:

    df_rest_cerca["_rating_num"] = pd.to_numeric(
        df_rest_cerca[cols_rest["rating"]],
        errors="coerce"
    )

    df_rest_cerca["_rating_num"] = (
        df_rest_cerca["_rating_num"].fillna(0)
    )

    # Equilibrio entre valoración y proximidad
    df_rest_cerca["_score"] = (
        df_rest_cerca["_rating_num"] * 0.7
        +
        (
            1
            /
            (1 + df_rest_cerca["distancia_km"])
        )
        * 0.3
    )

    df_rest_cerca = df_rest_cerca.sort_values(
        "_score",
        ascending=False
    )

else:

    df_rest_cerca = df_rest_cerca.sort_values(
        "distancia_km"
    )


# ============================================================
# 21. ACTIVIDADES
# ============================================================

# Quitamos el propio POI
df_actividades = df_rec_cerca.copy()

if nombre_rec:

    df_actividades = df_actividades[
        df_actividades[nombre_rec].astype(str).str.lower()
        != seleccion.lower()
    ]


# Intentar priorizar diversidad de categorías

if cols_rec["categoria"]:

    df_actividades = df_actividades.sort_values(
        "distancia_km"
    )

    seleccionadas = []

    categorias_vistas = set()

    for _, row in df_actividades.iterrows():

        categoria = str(
            row[cols_rec["categoria"]]
        )

        if categoria not in categorias_vistas:

            seleccionadas.append(row)
            categorias_vistas.add(categoria)

        if len(seleccionadas) >= 6:
            break

    if seleccionadas:

        df_actividades = pd.DataFrame(
            seleccionadas
        )

else:

    df_actividades = df_actividades.sort_values(
        "distancia_km"
    )


# ============================================================
# 22. RESUMEN DEL DESTINO
# ============================================================

st.markdown(
    '<div class="section-title">Tu experiencia alrededor del lugar</div>',
    unsafe_allow_html=True
)


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "📍 Punto seleccionado",
        "1"
    )


with c2:

    st.metric(
        "🍽️ Restaurantes",
        len(df_rest_cerca)
    )


with c3:

    st.metric(
        "🎭 Actividades",
        len(df_actividades)
    )


with c4:

    st.metric(
        "📏 Radio",
        f"{radio:g} km"
    )


# ============================================================
# 23. TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🗺️ Mapa",
        "🍽️ Comer",
        "🎭 Descubrir",
        "💡 Inteligencia"
    ]
)


# ============================================================
# TAB 1 - MAPA
# ============================================================

with tab1:

    st.markdown(
        "### Explora el entorno"
    )

    st.caption(
        f"Todo lo que aparece en el mapa se encuentra "
        f"a menos de {radio:g} km del lugar seleccionado."
    )

    mapa = folium.Map(
        location=[lat_poi, lon_poi],
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True
    )


    # --------------------------------------------------------
    # RADIO
    # --------------------------------------------------------

    folium.Circle(
        location=[lat_poi, lon_poi],
        radius=radio * 1000,
        color="#2c7be5",
        fill=True,
        fill_opacity=0.08,
        popup=f"Radio de exploración: {radio:g} km"
    ).add_to(mapa)


    # --------------------------------------------------------
    # POI PRINCIPAL
    # --------------------------------------------------------

    folium.Marker(
        [lat_poi, lon_poi],
        popup=folium.Popup(
            f"""
            <b>{seleccion}</b><br>
            {categoria_poi}<br>
            Valoración: {mostrar_valor(rating_poi)}
            """,
            max_width=300
        ),
        tooltip="Lugar seleccionado",
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


    if not df_rest_cerca.empty:

        for _, row in df_rest_cerca.head(50).iterrows():

            nombre = (
                str(row[cols_rest["nombre"]])
                if cols_rest["nombre"]
                else "Restaurante"
            )

            rating = (
                mostrar_valor(
                    row[cols_rest["rating"]],
                    " ⭐"
                )
                if cols_rest["rating"]
                else "No disponible"
            )

            distancia = (
                f"{row['distancia_km']:.2f} km"
            )

            folium.Marker(
                [
                    row[cols_rest["lat"]],
                    row[cols_rest["lon"]]
                ],
                tooltip=f"🍽️ {nombre}",
                popup=folium.Popup(
                    f"""
                    <b>{nombre}</b><br>
                    ⭐ {rating}<br>
                    📏 {distancia}
                    """,
                    max_width=300
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


    if not df_actividades.empty:

        for _, row in df_actividades.head(50).iterrows():

            nombre = (
                str(row[cols_rec["nombre"]])
                if cols_rec["nombre"]
                else "Actividad"
            )

            categoria = (
                str(row[cols_rec["categoria"]])
                if cols_rec["categoria"]
                else "Recurso turístico"
            )

            distancia = (
                f"{row['distancia_km']:.2f} km"
            )

            folium.Marker(
                [
                    row[cols_rec["lat"]],
                    row[cols_rec["lon"]]
                ],
                tooltip=f"🎭 {nombre}",
                popup=folium.Popup(
                    f"""
                    <b>{nombre}</b><br>
                    {categoria}<br>
                    📏 {distancia}
                    """,
                    max_width=300
                ),
                icon=folium.Icon(
                    color="blue",
                    icon="camera"
                )
            ).add_to(cluster_act)


    folium.LayerControl().add_to(mapa)


    st_folium(
        mapa,
        width=None,
        height=650,
        returned_objects=[]
    )


# ============================================================
# TAB 2 - COMER
# ============================================================

with tab2:

    st.markdown(
        "### 🍽️ Dónde comer"
    )

    st.caption(
        "Restaurantes cercanos priorizando la combinación "
        "de valoración y proximidad."
    )


    if df_rest_cerca.empty:

        st.info(
            "No se encontraron restaurantes con los "
            "filtros seleccionados."
        )

    else:

        mostrar = df_rest_cerca.head(8)

        for _, row in mostrar.iterrows():

            nombre = (
                str(row[cols_rest["nombre"]])
                if cols_rest["nombre"]
                else "Restaurante"
            )

            rating = (
                mostrar_valor(
                    row[cols_rest["rating"]]
                )
                if cols_rest["rating"]
                else "No disponible"
            )

            distancia = (
                f"{row['distancia_km']:.2f} km"
            )

            terraza = "No disponible"

            if cols_rest["terraza"]:

                valor = str(
                    row[cols_rest["terraza"]]
                ).lower()

                if valor in [
                    "true",
                    "1",
                    "sí",
                    "si",
                    "yes"
                ]:
                    terraza = "Sí"

                elif valor in [
                    "false",
                    "0",
                    "no"
                ]:
                    terraza = "No"


            st.markdown(
                f"""
                <div class="card">
                    <h4>🍽️ {nombre}</h4>
                    <p>
                    ⭐ <b>{rating}</b>
                    &nbsp;&nbsp;
                    📏 <b>{distancia}</b>
                    &nbsp;&nbsp;
                    🍷 Terraza: <b>{terraza}</b>
                    </p>
                    <p class="small-text">
                    Recomendado por su combinación de
                    proximidad y valoración.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TAB 3 - DESCUBRIR
# ============================================================

with tab3:

    st.markdown(
        "### 🎭 Qué hacer después"
    )

    st.caption(
        "Actividades complementarias cercanas al "
        "lugar seleccionado."
    )


    if df_actividades.empty:

        st.info(
            "No se encontraron actividades adicionales "
            "en el radio seleccionado."
        )

    else:

        for _, row in df_actividades.head(6).iterrows():

            nombre = (
                str(row[cols_rec["nombre"]])
                if cols_rec["nombre"]
                else "Actividad"
            )

            categoria = (
                str(row[cols_rec["categoria"]])
                if cols_rec["categoria"]
                else "Recurso turístico"
            )

            distancia = (
                f"{row['distancia_km']:.2f} km"
            )

            rating = (
                mostrar_valor(
                    row[cols_rec["rating"]]
                )
                if cols_rec["rating"]
                else "No disponible"
            )


            st.markdown(
                f"""
                <div class="recommendation">

                <h4>🎭 {nombre}</h4>

                <b>Categoría:</b> {categoria}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Distancia:</b> {distancia}

                &nbsp;&nbsp; | &nbsp;&nbsp;

                <b>Valoración:</b> ⭐ {rating}

                <br><br>

                💡 <b>¿Por qué puede interesarte?</b><br>

                Está próximo al lugar que has seleccionado
                y permite complementar la experiencia turística
                sin desplazamientos largos.

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# TAB 4 - INTELIGENCIA
# ============================================================

with tab4:

    st.markdown(
        "### 💡 Inteligencia de destino"
    )

    st.caption(
        "Una lectura orientada a TUI para entender "
        "la concentración y diversidad de la oferta."
    )


    # --------------------------------------------------------
    # DENSIDAD
    # --------------------------------------------------------

    total_oferta = (
        len(df_rest_cerca)
        +
        len(df_actividades)
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Oferta identificada",
            total_oferta
        )


    with col2:

        if len(df_rest_cerca) > 0:

            densidad_rest = (
                len(df_rest_cerca)
                /
                max(radio ** 2, 0.01)
            )

        else:

            densidad_rest = 0

        st.metric(
            "Restauración / km²",
            f"{densidad_rest:.1f}"
        )


    with col3:

        categorias = 0

        if cols_rec["categoria"] and not df_actividades.empty:

            categorias = (
                df_actividades[
                    cols_rec["categoria"]
                ]
                .nunique()
            )

        st.metric(
            "Tipos de actividad",
            categorias
        )


    # --------------------------------------------------------
    # INTERPRETACIÓN
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "#### Lectura automática"
    )


    if total_oferta == 0:

        st.warning(
            "El entorno presenta poca información disponible "
            "en los datasets analizados."
        )

    elif total_oferta < 10:

        st.info(
            "La zona presenta una oferta relativamente "
            "limitada dentro del radio seleccionado."
        )

    elif total_oferta < 30:

        st.success(
            "La zona presenta una oferta turística "
            "intermedia y permite construir una experiencia "
            "complementaria alrededor del punto principal."
        )

    else:

        st.success(
            "La zona presenta una elevada concentración "
            "de oferta turística. Puede resultar especialmente "
            "interesante para diseñar experiencias combinadas."
        )


    # --------------------------------------------------------
    # DIVERSIDAD
    # --------------------------------------------------------

    if categorias >= 3:

        st.markdown(
            """
            <div class="recommendation">

            <b>🎯 Diversidad de experiencia</b><br>

            Se identifican varias categorías de actividades.
            Esto permite plantear una experiencia turística
            más diversificada alrededor del recurso principal.

            </div>
            """,
            unsafe_allow_html=True
        )

    elif categorias > 0:

        st.markdown(
            """
            <div class="recommendation">

            <b>🎯 Diversidad moderada</b><br>

            Existe oferta complementaria, aunque concentrada
            en un número reducido de categorías.

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 24. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "TUI Destination Explorer · Desafío 3 · "
    "Prototipo académico de análisis y exploración turística"
)