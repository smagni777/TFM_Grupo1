# ============================================================
# TUI DESTINATION EXPLORER
# Desafío 3 - TFM Grupo 1
#
# Exploración de destinos por LUGAR o ZONA
# Fuente principal: dataset_completo_tfm.csv
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium

from pathlib import Path
from math import radians, sin, cos, asin, sqrt

from streamlit_folium import st_folium
from folium.plugins import MarkerCluster


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

.stApp {
    background-color: #f5f7fa;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dfe3e8;
}

h1, h2, h3 {
    color: #172033;
    font-family: "Segoe UI", sans-serif;
}

.hero {
    background: linear-gradient(135deg, #ffffff, #eef6fb);
    padding: 30px;
    border-radius: 18px;
    border: 1px solid #dce6ee;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #172033;
}

.hero-subtitle {
    font-size: 17px;
    color: #5f6b7a;
    margin-top: 8px;
}

.kpi {
    background: white;
    border-radius: 14px;
    padding: 20px;
    border: 1px solid #e2e8f0;
    min-height: 120px;
}

.kpi-title {
    font-size: 12px;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 700;
}

.kpi-value {
    font-size: 30px;
    font-weight: 800;
    color: #0284c7;
    margin-top: 8px;
}

.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #172033;
}

.card-text {
    color: #64748b;
    font-size: 14px;
}

.badge {
    display: inline-block;
    padding: 4px 9px;
    border-radius: 10px;
    background: #e0f2fe;
    color: #0369a1;
    font-size: 12px;
    font-weight: 700;
}

.insight {
    background: #ffffff;
    border-left: 5px solid #0284c7;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "Oro"


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def limpiar_columnas(df):
    if df is None:
        return None

    df = df.copy()
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def convertir_numerico(df, columnas):
    df = df.copy()

    for col in columnas:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def normalizar_texto(valor):
    if pd.isna(valor):
        return ""

    return (
        str(valor)
        .strip()
        .lower()
    )


def distancia_haversine(lat1, lon1, lat2, lon2):
    """
    Distancia aproximada en metros entre dos coordenadas.
    """

    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan

    lon1, lat1, lon2, lat2 = map(
        radians,
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * asin(sqrt(a))

    return 6371000 * c


def formato_rating(valor):
    if pd.isna(valor):
        return "No disponible"

    try:
        return f"{float(valor):.1f} ⭐"
    except Exception:
        return "No disponible"


def mostrar_kpi(titulo, valor):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-title">{titulo}</div>
            <div class="kpi-value">{valor}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 5. CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():

    archivos = {}

    # --------------------------------------------------------
    # CSV UNIFICADO
    # --------------------------------------------------------

    csv_path = DATA_DIR / "dataset_completo_tfm.csv"

    if csv_path.exists():

        try:
            df = pd.read_csv(
                csv_path,
                low_memory=False,
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                csv_path,
                low_memory=False,
                encoding="latin-1"
            )

    else:
        df = None

    # --------------------------------------------------------
    # PARQUETS
    # --------------------------------------------------------

    parquet_files = {
        "restaurantes": DATA_DIR / "Restaurantes.parquet",
        "poi": DATA_DIR / "POI.parquet",
        "barrios": DATA_DIR / "Barrios.parquet",
        "comentarios": DATA_DIR / "Comentarios.parquet",
    }

    for nombre, path in parquet_files.items():

        if path.exists():

            try:
                archivos[nombre] = pd.read_parquet(path)

            except Exception:
                archivos[nombre] = None

        else:
            archivos[nombre] = None

    return (
        limpiar_columnas(df),
        limpiar_columnas(archivos.get("restaurantes")),
        limpiar_columnas(archivos.get("poi")),
        limpiar_columnas(archivos.get("barrios")),
        limpiar_columnas(archivos.get("comentarios")),
    )


# ============================================================
# 6. INICIALIZAR
# ============================================================

df, df_rest, df_poi, df_barrios, df_comentarios = cargar_datos()


if df is None:

    st.error(
        "❌ No se encontró dataset_completo_tfm.csv "
        "en data/Oro/"
    )

    st.stop()


# ============================================================
# 7. PREPARACIÓN DEL DATASET PRINCIPAL
# ============================================================

df = convertir_numerico(
    df,
    [
        "LATITUD",
        "LONGITUD",
        "CALIFICACION",
        "distancia_parada_metros",
        "COORDENADA-X",
        "COORDENADA-Y",
    ]
)


# Coordenadas válidas
df["coord_validas"] = (
    df["LATITUD"].notna()
    & df["LONGITUD"].notna()
)


# Nombre limpio
df["nombre_busqueda"] = (
    df["NOMBRE_RECURSO"]
    .fillna(df.get("NOMBRE", ""))
    .astype(str)
    .str.strip()
)


# Categoría
df["categoria_busqueda"] = (
    df["CATEGORIA_RECURSO"]
    .fillna(df.get("categoria", ""))
    .astype(str)
    .str.strip()
)


# Barrio
df["barrio_busqueda"] = (
    df["BARRIO_RECURSO"]
    .fillna(df.get("BARRIO", ""))
    .astype(str)
    .str.strip()
)


# Distrito
if "DISTRITO" in df.columns:

    df["distrito_busqueda"] = (
        df["DISTRITO"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

else:

    df["distrito_busqueda"] = ""


# ============================================================
# 8. CREAR DATASET DE RECURSOS ÚNICOS
# ============================================================

columnas_recurso = [
    "NOMBRE_RECURSO",
    "CATEGORIA_RECURSO",
    "BARRIO_RECURSO",
    "DISTRITO",
    "LATITUD",
    "LONGITUD",
    "CALIFICACION",
    "DESCRIPCION",
    "HORARIO",
    "EQUIPAMIENTO",
    "ACCESIBILIDAD",
    "distancia_parada_metros",
    "parada_mas_cercana",
    "tipo_transporte_cercano",
]


columnas_recurso = [
    c for c in columnas_recurso
    if c in df.columns
]


recursos = df[columnas_recurso].copy()


recursos = recursos.drop_duplicates(
    subset=[
        c for c in [
            "NOMBRE_RECURSO",
            "LATITUD",
            "LONGITUD"
        ]
        if c in recursos.columns
    ]
)


recursos["nombre_busqueda"] = (
    recursos["NOMBRE_RECURSO"]
    .fillna("")
    .astype(str)
    .str.strip()
)


recursos["categoria_busqueda"] = (
    recursos["CATEGORIA_RECURSO"]
    .fillna("")
    .astype(str)
    .str.strip()
)


recursos["barrio_busqueda"] = (
    recursos["BARRIO_RECURSO"]
    .fillna("")
    .astype(str)
    .str.strip()
)


if "DISTRITO" in recursos.columns:

    recursos["distrito_busqueda"] = (
        recursos["DISTRITO"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

else:

    recursos["distrito_busqueda"] = ""


# ============================================================
# 9. BARRIOS Y DISTRITOS DISPONIBLES
# ============================================================

barrios = sorted(
    [
        x for x in
        recursos["barrio_busqueda"].dropna().unique()
        if str(x).strip()
    ]
)

distritos = sorted(
    [
        x for x in
        recursos["distrito_busqueda"].dropna().unique()
        if str(x).strip()
    ]
)


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
            Explora un destino a partir de un lugar o una zona.
            Descubre qué comer, qué visitar y cómo moverte.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 11. SIDEBAR
# ============================================================

st.sidebar.markdown("## 🔎 Explorar destino")


modo = st.sidebar.radio(
    "¿Qué quieres explorar?",
    [
        "📍 Un lugar",
        "🏙️ Una zona"
    ]
)


# ============================================================
# 12. MODO LUGAR
# ============================================================

lugar_seleccionado = None
zona_seleccionada = None
radio = None


if modo == "📍 Un lugar":

    lugares = sorted(
        [
            x for x in
            recursos["nombre_busqueda"].dropna().unique()
            if str(x).strip()
            and str(x).lower() != "nan"
        ]
    )

    if not lugares:

        st.error(
            "No se encontraron lugares con nombre."
        )

        st.stop()


    lugar_seleccionado = st.sidebar.selectbox(
        "Busca un lugar",
        lugares
    )


    radio = st.sidebar.select_slider(
        "Radio de exploración",
        options=[
            250,
            400,
            600,
            800,
            1000,
            1500,
            2000
        ],
        value=800,
        format_func=lambda x: f"{x} m"
    )


    candidatos = recursos[
        recursos["nombre_busqueda"]
        .str.lower()
        == str(lugar_seleccionado).lower()
    ]


    candidatos = candidatos[
        candidatos["LATITUD"].notna()
        & candidatos["LONGITUD"].notna()
    ]


    if candidatos.empty:

        st.warning(
            "⚠️ El lugar seleccionado no tiene "
            "coordenadas geográficas válidas."
        )

        st.info(
            "Puedes seleccionar otro lugar o utilizar "
            "el modo 'Zona'."
        )

        st.stop()


    lugar = candidatos.iloc[0]

    lat_centro = float(lugar["LATITUD"])
    lon_centro = float(lugar["LONGITUD"])


    # --------------------------------------------------------
    # CALCULAR DISTANCIAS
    # --------------------------------------------------------

    recursos_cercanos = recursos.copy()

    recursos_cercanos["distancia_m"] = recursos_cercanos.apply(
        lambda row: distancia_haversine(
            lat_centro,
            lon_centro,
            row["LATITUD"],
            row["LONGITUD"]
        ),
        axis=1
    )


    recursos_cercanos = recursos_cercanos[
        recursos_cercanos["distancia_m"].notna()
    ]


    recursos_cercanos = recursos_cercanos[
        recursos_cercanos["distancia_m"] <= radio
    ]


# ============================================================
# 13. MODO ZONA
# ============================================================

else:

    st.sidebar.markdown("### 🏙️ Selecciona una zona")


    tipo_zona = st.sidebar.radio(
        "Nivel territorial",
        [
            "Barrio",
            "Distrito"
        ],
        horizontal=True
    )


    if tipo_zona == "Barrio":

        if not barrios:

            st.warning(
                "No se encontraron barrios."
            )

            st.stop()


        zona_seleccionada = st.sidebar.selectbox(
            "Barrio",
            barrios
        )


        recursos_cercanos = recursos[
            recursos["barrio_busqueda"]
            .str.lower()
            == str(zona_seleccionada).lower()
        ].copy()


    else:

        if not distritos:

            st.warning(
                "No se encontraron distritos."
            )

            st.stop()


        zona_seleccionada = st.sidebar.selectbox(
            "Distrito",
            distritos
        )


        recursos_cercanos = recursos[
            recursos["distrito_busqueda"]
            .str.lower()
            == str(zona_seleccionada).lower()
        ].copy()


    lat_validas = pd.to_numeric(
        recursos_cercanos["LATITUD"],
        errors="coerce"
    ).dropna()


    lon_validas = pd.to_numeric(
        recursos_cercanos["LONGITUD"],
        errors="coerce"
    ).dropna()


    if lat_validas.empty or lon_validas.empty:

        st.warning(
            "La zona seleccionada no contiene "
            "coordenadas geográficas suficientes."
        )

        st.stop()


    lat_centro = lat_validas.mean()
    lon_centro = lon_validas.mean()


# ============================================================
# 14. INFORMACIÓN DEL DESTINO
# ============================================================

if modo == "📍 Un lugar":

    titulo_contexto = lugar_seleccionado

else:

    titulo_contexto = zona_seleccionada


st.markdown(
    f"""
    ### 📍 {titulo_contexto}
    """
)


if modo == "📍 Un lugar":

    barrio_contexto = lugar.get(
        "BARRIO_RECURSO",
        "No disponible"
    )

    categoria_contexto = lugar.get(
        "CATEGORIA_RECURSO",
        "No disponible"
    )

    rating_contexto = lugar.get(
        "CALIFICACION",
        np.nan
    )

else:

    barrio_contexto = zona_seleccionada
    categoria_contexto = "Zona territorial"
    rating_contexto = np.nan


# ============================================================
# 15. KPIs
# ============================================================

n_recursos = len(recursos_cercanos)


if "CALIFICACION" in recursos_cercanos.columns:

    ratings = pd.to_numeric(
        recursos_cercanos["CALIFICACION"],
        errors="coerce"
    ).dropna()

else:

    ratings = pd.Series(dtype=float)


rating_medio = (
    ratings.mean()
    if not ratings.empty
    else np.nan
)


categorias = (
    recursos_cercanos["categoria_busqueda"]
    .replace("", np.nan)
    .dropna()
)


n_categorias = categorias.nunique()


if "distancia_parada_metros" in recursos_cercanos.columns:

    transporte = pd.to_numeric(
        recursos_cercanos["distancia_parada_metros"],
        errors="coerce"
    ).dropna()

    transporte_disponible = not transporte.empty

else:

    transporte_disponible = False


k1, k2, k3, k4 = st.columns(4)


with k1:

    mostrar_kpi(
        "Recursos encontrados",
        n_recursos
    )


with k2:

    mostrar_kpi(
        "Categorías",
        n_categorias
    )


with k3:

    mostrar_kpi(
        "Valoración media",
        formato_rating(rating_medio)
    )


with k4:

    mostrar_kpi(
        "Transporte",
        "Disponible"
        if transporte_disponible
        else "No disponible"
    )


# ============================================================
# 16. TABS
# ============================================================

tab_explorar, tab_comer, tab_descubrir, tab_inteligencia = st.tabs(
    [
        "🗺️ Explorar",
        "🍽️ Comer",
        "🏛️ Descubrir",
        "🧠 Inteligencia TUI"
    ]
)


# ============================================================
# 17. TAB EXPLORAR
# ============================================================

with tab_explorar:

    st.markdown("### 🗺️ Mapa del destino")


    mapa = folium.Map(
        location=[
            lat_centro,
            lon_centro
        ],
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True
    )


    # --------------------------------------------------------
    # MARCADOR CENTRAL
    # --------------------------------------------------------

    if modo == "📍 Un lugar":

        folium.Marker(
            [
                lat_centro,
                lon_centro
            ],
            popup=f"""
                <b>{titulo_contexto}</b><br>
                {categoria_contexto}
            """,
            tooltip="Lugar seleccionado",
            icon=folium.Icon(
                color="red",
                icon="star"
            )
        ).add_to(mapa)


        folium.Circle(
            [
                lat_centro,
                lon_centro
            ],
            radius=radio,
            color="#0284c7",
            fill=False,
            weight=2
        ).add_to(mapa)


    else:

        folium.Marker(
            [
                lat_centro,
                lon_centro
            ],
            popup=f"<b>{titulo_contexto}</b>",
            tooltip="Centro aproximado de la zona",
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(mapa)


    # --------------------------------------------------------
    # CAPA DE RECURSOS
    # --------------------------------------------------------

    cluster = MarkerCluster(
        name="Recursos"
    ).add_to(mapa)


    limite_mapa = recursos_cercanos.head(300)


    for _, row in limite_mapa.iterrows():

        lat = row.get("LATITUD")
        lon = row.get("LONGITUD")


        if pd.isna(lat) or pd.isna(lon):
            continue


        nombre = row.get(
            "NOMBRE_RECURSO",
            "Recurso"
        )


        categoria = row.get(
            "CATEGORIA_RECURSO",
            "No disponible"
        )


        rating = formato_rating(
            row.get(
                "CALIFICACION",
                np.nan
            )
        )


        distancia = row.get(
            "distancia_m",
            np.nan
        )


        distancia_txt = (
            f"{distancia:.0f} m"
            if pd.notna(distancia)
            else "Zona seleccionada"
        )


        popup = f"""
        <b>{nombre}</b><br>
        Categoría: {categoria}<br>
        Valoración: {rating}<br>
        Distancia: {distancia_txt}
        """


        folium.Marker(
            [
                lat,
                lon
            ],
            popup=popup,
            tooltip=str(nombre)[:70]
        ).add_to(cluster)


    folium.LayerControl().add_to(mapa)


    st_folium(
        mapa,
        width=1200,
        height=600
    )


# ============================================================
# 18. TAB COMER
# ============================================================

with tab_comer:

    st.markdown(
        "### 🍽️ Restaurantes alrededor del destino"
    )


    # --------------------------------------------------------
    # RESTAURANTES
    # --------------------------------------------------------

    restaurantes = None


    if df_rest is not None and not df_rest.empty:

        restaurantes = df_rest.copy()

    else:

        # Intentar identificar restaurantes
        # dentro del dataset unificado

        mask_rest = (
            df["CATEGORIA_RECURSO"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                "restaurante|restaurant|hosteler"
            )
        )

        restaurantes = df[
            mask_rest
        ].copy()


    if restaurantes is None or restaurantes.empty:

        st.info(
            "No hay datos de restaurantes disponibles."
        )

    else:

        restaurantes = convertir_numerico(
            restaurantes,
            [
                "LATITUD",
                "LONGITUD",
                "CALIFICACION"
            ]
        )


        # ----------------------------------------------------
        # DISTANCIA
        # ----------------------------------------------------

        restaurantes["distancia_m"] = restaurantes.apply(
            lambda row: distancia_haversine(
                lat_centro,
                lon_centro,
                row.get("LATITUD"),
                row.get("LONGITUD")
            ),
            axis=1
        )


        restaurantes = restaurantes[
            restaurantes["distancia_m"].notna()
        ]


        if modo == "📍 Un lugar":

            restaurantes = restaurantes[
                restaurantes["distancia_m"] <= radio
            ]


        # ----------------------------------------------------
        # FILTROS
        # ----------------------------------------------------

        c1, c2 = st.columns(2)


        with c1:

            min_rating = st.slider(
                "Valoración mínima",
                0.0,
                5.0,
                0.0,
                0.5
            )


        with c2:

            terraza_disponible = (
                "tiene_terraza"
                in restaurantes.columns
            )


            if terraza_disponible:

                filtro_terraza = st.selectbox(
                    "Terraza",
                    [
                        "Todas",
                        "Con terraza",
                        "Sin terraza"
                    ]
                )

            else:

                filtro_terraza = "Todas"

                st.caption(
                    "ℹ️ Información de terraza "
                    "no disponible en esta fuente."
                )


        # ----------------------------------------------------
        # RATING
        # ----------------------------------------------------

        if "CALIFICACION" in restaurantes.columns:

            restaurantes["CALIFICACION"] = pd.to_numeric(
                restaurantes["CALIFICACION"],
                errors="coerce"
            )

            restaurantes = restaurantes[
                restaurantes["CALIFICACION"].fillna(0)
                >= min_rating
            ]


        # ----------------------------------------------------
        # TERRAZA
        # ----------------------------------------------------

        if (
            terraza_disponible
            and filtro_terraza != "Todas"
        ):

            if filtro_terraza == "Con terraza":

                restaurantes = restaurantes[
                    restaurantes["tiene_terraza"]
                    .astype(str)
                    .str.lower()
                    .isin([
                        "true",
                        "1",
                        "si",
                        "sí",
                        "yes"
                    ])
                ]

            else:

                restaurantes = restaurantes[
                    ~restaurantes["tiene_terraza"]
                    .astype(str)
                    .str.lower()
                    .isin([
                        "true",
                        "1",
                        "si",
                        "sí",
                        "yes"
                    ])
                ]


        # ----------------------------------------------------
        # ORDEN
        # ----------------------------------------------------

        orden_cols = []


        if "CALIFICACION" in restaurantes.columns:

            orden_cols.append(
                "CALIFICACION"
            )


        orden_cols.append(
            "distancia_m"
        )


        restaurantes = restaurantes.sort_values(
            orden_cols,
            ascending=[
                False if c == "CALIFICACION"
                else True
                for c in orden_cols
            ]
        )


        # ----------------------------------------------------
        # MOSTRAR
        # ----------------------------------------------------

        if restaurantes.empty:

            st.info(
                "No encontramos restaurantes "
                "que cumplan los filtros."
            )

        else:

            st.success(
                f"Se encontraron "
                f"**{len(restaurantes)} restaurantes**."
            )


            for _, row in restaurantes.head(15).iterrows():

                nombre = (
                    row.get(
                        "nombre",
                        row.get(
                            "NOMBRE_RECURSO",
                            "Restaurante"
                        )
                    )
                )


                rating = formato_rating(
                    row.get(
                        "CALIFICACION",
                        np.nan
                    )
                )


                distancia = row.get(
                    "distancia_m",
                    np.nan
                )


                distancia_txt = (
                    f"{distancia:.0f} m"
                    if pd.notna(distancia)
                    else "No disponible"
                )


                terraza_txt = "No disponible"


                if "tiene_terraza" in row.index:

                    valor = str(
                        row["tiene_terraza"]
                    ).lower()


                    if valor in [
                        "true",
                        "1",
                        "si",
                        "sí",
                        "yes"
                    ]:

                        terraza_txt = "Sí"

                    elif valor in [
                        "false",
                        "0",
                        "no"
                    ]:

                        terraza_txt = "No"


                st.markdown(
                    f"""
                    <div class="card">

                        <div class="card-title">
                            🍽️ {nombre}
                        </div>

                        <div class="card-text">

                            ⭐ {rating}
                            &nbsp;&nbsp;|&nbsp;&nbsp;

                            📏 {distancia_txt}
                            &nbsp;&nbsp;|&nbsp;&nbsp;

                            🌿 Terraza: {terraza_txt}

                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# 19. TAB DESCUBRIR
# ============================================================

with tab_descubrir:

    st.markdown(
        "### 🏛️ Descubre qué hacer alrededor"
    )


    descubrir = recursos_cercanos.copy()


    # No mostrar el propio lugar como recomendación
    if modo == "📍 Un lugar":

        descubrir = descubrir[
            descubrir["nombre_busqueda"]
            .str.lower()
            != str(lugar_seleccionado).lower()
        ]


    # --------------------------------------------------------
    # CATEGORÍA
    # --------------------------------------------------------

    categorias_disponibles = sorted(
        [
            x for x in
            descubrir["categoria_busqueda"]
            .dropna()
            .unique()
            if str(x).strip()
        ]
    )


    if categorias_disponibles:

        categoria_filtro = st.multiselect(
            "Filtrar por categoría",
            categorias_disponibles
        )


        if categoria_filtro:

            descubrir = descubrir[
                descubrir["categoria_busqueda"]
                .isin(categoria_filtro)
            ]


    # --------------------------------------------------------
    # ORDEN
    # --------------------------------------------------------

    if "CALIFICACION" in descubrir.columns:

        descubrir["rating_num"] = pd.to_numeric(
            descubrir["CALIFICACION"],
            errors="coerce"
        )

    else:

        descubrir["rating_num"] = np.nan


    descubrir = descubrir.sort_values(
        [
            "rating_num",
            "distancia_m"
        ],
        ascending=[
            False,
            True
        ],
        na_position="last"
    )


    if descubrir.empty:

        st.info(
            "No hay recursos adicionales "
            "con los filtros seleccionados."
        )

    else:

        st.success(
            f"Encontramos "
            f"**{len(descubrir)} recursos** "
            "para explorar."
        )


        for _, row in descubrir.head(12).iterrows():

            nombre = row.get(
                "NOMBRE_RECURSO",
                "Recurso"
            )


            categoria = row.get(
                "CATEGORIA_RECURSO",
                "No disponible"
            )


            rating = formato_rating(
                row.get(
                    "CALIFICACION",
                    np.nan
                )
            )


            distancia = row.get(
                "distancia_m",
                np.nan
            )


            distancia_txt = (
                f"{distancia:.0f} m"
                if pd.notna(distancia)
                else "Dentro de la zona"
            )


            descripcion = row.get(
                "DESCRIPCION",
                ""
            )


            if pd.isna(descripcion):
                descripcion = ""


            descripcion = str(
                descripcion
            ).strip()


            if len(descripcion) > 180:

                descripcion = (
                    descripcion[:180]
                    + "..."
                )


            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        🏛️ {nombre}
                    </div>

                    <div>
                        <span class="badge">
                            {categoria}
                        </span>
                    </div>

                    <div class="card-text">

                        ⭐ {rating}
                        &nbsp;&nbsp;|&nbsp;&nbsp;
                        📏 {distancia_txt}

                    </div>

                    <div class="card-text">
                        {descripcion}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 20. TAB INTELIGENCIA TUI
# ============================================================

with tab_inteligencia:

    st.markdown(
        "### 🧠 Inteligencia TUI"
    )


    st.markdown(
        """
        Esta vista transforma la información georreferenciada
        en indicadores descriptivos para entender el ecosistema
        turístico del entorno seleccionado.
        """
    )


    # --------------------------------------------------------
    # DISTRIBUCIÓN POR CATEGORÍAS
    # --------------------------------------------------------

    if not recursos_cercanos.empty:

        categorias = (
            recursos_cercanos
            ["categoria_busqueda"]
            .replace("", np.nan)
            .dropna()
            .value_counts()
        )


        if not categorias.empty:

            st.markdown(
                "#### Distribución de la oferta"
            )


            st.bar_chart(
                categorias
                .head(10)
            )


    # --------------------------------------------------------
    # RATING
    # --------------------------------------------------------

    st.markdown(
        "#### ⭐ Percepción del entorno"
    )


    if "CALIFICACION" in recursos_cercanos.columns:

        ratings = pd.to_numeric(
            recursos_cercanos["CALIFICACION"],
            errors="coerce"
        ).dropna()


        if not ratings.empty:

            col1, col2, col3 = st.columns(3)


            with col1:

                mostrar_kpi(
                    "Valoración media",
                    f"{ratings.mean():.2f} ⭐"
                )


            with col2:

                mostrar_kpi(
                    "Valoración máxima",
                    f"{ratings.max():.1f} ⭐"
                )


            with col3:

                mostrar_kpi(
                    "Elementos valorados",
                    len(ratings)
                )


        else:

            st.info(
                "No hay suficientes calificaciones "
                "disponibles para calcular este indicador."
            )


    # --------------------------------------------------------
    # TRANSPORTE
    # --------------------------------------------------------

    st.markdown(
        "#### 🚇 Accesibilidad mediante transporte"
    )


    if "distancia_parada_metros" in recursos_cercanos.columns:

        dist_transporte = pd.to_numeric(
            recursos_cercanos[
                "distancia_parada_metros"
            ],
            errors="coerce"
        ).dropna()


        if not dist_transporte.empty:

            c1, c2 = st.columns(2)


            with c1:

                mostrar_kpi(
                    "Distancia media",
                    f"{dist_transporte.mean():.0f} m"
                )


            with c2:

                mostrar_kpi(
                    "Distancia mínima",
                    f"{dist_transporte.min():.0f} m"
                )

        else:

            st.info(
                "No hay información de transporte "
                "suficiente para esta selección."
            )


    # --------------------------------------------------------
    # INSIGHTS AUTOMÁTICOS
    # --------------------------------------------------------

    st.markdown(
        "#### 💡 Insights del destino"
    )


    if n_recursos == 0:

        st.warning(
            "No hay suficientes datos para generar insights."
        )

    else:

        # Insight 1
        if n_categorias > 0:

            st.markdown(
                f"""
                <div class="insight">

                <b>1. Diversidad de oferta</b><br>

                El entorno analizado presenta
                <b>{n_categorias} categorías diferentes</b>
                de recursos identificadas en los datos.

                </div>
                """,
                unsafe_allow_html=True
            )


        # Insight 2
        if not ratings.empty:

            if rating_medio >= 4:

                texto_rating = (
                    "La valoración media disponible "
                    "es elevada."
                )

            elif rating_medio >= 3:

                texto_rating = (
                    "La valoración media disponible "
                    "se encuentra en un nivel intermedio."
                )

            else:

                texto_rating = (
                    "La valoración media disponible "
                    "es relativamente baja."
                )


            st.markdown(
                f"""
                <div class="insight">

                <b>2. Percepción del entorno</b><br>

                {texto_rating}

                Valoración media:
                <b>{rating_medio:.2f} ⭐</b>.

                </div>
                """,
                unsafe_allow_html=True
            )


        # Insight 3
        if transporte_disponible:

            st.markdown(
                """
                <div class="insight">

                <b>3. Conectividad</b><br>

                El dataset dispone de información
                georreferenciada sobre transporte cercano,
                permitiendo incorporar la conectividad
                como dimensión del análisis territorial.

                </div>
                """,
                unsafe_allow_html=True
            )


        # Insight 4
        if modo == "📍 Un lugar":

            st.markdown(
                f"""
                <div class="insight">

                <b>4. Área de exploración</b><br>

                El análisis se ha realizado dentro de un
                radio de <b>{radio} metros</b> alrededor
                de <b>{lugar_seleccionado}</b>.

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="insight">

                <b>4. Análisis territorial</b><br>

                El análisis se ha realizado sobre la zona
                <b>{zona_seleccionada}</b>, utilizando
                los recursos georreferenciados disponibles
                para ese territorio.

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 21. RESEÑAS
# ============================================================

st.markdown("---")

st.markdown(
    "### 💬 Opiniones disponibles"
)


if df_comentarios is not None and not df_comentarios.empty:

    comentarios = df_comentarios.copy()


    # --------------------------------------------------------
    # FILTRAR POR NOMBRE
    # --------------------------------------------------------

    if modo == "📍 Un lugar":

        if "NOMBRE_RECURSO" in comentarios.columns:

            comentarios["nombre_tmp"] = (
                comentarios["NOMBRE_RECURSO"]
                .fillna("")
                .astype(str)
                .str.lower()
            )


            comentarios = comentarios[
                comentarios["nombre_tmp"]
                == str(
                    lugar_seleccionado
                ).lower()
            ]


    # --------------------------------------------------------
    # FILTRAR POR ZONA
    # --------------------------------------------------------

    else:

        if "BARRIO_RECURSO" in comentarios.columns:

            comentarios = comentarios[
                comentarios[
                    "BARRIO_RECURSO"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                == str(
                    zona_seleccionada
                ).lower()
            ]


        elif "BARRIO" in comentarios.columns:

            comentarios = comentarios[
                comentarios[
                    "BARRIO"
                ]
                .fillna("")
                .astype(str)
                .str.lower()
                == str(
                    zona_seleccionada
                ).lower()
            ]


    if comentarios.empty:

        st.info(
            "No hay reseñas asociadas directamente "
            "a esta selección."
        )

    else:

        if "CALIFICACION" in comentarios.columns:

            comentarios["CALIFICACION"] = pd.to_numeric(
                comentarios["CALIFICACION"],
                errors="coerce"
            )


        for _, rev in comentarios.head(10).iterrows():

            rating = rev.get(
                "CALIFICACION",
                np.nan
            )


            usuario = rev.get(
                "USUARIO",
                "Usuario"
            )


            fecha = rev.get(
                "FECHA",
                ""
            )


            texto = rev.get(
                "TEXTO_RESENA",
                "Sin texto."
            )


            nombre = rev.get(
                "NOMBRE_RECURSO",
                titulo_contexto
            )


            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        ⭐ {formato_rating(rating)}
                    </div>

                    <div class="card-text">

                        <b>{nombre}</b>
                        &nbsp; | &nbsp;
                        {usuario}
                        &nbsp; | &nbsp;
                        {fecha}

                    </div>

                    <div class="card-text">
                        "{texto}"
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


else:

    st.info(
        "No se ha encontrado información de reseñas."
    )


# ============================================================
# 22. PIE DE PÁGINA
# ============================================================

st.markdown("---")

st.caption(
    "TUI Destination Explorer · Desafío 3 · TFM Grupo 1"
)

st.caption(
    "Los indicadores se calculan exclusivamente a partir "
    "de los datos disponibles en el repositorio."
)