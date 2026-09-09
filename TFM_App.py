# ============================================================
# TUI TERRITORIAL INTELLIGENCE DASHBOARD
# Reto 3 - Máster Data Science / Big Data & Business Analytics
# AI-Dashboard para la gestión de oferta turística
# georreferenciada e integración con datos abiertos
# ============================================================

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import folium
import html
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="TUI | Territorial Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. ESTILO CORPORATIVO
# ============================================================

st.markdown("""
<style>

    /* Fondo general */
    .stApp {
        background-color: #f5f7fa;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e6e9ef;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #17365D;
    }

    /* KPI */
    .kpi-card {
        background: #ffffff;
        padding: 18px 20px;
        border-radius: 12px;
        border: 1px solid #e5e9f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        min-height: 125px;
    }

    .kpi-title {
        color: #6b7280;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .kpi-value {
        color: #17365D;
        font-size: 28px;
        font-weight: 700;
    }

    .kpi-description {
        color: #7b8491;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Insights */
    .insight-box {
        background: #ffffff;
        border-left: 5px solid #1967D2;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    .opportunity-box {
        background: #eef8f1;
        border-left: 5px solid #2e8b57;
        padding: 18px;
        border-radius: 8px;
    }

    .warning-box {
        background: #fff8e6;
        border-left: 5px solid #e0a800;
        padding: 18px;
        border-radius: 8px;
    }

    /* Separadores */
    .section-divider {
        margin-top: 20px;
        margin-bottom: 20px;
        border-top: 1px solid #e1e5eb;
    }

    /* Texto secundario */
    .muted {
        color: #6b7280;
        font-size: 14px;
    }


/* TEMA_ADAPTABLE_V1 */

.stApp {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}

[data-testid="stSidebar"] {
    background-color:: var(--secondary-background-color) !important;
    border-color: rgba(128, 128, 128, 0.25) !important;
}

h1, h2, h3,
.kpi-title, .kpi-value, .kpi-description, .muted {
    color: var(--text-color) !important;
}

.kpi-card,
.insight-box,
.opportunity-box,
.advertencia-caja {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color) !important;
    border-color: rgba(128, 128, 128, 0.25) !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def columna_existe(df, columna):
    """Comprueba si una columna existe."""
    return columna in df.columns


def normalizar_0_1(serie):
    """
    Normalización Min-Max entre 0 y 1.
    Si todos los valores son iguales devuelve 0.5.
    """
    serie = pd.to_numeric(serie, errors="coerce")

    minimo = serie.min()
    maximo = serie.max()

    if pd.isna(minimo) or pd.isna(maximo):
        return pd.Series(0.5, index=serie.index)

    if maximo == minimo:
        return pd.Series(0.5, index=serie.index)

    return (serie - minimo) / (maximo - minimo)


def tarjeta_kpi(titulo, valor, descripcion, icono=""):
    """Genera una tarjeta KPI."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{icono} {titulo}</div>
            <div class="kpi-value">{valor}</div>
            <div class="kpi-description">{descripcion}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def texto_mapa(valor, defecto="N/D"):
    """Devuelve un texto seguro y legible para los popups del mapa."""
    if pd.isna(valor):
        return defecto

    texto = str(valor).strip()

    if not texto or texto.lower() in {"nan", "none", "<na>"}:
        return defecto

    return html.escape(texto)


def nombre_mapa(row, tipo_activo):
    """Evita mostrar S/R o valores vacíos como si fueran nombres reales."""
    nombre = texto_mapa(row.get("nombre"), defecto="")

    if nombre.casefold() in {"", "s/r", "sin rotulo", "sin rótulo"}:
        if tipo_activo == "Negocio":
            return "Establecimiento sin rótulo"
        return "Punto de interés sin nombre"

    return nombre


def obtener_limites_mapa(estado_mapa):
    """Extrae y valida los límites visibles devueltos por st_folium."""
    if not isinstance(estado_mapa, dict):
        return None

    limites = estado_mapa.get("bounds")
    if not isinstance(limites, dict):
        return None

    try:
        suroeste = limites["_southWest"]
        noreste = limites["_northEast"]

        sur = float(suroeste["lat"])
        oeste = float(suroeste["lng"])
        norte = float(noreste["lat"])
        este = float(noreste["lng"])
    except (KeyError, TypeError, ValueError):
        return None

    if sur >= norte or oeste >= este:
        return None

    return {
        "sur": sur,
        "oeste": oeste,
        "norte": norte,
        "este": este
    }


def filtrar_zona_visible(df, limites):
    """Limita los activos a la extensión que está viendo el usuario."""
    if limites is None or df.empty:
        return df.copy()

    if not {"lat", "lon"}.issubset(df.columns):
        return df.iloc[0:0].copy()

    resultado = df.copy()
    latitud = pd.to_numeric(resultado["lat"], errors="coerce")
    longitud = pd.to_numeric(resultado["lon"], errors="coerce")

    return resultado[
        latitud.between(limites["sur"], limites["norte"])
        & longitud.between(limites["oeste"], limites["este"])
    ].copy()


def maximo_activos_por_zoom(zoom):
    """Limita los puntos para que el mapa siga siendo fluido."""
    if zoom >= 16:
        return 400
    if zoom >= 15:
        return 250
    if zoom >= 14:
        return 150
    return 100


# ============================================================
# 4. CARGA DE DATOS
# ============================================================

@st.cache_data
def cargar_datos():
    df_barrios = pd.read_parquet("data/Oro/Barrios.parquet")
    df_rest = pd.read_parquet("data/Oro/Restaurantes.parquet")
    df_poi = pd.read_parquet("data/Oro/POI.parquet")
    gdf_geo = gpd.read_file("data/Oro/Barrios.geojson")

    return df_barrios, df_rest, df_poi, gdf_geo


try:

    df_barrios, df_rest, df_poi, gdf_geo = cargar_datos()

except Exception as e:

    st.error("❌ No se pudieron cargar los datos.")

    st.code(str(e))

    st.info("""
    Comprueba que la aplicación mantiene esta estructura:

    app.py
    Oro/
        Barrios.parquet
        Restaurantes.parquet
        POI.parquet
        Barrios.geojson
    """)

    st.stop()


# ============================================================
# 5. PREPARACIÓN DE DATOS
# ============================================================

# Eliminar espacios accidentales de nombres de columnas

df_barrios.columns = df_barrios.columns.str.strip()
df_rest.columns = df_rest.columns.str.strip()
df_poi.columns = df_poi.columns.str.strip()
gdf_geo.columns = gdf_geo.columns.str.strip()


# ------------------------------------------------------------
# Asegurar booleanos de terraza
# ------------------------------------------------------------

if "tiene_terraza" in df_rest.columns:

    df_rest["tiene_terraza"] = (
        df_rest["tiene_terraza"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "si", "sí", "yes"])
    )


# ------------------------------------------------------------
# Crear índice de oportunidad territorial
# ------------------------------------------------------------

if not df_barrios.empty:
    neutro = pd.Series(0.5, index=df_barrios.index)

    accesibilidad_norm = (
        normalizar_0_1(df_barrios["accesibilidad_media"])
        if "accesibilidad_media" in df_barrios.columns else neutro
    )

    densidad_norm = (
        normalizar_0_1(df_barrios["densidad_negocios"])
        if "densidad_negocios" in df_barrios.columns else neutro
    )

    orientacion_turistica = (
        pd.to_numeric(df_barrios["pct_turistico"], errors="coerce")
        .fillna(0).div(100).clip(0, 1)
        if "pct_turistico" in df_barrios.columns else neutro
    )

    variedad = (
        pd.to_numeric(df_barrios["diversidad"], errors="coerce")
        .fillna(0).clip(0, 1)
        if "diversidad" in df_barrios.columns else neutro
    )

    df_barrios["indice_oportunidad"] = (
        0.35 * accesibilidad_norm
        + 0.30 * (1 - densidad_norm)
        + 0.20 * (1 - orientacion_turistica)
        + 0.15 * variedad
    ).mul(100).round(1)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.markdown("## 🌍 TUI")
st.sidebar.markdown(
    "**Territorial Intelligence Dashboard**"
)

st.sidebar.caption(
    "Inteligencia georreferenciada para la gestión "
    "turística de Madrid"
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 1. Territorio")

# Distrito

distritos = sorted(
    df_barrios["distrito"].dropna().unique()
    if "distrito" in df_barrios.columns
    else []
)

distrito_seleccionado = st.sidebar.multiselect(
    "Distrito",
    options=distritos,
    default=["Centro"] if "Centro" in distritos else distritos[:1]
)


# ------------------------------------------------------------
# Barrio dependiente del distrito
# ------------------------------------------------------------

if (
    distrito_seleccionado
    and "distrito" in df_barrios.columns
):

    barrios_disponibles = sorted(
        df_barrios[
            df_barrios["distrito"].isin(
                distrito_seleccionado
            )
        ]["barrio"]
        .dropna()
        .unique()
    )

else:

    barrios_disponibles = sorted(
        df_barrios["barrio"].dropna().unique()
        if "barrio" in df_barrios.columns
        else []
    )


barrios_seleccionados = st.sidebar.multiselect(
    "Barrio",
    options=barrios_disponibles,
    help="Permite profundizar el análisis dentro de los distritos seleccionados."
)


# ============================================================
# FILTRO DE ACTIVOS
# ============================================================

st.sidebar.markdown("### 🏨 2. Tipo de activo")

tipo_activo = st.sidebar.multiselect(
    "Tipo de activo",
    options=["Negocios", "Puntos de interés (POI)"],
    default=["Negocios"]
)

etiquetas_macro = {
    "restaurantes": "Negocios",
    "alimentacion": "Alimentación",
    "ocio": "Ocio",
    "deporte": "Deporte",
    "cultura": "Cultura",
    "hoteles": "Hoteles",
    "comercio": "Comercio",
    "servicios": "Servicios",
    "salud": "Salud",
    "educacion": "Educación",
    "otros": "Otros"
}

macro_categorias_disponibles = sorted(
    df_rest["macro_categoria"].dropna().unique()
)

macro_categorias_seleccionadas = st.sidebar.multiselect(
    "Categorías de negocios",
    options=macro_categorias_disponibles,
    default=["restaurantes"],
    format_func=lambda categoria: etiquetas_macro.get(
        categoria, categoria.title()
    )
)


# ============================================================
# ACCESIBILIDAD
# ============================================================

st.sidebar.markdown("### 🚇 3. Accesibilidad")

if "n_paradas_400m" in df_rest.columns:

    min_paradas = int(
        df_rest["n_paradas_400m"]
        .fillna(0)
        .min()
    )

    max_paradas = int(
        df_rest["n_paradas_400m"]
        .fillna(0)
        .max()
    )

    paradas_minimas = st.sidebar.slider(
        "Paradas de transporte <400 m",
        min_value=min_paradas,
        max_value=max_paradas,
        value=min_paradas,
        help="""
        Número mínimo de paradas de transporte público
        situadas a menos de 400 metros del activo.
        """
    )

else:

    paradas_minimas = 0


# ============================================================
# OFERTA / TERRAZAS
# ============================================================

st.sidebar.markdown("### ☀️ 4. Características de oferta")

terraza = st.sidebar.selectbox(
    "Oferta exterior",
    options=[
        "Todos",
        "Con terraza",
        "Sin terraza"
    ],
    help="""
    La terraza se utiliza como atributo de oferta exterior.
    No se considera por sí sola un indicador ESG.
    """
)


# ============================================================
# ESCENARIO CLIMÁTICO
# ============================================================

st.sidebar.markdown("### 🌡️ 5. Escenario contextual")

temperatura = st.sidebar.slider(
    "Temperatura simulada",
    min_value=15,
    max_value=45,
    value=25,
    step=1,
    help="""
    Variable contextual utilizada para simular cambios
    potenciales en la presión sobre la oferta exterior.
    No representa una predicción meteorológica.
    """
)


if temperatura < 25:

    escenario_clima = "Favorable"

elif temperatura < 32:

    escenario_clima = "Cálido"

elif temperatura < 38:

    escenario_clima = "Estrés térmico potencial"

else:

    escenario_clima = "Condiciones extremas"


# ============================================================
# RESET
# ============================================================

if st.sidebar.button("🔄 Restablecer filtros"):

    st.rerun()


# ============================================================
# 7. FILTRADO DE DATOS
# ============================================================

# ------------------------------------------------------------
# Barrios
# ------------------------------------------------------------

df_barrios_filtrado = df_barrios.copy()

if distrito_seleccionado:

    df_barrios_filtrado = df_barrios_filtrado[
        df_barrios_filtrado["distrito"].isin(
            distrito_seleccionado
        )
    ]

if barrios_seleccionados:

    df_barrios_filtrado = df_barrios_filtrado[
        df_barrios_filtrado["barrio"].isin(
            barrios_seleccionados
        )
    ]


# ------------------------------------------------------------
# GeoJSON
# ------------------------------------------------------------

gdf_geo_filtrado = gdf_geo.copy()

if distrito_seleccionado:

    gdf_geo_filtrado = gdf_geo_filtrado[
        gdf_geo_filtrado["distrito"].isin(
            distrito_seleccionado
        )
    ]

if barrios_seleccionados:

    gdf_geo_filtrado = gdf_geo_filtrado[
        gdf_geo_filtrado["barrio"].isin(
            barrios_seleccionados
        )
    ]


# ------------------------------------------------------------
# Restaurantes
# ------------------------------------------------------------

df_rest_filtrado = df_rest.copy()

if distrito_seleccionado and "distrito" in df_rest.columns:

    df_rest_filtrado = df_rest_filtrado[
        df_rest_filtrado["distrito"].isin(
            distrito_seleccionado
        )
    ]

if barrios_seleccionados and "barrio" in df_rest.columns:

    df_rest_filtrado = df_rest_filtrado[
        df_rest_filtrado["barrio"].isin(
            barrios_seleccionados
        )
    ]



if "Negocios" in tipo_activo and "macro_categoria" in df_rest_filtrado.columns:
    df_rest_filtrado = df_rest_filtrado[
        df_rest_filtrado["macro_categoria"].isin(
            macro_categorias_seleccionadas
        )
    ]
elif "Negocios" not in tipo_activo:
    df_rest_filtrado = df_rest_filtrado.iloc[0:0]

# ------------------------------------------------------------
# Accesibilidad
# ------------------------------------------------------------

if "n_paradas_400m" in df_rest_filtrado.columns:

    df_rest_filtrado = df_rest_filtrado[
        df_rest_filtrado["n_paradas_400m"].fillna(0)
        >= paradas_minimas
    ]


# ------------------------------------------------------------
# Terraza
# ------------------------------------------------------------

if "tiene_terraza" in df_rest_filtrado.columns:

    if terraza == "Con terraza":

        df_rest_filtrado = df_rest_filtrado[
            df_rest_filtrado["tiene_terraza"] == True
        ]

    elif terraza == "Sin terraza":

        df_rest_filtrado = df_rest_filtrado[
            df_rest_filtrado["tiene_terraza"] == False
        ]


# ------------------------------------------------------------
# Puntos de interés para el mapa
# ------------------------------------------------------------

df_poi_filtrado = df_poi.copy()

if distrito_seleccionado and "distrito" in df_poi_filtrado.columns:

    df_poi_filtrado = df_poi_filtrado[
        df_poi_filtrado["distrito"].isin(
            distrito_seleccionado
        )
    ]

if barrios_seleccionados and "barrio" in df_poi_filtrado.columns:

    df_poi_filtrado = df_poi_filtrado[
        df_poi_filtrado["barrio"].isin(
            barrios_seleccionados
        )
    ]

if "n_paradas_400m" in df_poi_filtrado.columns:

    df_poi_filtrado = df_poi_filtrado[
        df_poi_filtrado["n_paradas_400m"].fillna(0)
        >= paradas_minimas
    ]


# ============================================================
# 8. CABECERA
# ============================================================

st.markdown(
    """
    # 🌍 TUI Territorial Intelligence

    ### Plataforma de inteligencia georreferenciada para la gestión turística

    **Madrid · Oferta · Accesibilidad · Sostenibilidad · Oportunidades**
    """
)

st.markdown(
    """
    <div class="insight-box">
    <b>Objetivo del dashboard</b><br>
    Integrar información territorial y de oferta turística para
    identificar patrones de concentración, accesibilidad,
    zonas infrautilizadas y oportunidades de desarrollo.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 9. KPI
# ============================================================

st.markdown("## 📌 Indicadores principales")

k1, k2, k3, k4, k5 = st.columns(5)


# ------------------------------------------------------------
# KPI 1
# ------------------------------------------------------------

with k1:

    tarjeta_kpi(
        "Negocios seleccionados",
        f"{len(df_rest_filtrado):,}",
        "Activos que cumplen los filtros seleccionados",
        "🏨"
    )


# ------------------------------------------------------------
# KPI 2
# ------------------------------------------------------------

if (
    not df_barrios_filtrado.empty
    and "densidad_negocios" in df_barrios_filtrado.columns
):

    densidad_media = (
        df_barrios_filtrado["densidad_negocios"]
        .mean()
    )

else:

    densidad_media = 0


with k2:

    tarjeta_kpi(
        "Densidad media",
        f"{densidad_media:.1f}",
        "Negocios por km²",
        "📍"
    )


# ------------------------------------------------------------
# KPI 3
# ------------------------------------------------------------

if (
    not df_barrios_filtrado.empty
    and "accesibilidad_media" in df_barrios_filtrado.columns
):

    accesibilidad_media = (
        df_barrios_filtrado["accesibilidad_media"]
        .mean()
    )

else:

    accesibilidad_media = 0


with k3:

    tarjeta_kpi(
        "Accesibilidad",
        f"{accesibilidad_media:.1f}",
        "Paradas medias próximas a los activos",
        "🚇"
    )


# ------------------------------------------------------------
# KPI 4
# ------------------------------------------------------------

if (
    not df_barrios_filtrado.empty
    and "pct_terrazas" in df_barrios_filtrado.columns
):

    pct_terraza = (
        df_barrios_filtrado["pct_terrazas"]
        .mean()
    )

else:

    pct_terraza = 0


with k4:

    tarjeta_kpi(
        "Oferta exterior",
        f"{pct_terraza:.1f}%",
        "Establecimientos con terraza",
        "☀️"
    )


# ------------------------------------------------------------
# KPI 5
# ------------------------------------------------------------

if (
    not df_barrios_filtrado.empty
    and "indice_oportunidad" in df_barrios_filtrado.columns
):

    oportunidad_media = (
        df_barrios_filtrado["indice_oportunidad"]
        .mean()
    )

else:

    oportunidad_media = 0


with k5:

    tarjeta_kpi(
        "Oportunidad",
        f"{oportunidad_media:.0f}/100",
        "Potencial territorial estimado",
        "💡"
    )



# ============================================================
# KPIs NUEVOS DE COMPOSICIÓN TERRITORIAL
# ============================================================

st.markdown("### 🧭 Composición territorial")

if not df_barrios_filtrado.empty:
    negocios_zona = int(
        df_barrios_filtrado.get("n_negocios", pd.Series(dtype=float)).sum()
    )
    restaurantes_zona = int(
        df_barrios_filtrado.get("n_restaurantes", pd.Series(dtype=float)).sum()
    )

    pesos = df_barrios_filtrado["n_negocios"]

    variedad_media = (
        np.average(df_barrios_filtrado["diversidad"], weights=pesos)
        if pesos.sum() > 0 else 0
    )

    turismo_medio = (
        np.average(df_barrios_filtrado["pct_turistico"], weights=pesos)
        if pesos.sum() > 0 else 0
    )

    densidad_poi_media = (
        pd.to_numeric(
            df_barrios_filtrado["densidad_poi"], errors="coerce"
        ).mean()
        if "densidad_poi" in df_barrios_filtrado.columns else 0
    )
else:
    negocios_zona = 0
    restaurantes_zona = 0
    variedad_media = 0
    turismo_medio = 0
    densidad_poi_media = 0

c1_nuevo, c2_nuevo, c3_nuevo, c4_nuevo, c5_nuevo = st.columns(5)

with c1_nuevo:
    tarjeta_kpi(
        "Negocios totales",
        f"{negocios_zona:,}",
        "Negocios existentes en la zona seleccionada",
        "🏢"
    )

with c2_nuevo:
    tarjeta_kpi(
        "Restaurantes reales",
        f"{restaurantes_zona:,}",
        "Locales clasificados como restaurantes",
        "🍽️"
    )

with c3_nuevo:
    tarjeta_kpi(
        "Variedad de oferta",
        f"{variedad_media:.2f}/1",
        "Equilibrio entre las diferentes categorías",
        "🧩"
    )

with c4_nuevo:
    tarjeta_kpi(
        "Orientación turística",
        f"{turismo_medio:.1f}%",
        "Peso de restauración, hoteles, cultura y ocio",
        "🧳"
    )

with c5_nuevo:
    tarjeta_kpi(
        "Densidad de POI",
        f"{densidad_poi_media:.1f}",
        "Puntos de interés por km²",
        "📌"
    )

# ============================================================
# 10. ESCENARIO CLIMÁTICO
# ============================================================

st.markdown("---")

st.markdown("## 🌡️ Contexto climático simulado")

c1, c2, c3 = st.columns(3)

with c1:

    st.metric(
        "Temperatura",
        f"{temperatura} °C"
    )

with c2:

    st.metric(
        "Escenario",
        escenario_clima
    )

with c3:

    if temperatura < 32:
        presion = "Baja"
    elif temperatura < 38:
        presion = "Moderada"
    else:
        presion = "Alta"

    st.metric(
        "Presión potencial",
        presion
    )


st.caption(
    """
    La temperatura se utiliza exclusivamente como variable
    contextual de simulación. No constituye una predicción
    meteorológica ni demuestra por sí misma la existencia de
    congestión turística.
    """
)


# ============================================================
# 11. TABS PRINCIPALES
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📊 Resumen ejecutivo",
        "🗺️ Mapa territorial",
        "🚇 Oferta y accesibilidad",
        "💡 Oportunidades",
        "🎛️ Escenarios",
        "🤖 Recomendador"
    ]
)


# ============================================================
# TAB 1 - RESUMEN
# ============================================================

with tab1:

    st.markdown("## 📊 Lectura ejecutiva")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### 🧭 Situación territorial")

        st.markdown(
            f"""
            <div class="insight-box">

            El análisis actual comprende
            <b>{len(df_barrios_filtrado)}</b> unidades territoriales
            y <b>{len(df_rest_filtrado):,}</b> activos turísticos.

            La accesibilidad media registrada es de
            <b>{accesibilidad_media:.1f}</b> paradas próximas.

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown("### 💡 Interpretación")

        if oportunidad_media >= 70:

            mensaje = """
            El territorio presenta un nivel elevado de
            oportunidad según las variables disponibles.
            """

        elif oportunidad_media >= 45:

            mensaje = """
            El territorio presenta oportunidades moderadas
            que requieren análisis específico por barrio.
            """

        else:

            mensaje = """
            La oferta aparece relativamente consolidada.
            Conviene investigar zonas concretas con menor densidad.
            """

        st.markdown(
            f"""
            <div class="opportunity-box">
            {mensaje}
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Ranking de oportunidades
    # --------------------------------------------------------

    st.markdown("### 🏆 Zonas con mayor potencial")

    if (
        not df_barrios_filtrado.empty
        and "indice_oportunidad" in df_barrios_filtrado.columns
    ):

        ranking = (
            df_barrios_filtrado
            .sort_values(
                "indice_oportunidad",
                ascending=False
            )
            [
                [
                    "barrio",
                    "indice_oportunidad"
                ]
            ]
            .head(10)
        )

        ranking["indice_oportunidad"] = (
            ranking["indice_oportunidad"]
            .round(1)
        )

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAB 2 - MAPA
# ============================================================

with tab2:

    st.markdown("## 🗺️ Mapa territorial")

    st.markdown(
        """
        Visualización de la concentración de oferta y de los
        activos turísticos disponibles dentro del territorio analizado.
        """
    )

    # --------------------------------------------------------
    # Buscador de activos dentro de los filtros seleccionados
    # --------------------------------------------------------

    firma_contexto_mapa = (
        tuple(distrito_seleccionado),
        tuple(barrios_seleccionados),
        tuple(tipo_activo),
        tuple(macro_categorias_seleccionadas),
        int(paradas_minimas),
        terraza
    )

    if (
        st.session_state.get("_firma_contexto_mapa")
        != firma_contexto_mapa
    ):
        st.session_state["_firma_contexto_mapa"] = (
            firma_contexto_mapa
        )
        st.session_state.pop("buscador_mapa", None)

    candidatos_busqueda = []

    if "Negocios" in tipo_activo:
        restaurantes_busqueda = df_rest_filtrado.copy()
        restaurantes_busqueda["_tipo_busqueda"] = (
            restaurantes_busqueda["macro_categoria"]
            .map(etiquetas_macro)
            .fillna("Negocio")
        )
        candidatos_busqueda.append(restaurantes_busqueda)

    if "Puntos de interés (POI)" in tipo_activo:
        poi_busqueda = df_poi_filtrado.copy()
        poi_busqueda["_tipo_busqueda"] = "Punto de interés"
        candidatos_busqueda.append(poi_busqueda)

    if candidatos_busqueda:
        df_busqueda = pd.concat(
            candidatos_busqueda,
            ignore_index=True,
            sort=False
        )

        if {"nombre", "lat", "lon"}.issubset(df_busqueda.columns):
            df_busqueda["lat"] = pd.to_numeric(
                df_busqueda["lat"],
                errors="coerce"
            )
            df_busqueda["lon"] = pd.to_numeric(
                df_busqueda["lon"],
                errors="coerce"
            )
            df_busqueda["_nombre_busqueda"] = (
                df_busqueda["nombre"]
                .astype("string")
                .str.strip()
            )

            nombres_no_validos = {
                "",
                "s/r",
                "sin rotulo",
                "sin rótulo",
                "nan",
                "none",
                "<na>"
            }

            df_busqueda = df_busqueda[
                df_busqueda["lat"].notna()
                & df_busqueda["lon"].notna()
                & df_busqueda["_nombre_busqueda"].notna()
                & ~df_busqueda["_nombre_busqueda"]
                .str.casefold()
                .isin(nombres_no_validos)
            ].copy()

            df_busqueda = df_busqueda.sort_values(
                "_nombre_busqueda",
                key=lambda serie: serie.str.casefold()
            ).reset_index(drop=True)
        else:
            df_busqueda = pd.DataFrame()
    else:
        df_busqueda = pd.DataFrame()

    def etiqueta_resultado(indice):
        if indice is None:
            return "Escribe o selecciona un nombre"

        fila = df_busqueda.loc[indice]
        etiqueta = (
            f"{fila['_nombre_busqueda']} — "
            f"{fila['_tipo_busqueda']}"
        )

        barrio = fila.get("barrio")
        if pd.notna(barrio) and str(barrio).strip():
            etiqueta += f" · {str(barrio).strip()}"

        return etiqueta

    opciones_busqueda = (
        [None] + df_busqueda.index.tolist()
        if not df_busqueda.empty
        else [None]
    )

    activo_encontrado = st.selectbox(
        "🔎 Buscar un establecimiento o punto de interés",
        options=opciones_busqueda,
        format_func=etiqueta_resultado,
        key="buscador_mapa",
        help=(
            "La búsqueda utiliza los distritos, barrios y tipos de "
            "activo seleccionados en el panel lateral."
        )
    )

    centro_mapa = [40.4167, -3.7037]
    zoom_mapa = 13
    limites_mapa = None

    firma_estado_mapa = (
        firma_contexto_mapa,
        activo_encontrado
    )

    if (
        st.session_state.get("_firma_estado_mapa")
        != firma_estado_mapa
    ):
        st.session_state["_firma_estado_mapa"] = firma_estado_mapa
        estado_mapa_previo = {}
    else:
        estado_mapa_previo = st.session_state.get(
            "mapa_territorial",
            {}
        )

    if activo_encontrado is not None:
        fila_encontrada = df_busqueda.loc[activo_encontrado]
        centro_mapa = [
            float(fila_encontrada["lat"]),
            float(fila_encontrada["lon"])
        ]
        zoom_mapa = 17

    else:
        limites_mapa = obtener_limites_mapa(
            estado_mapa_previo
        )

        if limites_mapa is not None:
            centro_mapa = [
                (
                    limites_mapa["sur"]
                    + limites_mapa["norte"]
                ) / 2,
                (
                    limites_mapa["oeste"]
                    + limites_mapa["este"]
                ) / 2
            ]

            try:
                zoom_mapa = int(
                    estado_mapa_previo.get("zoom", 13)
                )
            except (TypeError, ValueError):
                zoom_mapa = 13

            zoom_mapa = min(max(zoom_mapa, 10), 18)

    # Centro de Madrid
    mapa = folium.Map(
        location=centro_mapa,
        zoom_start=zoom_mapa,
        tiles="CartoDB positron"
    )


    # --------------------------------------------------------
    # Choropleth
    # --------------------------------------------------------

    if (
        not gdf_geo_filtrado.empty
        and "densidad_negocios" in df_barrios_filtrado.columns
    ):

        folium.Choropleth(
            geo_data=gdf_geo_filtrado,
            data=df_barrios_filtrado,
            columns=[
                "barrio",
                "densidad_negocios"
            ],
            key_on="feature.properties.barrio",
            fill_color="YlOrRd",
            fill_opacity=0.65,
            line_opacity=0.25,
            legend_name="Densidad de negocios por km²"
        ).add_to(mapa)

        # Información territorial al pasar el ratón.
        folium.GeoJson(
            gdf_geo_filtrado,
            name="Información de barrios",
            style_function=lambda feature: {
                "fillOpacity": 0,
                "color": "transparent"
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["barrio", "distrito", "densidad_negocios"],
                aliases=["Barrio:", "Distrito:", "Densidad:"],
                sticky=False
            )
        ).add_to(mapa)


    # --------------------------------------------------------
    # Restaurantes
    # --------------------------------------------------------

    limite_activos_mapa = maximo_activos_por_zoom(
        zoom_mapa
    )

    restaurantes_candidatos_mapa = filtrar_zona_visible(
        df_rest_filtrado,
        limites_mapa
    )

    restaurantes_mapa = restaurantes_candidatos_mapa.iloc[0:0]

    if "Negocios" in tipo_activo:

        restaurantes_mapa = restaurantes_candidatos_mapa.sample(
            n=min(
                limite_activos_mapa,
                len(restaurantes_candidatos_mapa)
            ),
            random_state=42
        )

    capa_restaurantes = folium.FeatureGroup(
        name="Negocios",
        show="Negocios" in tipo_activo
    )

    cluster_restaurantes = MarkerCluster(
        name="Agrupación de negocios",
        control=False,
        disableClusteringAtZoom=17
    ).add_to(capa_restaurantes)

    estilos_macro = {
        "restaurantes": ("Restaurante", "blue", "cutlery"),
        "alimentacion": ("Alimentación", "green", "shopping-basket"),
        "ocio": ("Ocio", "purple", "ticket"),
        "deporte": ("Deporte", "orange", "futbol-o"),
        "cultura": ("Cultura", "darkpurple", "university"),
        "hoteles": ("Hotel", "cadetblue", "bed"),
        "comercio": ("Comercio", "darkblue", "shopping-bag"),
        "servicios": ("Servicios", "gray", "briefcase"),
        "salud": ("Salud", "red", "plus-square"),
        "educacion": ("Educación", "lightblue", "graduation-cap"),
        "otros": ("Otros", "lightgray", "map-marker")
    }

    for _, row in restaurantes_mapa.iterrows():

        if (
            "lat" not in row
            or "lon" not in row
        ):
            continue

        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        macro_visible = str(
            row.get("macro_categoria", "otros")
        ).lower()

        etiqueta_macro, color_macro, icono_macro = (
            estilos_macro.get(
                macro_visible,
                estilos_macro["otros"]
            )
        )

        terraza_icon = (
            "Sí" if row.get("tiene_terraza", False) else "No"
        ) if macro_visible == "restaurantes" else "No aplica"

        nombre_visible = nombre_mapa(row, "Negocio")
        categoria_visible = texto_mapa(
            row.get("categoria"),
            "Sin detalle"
        )
        barrio_visible = texto_mapa(row.get("barrio"))
        distrito_visible = texto_mapa(row.get("distrito"))
        paradas_visibles = texto_mapa(
            row.get("n_paradas_400m")
        )

        popup_text = f"""
        <b>{nombre_visible}</b><br>
        Tipo: {etiqueta_macro}<br>
        Actividad original: {categoria_visible}<br>
        Barrio: {barrio_visible}<br>
        Distrito: {distrito_visible}<br>
        Paradas &lt;400 m: {paradas_visibles}<br>
        Terraza: {terraza_icon}
        """

        folium.Marker(
            location=[
                row["lat"],
                row["lon"]
            ],
            icon=folium.Icon(
                color=color_macro,
                icon=icono_macro,
                prefix="fa"
            ),
            tooltip=folium.Tooltip(
                nombre_visible,
                sticky=True
            ),
            popup=folium.Popup(
                popup_text,
                max_width=320
            )
        ).add_to(cluster_restaurantes)

    capa_restaurantes.add_to(mapa)


    # --------------------------------------------------------
    # Puntos de interés
    # --------------------------------------------------------

    poi_candidatos_mapa = filtrar_zona_visible(
        df_poi_filtrado,
        limites_mapa
    )

    poi_mapa = poi_candidatos_mapa.iloc[0:0]

    if "Puntos de interés (POI)" in tipo_activo:

        poi_mapa = poi_candidatos_mapa.sample(
            n=min(
                limite_activos_mapa,
                len(poi_candidatos_mapa)
            ),
            random_state=42
        )

    capa_poi = folium.FeatureGroup(
        name="Puntos de interés",
        show="Puntos de interés (POI)" in tipo_activo
    )

    cluster_poi = MarkerCluster(
        name="Agrupación de puntos de interés",
        control=False,
        disableClusteringAtZoom=17
    ).add_to(capa_poi)

    for _, row in poi_mapa.iterrows():

        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue

        nombre_visible = nombre_mapa(row, "POI")
        categoria_visible = texto_mapa(row.get("categoria"))
        barrio_visible = texto_mapa(row.get("barrio"))
        distrito_visible = texto_mapa(row.get("distrito"))
        paradas_visibles = texto_mapa(
            row.get("n_paradas_400m")
        )

        popup_text = f"""
        <b>{nombre_visible}</b><br>
        Categoría: {categoria_visible}<br>
        Barrio: {barrio_visible}<br>
        Distrito: {distrito_visible}<br>
        Paradas &lt;400 m: {paradas_visibles}
        """

        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.Icon(
                color="green",
                icon="info-sign"
            ),
            tooltip=folium.Tooltip(
                nombre_visible,
                sticky=True
            ),
            popup=folium.Popup(
                popup_text,
                max_width=320
            )
        ).add_to(cluster_poi)

    capa_poi.add_to(mapa)

    # Resalta el resultado seleccionado y centra el mapa sobre él.
    if activo_encontrado is not None:
        nombre_encontrado = texto_mapa(
            fila_encontrada.get("_nombre_busqueda"),
            "Activo seleccionado"
        )
        tipo_encontrado = texto_mapa(
            fila_encontrada.get("_tipo_busqueda")
        )
        barrio_encontrado = texto_mapa(
            fila_encontrada.get("barrio")
        )

        folium.Marker(
            location=centro_mapa,
            icon=folium.Icon(
                color="red",
                icon="search",
                prefix="fa"
            ),
            tooltip=folium.Tooltip(
                f"Resultado: {nombre_encontrado}",
                sticky=True
            ),
            popup=folium.Popup(
                (
                    f"<b>{nombre_encontrado}</b><br>"
                    f"Tipo: {tipo_encontrado}<br>"
                    f"Barrio: {barrio_encontrado}"
                ),
                max_width=320
            )
        ).add_to(mapa)

    elementos_leyenda = ""

    if "Negocios" in tipo_activo:
        for categoria in macro_categorias_seleccionadas:
            etiqueta, color, _ = estilos_macro.get(
                categoria, estilos_macro["otros"]
            )
            elementos_leyenda += (
                f'<span style="color:{color};font-size:18px;">●</span> '
                f'{etiqueta}<br>'
            )

    if "Puntos de interés (POI)" in tipo_activo:
        elementos_leyenda += (
            '<span style="color:#2E8B57;font-size:18px;">●</span> '
            'Punto de interés<br>'
        )

    elementos_leyenda += (
        '<span style="color:#D93025;font-size:18px;">●</span> '
        'Resultado de búsqueda'
    )

    leyenda_mapa = f"""
    <div style="
        position: fixed;
        bottom: 35px;
        left: 35px;
        z-index: 9999;
        background: white;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px 13px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        font-size: 13px;
        color: #111827 !important;
        line-height: 1.5;
    ">
        <b>Leyenda</b><br>
        {elementos_leyenda}
    </div>
    """

    mapa.get_root().html.add_child(
        folium.Element(leyenda_mapa)
    )

    folium.LayerControl(
        collapsed=False
    ).add_to(mapa)


    components.html(
        mapa.get_root().render(),
        height=620,
        scrolling=False
    )

    contexto_recuento = (
        "en la zona visible"
        if limites_mapa is not None
        else "con los filtros seleccionados"
    )

    st.caption(
        f"Se muestran {len(restaurantes_mapa):,} de "
        f"{len(restaurantes_candidatos_mapa):,} negocios y "
        f"{len(poi_mapa):,} de "
        f"{len(poi_candidatos_mapa):,} puntos de interés "
        f"{contexto_recuento} (zoom {zoom_mapa}). "
        "Los marcadores cercanos se agrupan para facilitar la lectura. "
        "Pasa el ratón para ver el nombre y pulsa para abrir la ficha. "
        f"El límite dinámico actual es de "
        f"{limite_activos_mapa:,} activos de cada tipo."
    )


# ============================================================
# TAB 3 - OFERTA Y ACCESIBILIDAD
# ============================================================

with tab3:

    st.markdown("## 🚇 Oferta y accesibilidad")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # Densidad
    # --------------------------------------------------------

    with col1:

        if (
            not df_barrios_filtrado.empty
            and "densidad_negocios" in df_barrios_filtrado.columns
        ):

            fig_densidad = px.bar(
                df_barrios_filtrado
                .sort_values(
                    "densidad_negocios",
                    ascending=False
                ),
                x="barrio",
                y="densidad_negocios",
                title="Densidad de negocios por barrio",
                labels={
                    "densidad_negocios":
                    "Densidad de negocios"
                }
            )

            fig_densidad.update_layout(
                xaxis_tickangle=-45
            )

            st.plotly_chart(
                fig_densidad,
                use_container_width=True
            )


    # --------------------------------------------------------
    # Accesibilidad vs densidad
    # --------------------------------------------------------

    with col2:

        if (
            not df_barrios_filtrado.empty
            and "accesibilidad_media"
            in df_barrios_filtrado.columns
            and "densidad_negocios"
            in df_barrios_filtrado.columns
        ):

            fig_scatter = px.scatter(
                df_barrios_filtrado,
                x="accesibilidad_media",
                y="densidad_negocios",
                size=(
                    "n_restaurantes"
                    if "n_restaurantes"
                    in df_barrios_filtrado.columns
                    else None
                ),
                hover_name="barrio",
                title="Accesibilidad vs densidad de oferta",
                labels={
                    "accesibilidad_media":
                    "Accesibilidad",
                    "densidad_negocios":
                    "Densidad"
                }
            )

            st.plotly_chart(
                fig_scatter,
                use_container_width=True
            )


    # --------------------------------------------------------
    # Definición de métricas
    # --------------------------------------------------------

    st.markdown("### 📚 ¿Cómo interpretar las métricas?")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.info(
            """
            **Densidad de negocios**

            Mide la intensidad de establecimientos
            turísticos dentro de cada unidad territorial.

            Una densidad elevada puede indicar
            concentración de oferta.
            """
        )

    with m2:

        st.info(
            """
            **Accesibilidad**

            Representa la disponibilidad media de
            transporte público próximo a los activos.

            En este prototipo se utiliza el umbral
            de 400 metros.
            """
        )

    with m3:

        st.info(
            """
            **Oferta exterior**

            Representa la proporción de establecimientos
            que disponen de terraza.

            Se interpreta como característica de la oferta,
            no como indicador ESG por sí sola.
            """
        )


# ============================================================
# TAB 4 - OPORTUNIDADES
# ============================================================

with tab4:

    st.markdown("## 💡 Oportunidades territoriales")

    st.markdown(
        """
        El índice combina accesibilidad y ausencia relativa
        de concentración de oferta para identificar zonas
        potencialmente interesantes.
        """
    )

    if (
        not df_barrios_filtrado.empty
        and "indice_oportunidad"
        in df_barrios_filtrado.columns
    ):

        fig_op = px.bar(
            df_barrios_filtrado
            .sort_values(
                "indice_oportunidad",
                ascending=False
            ),
            x="barrio",
            y="indice_oportunidad",
            title="Índice de oportunidad territorial",
            labels={
                "indice_oportunidad":
                "Índice de oportunidad (0–100)"
            }
        )

        fig_op.add_hline(
            y=70,
            line_dash="dash",
            annotation_text="Alta oportunidad"
        )

        st.plotly_chart(
            fig_op,
            use_container_width=True
        )


    st.markdown("### 🧮 ¿Cómo se calcula?")

    st.latex(
        r"""
        IOT =
        0.45A +
        0.35(1-D) +
        0.20(1-O)
        """
    )

    st.markdown(
        """
        Donde:

        **A** = accesibilidad normalizada.

        **D** = densidad de oferta normalizada.

        **O** = volumen de oferta normalizado.

        El resultado se transforma a una escala de
        **0 a 100**, donde un valor superior indica
        una combinación más favorable de accesibilidad
        y menor concentración relativa de oferta.

        **Importante:** este índice es un indicador
        exploratorio del prototipo y no una valoración
        definitiva de inversión.
        """
    )


# ============================================================
# TAB 5 - ESCENARIOS
# ============================================================

with tab5:

    st.markdown("## 🤖 Simulador de escenarios")

    st.markdown(
        """
        Este módulo permite explorar cómo un cambio en el
        contexto climático podría modificar la presión potencial
        sobre determinados tipos de oferta.
        """
    )

    # --------------------------------------------------------
    # Temperatura
    # --------------------------------------------------------

    temperatura_escenario = st.slider(
        "🌡️ Temperatura del escenario",
        15,
        45,
        temperatura
    )


    if temperatura_escenario < 25:

        nivel = "🟢 Condiciones favorables"

        recomendacion = """
        El contexto favorece potencialmente las experiencias
        exteriores, terrazas y movilidad peatonal.
        """

    elif temperatura_escenario < 32:

        nivel = "🟡 Condiciones cálidas"

        recomendacion = """
        Las actividades exteriores siguen siendo viables,
        aunque puede aumentar la preferencia por espacios
        interiores y recorridos de menor exposición.
        """

    elif temperatura_escenario < 38:

        nivel = "🟠 Estrés térmico potencial"

        recomendacion = """
        Conviene favorecer activos interiores, zonas con buena
        accesibilidad al transporte y alternativas de menor
        exposición térmica.
        """

    else:

        nivel = "🔴 Condiciones extremas"

        recomendacion = """
        Se recomienda priorizar experiencias interiores y activos
        con elevada accesibilidad mediante transporte público.
        La redistribución territorial puede ayudar a reducir
        la presión sobre las zonas más concentradas.
        """


    st.markdown(
        f"""
        <div class="warning-box">

        <h3>{nivel}</h3>

        <p>{recomendacion}</p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Recomendación territorial
    # --------------------------------------------------------

    st.markdown("### 🎯 Recomendación territorial")

    if (
        not df_barrios_filtrado.empty
        and "indice_oportunidad"
        in df_barrios_filtrado.columns
    ):

        recomendados = (
            df_barrios_filtrado
            .sort_values(
                "indice_oportunidad",
                ascending=False
            )
            .head(5)
        )

        for _, row in recomendados.iterrows():

            st.markdown(
                f"""
                **📍 {row['barrio']}**

                Índice de oportunidad:
                **{row['indice_oportunidad']:.1f}/100**
                """
            )


    st.info(
        """
        ⚠️ **Nota metodológica**

        La temperatura funciona aquí como variable contextual
        para simulación. El sistema no está afirmando una relación
        causal entre temperatura y congestión turística.

        En una siguiente versión puede integrarse una fuente
        meteorológica histórica/actual para convertir este módulo
        en un escenario basado en datos reales.
        """
    )


# ============================================================
# 12. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    """
    TUI Territorial Intelligence Dashboard · Reto 3 ·
    Prototipo académico · Máster Data Science, Big Data &
    Business Analytics
    """
)


# ============================================================
# RECOMENDADOR_PERSONALIZADO_V1
# ============================================================

@st.cache_data
def cargar_modelo_recomendacion():
    return pd.read_parquet(
        "data/Oro/Modelo_Recomendacion_Recursos_prueba.parquet"
    )


def calcular_distancias(lat_usuario, lon_usuario, lat, lon):
    radio_tierra = 6371

    lat1 = np.radians(lat_usuario)
    lon1 = np.radians(lon_usuario)
    lat2 = np.radians(pd.to_numeric(lat))
    lon2 = np.radians(pd.to_numeric(lon))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return radio_tierra * 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )


with tab6:
    st.header("🤖 Recomendador personalizado")

    st.caption(
        "Compara calidad, cercanía y tranquilidad estimada. "
        "La demanda procede del volumen de reseñas sintéticas "
        "y no representa ocupación en tiempo real."
    )

    modelo_recomendador = cargar_modelo_recomendacion()

    barrios_origen = sorted(
        modelo_recomendador["barrio"].dropna().unique()
    )

    indice_sol = (
        barrios_origen.index("Sol")
        if "Sol" in barrios_origen else 0
    )

    col1, col2 = st.columns(2)

    barrio_origen = col1.selectbox(
        "Barrio de partida",
        barrios_origen,
        index=indice_sol,
        key="rec_barrio"
    )

    origenes = (
        modelo_recomendador[
            modelo_recomendador["barrio"] == barrio_origen
        ]
        .reset_index(drop=True)
    )

    indice_origen = col2.selectbox(
        "Punto de partida",
        origenes.index.tolist(),
        format_func=lambda i: origenes.loc[i, "nombre"],
        key="rec_origen"
    )

    origen = origenes.loc[indice_origen]

    perfiles = {
        "Equilibrado": (45, 35, 20),
        "Priorizar calidad": (65, 20, 15),
        "Priorizar cercanía": (25, 60, 15),
        "Evitar lugares demandados": (30, 20, 50)
    }

    col3, col4, col5 = st.columns(3)

    perfil = col3.selectbox(
        "Preferencia",
        list(perfiles),
        key="rec_perfil"
    )

    categorias = [
        "Todas",
        *sorted(
            modelo_recomendador[
                "macro_categoria"
            ].dropna().unique()
        )
    ]

    categoria = col4.selectbox(
        "Categoría",
        categorias,
        format_func=lambda x: (
            x if x == "Todas"
            else str(x).replace("_", " ").title()
        ),
        key="rec_categoria"
    )

    radio_km = col5.slider(
        "Distancia máxima",
        0.5,
        10.0,
        2.0,
        0.5,
        format="%.1f km",
        key="rec_radio"
    )

    candidatos = modelo_recomendador[
        modelo_recomendador["n_resenas"] >= 3
    ].copy()

    if categoria != "Todas":
        candidatos = candidatos[
            candidatos["macro_categoria"] == categoria
        ]

    candidatos = candidatos[
        candidatos["id_origen"] != origen["id_origen"]
    ]

    candidatos["distancia_km"] = calcular_distancias(
        origen["lat"],
        origen["lon"],
        candidatos["lat"],
        candidatos["lon"]
    )

    candidatos = candidatos[
        candidatos["distancia_km"] <= radio_km
    ].copy()

    candidatos["cercania_pct"] = (
        100
        * (1 - candidatos["distancia_km"] / radio_km)
    ).clip(0, 100)

    peso_calidad, peso_cercania, peso_tranquilidad = (
        perfiles[perfil]
    )

    candidatos["score_recomendacion"] = (
        peso_calidad
        * candidatos["calidad_confianza_pct"]
        + peso_cercania
        * candidatos["cercania_pct"]
        + peso_tranquilidad
        * candidatos["tranquilidad_estimada_pct"]
    ) / 100

    recomendaciones = (
        candidatos
        .sort_values("score_recomendacion", ascending=False)
        .head(10)
    )

    if recomendaciones.empty:
        st.warning(
            "No hay recursos que cumplan estos criterios."
        )
    else:
        mejor = recomendaciones.iloc[0]

        st.success(
            f"Mejor opción: {mejor['nombre']} · "
            f"{mejor['distancia_km']:.2f} km · "
            f"{mejor['score_recomendacion']:.1f}/100"
        )

        tabla = recomendaciones[
            [
                "nombre",
                "barrio",
                "macro_categoria",
                "n_resenas",
                "rating_media",
                "distancia_km",
                "demanda_estimada_pct",
                "tranquilidad_estimada_pct",
                "score_recomendacion"
            ]
        ].copy()

        tabla.columns = [
            "Lugar",
            "Barrio",
            "Categoría",
            "Reseñas",
            "Valoración",
            "Distancia (km)",
            "Demanda estimada",
            "Tranquilidad estimada",
            "Recomendación"
        ]

        st.dataframe(
            tabla.round(2),
            hide_index=True,
            width="stretch"
        )
