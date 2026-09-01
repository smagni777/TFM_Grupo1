# ============================================================
# TUI TERRITORIAL INTELLIGENCE DASHBOARD
# Reto 3 - Máster Data Science / Big Data & Business Analytics
# AI-Dashboard para la gestión de oferta turística
# georreferenciada e integración con datos abiertos
# ============================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
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

    # Accesibilidad
    if "accesibilidad_media" in df_barrios.columns:
        accesibilidad_norm = normalizar_0_1(
            df_barrios["accesibilidad_media"]
        )
    else:
        accesibilidad_norm = pd.Series(
            0.5, index=df_barrios.index
        )

    # Densidad de oferta
    if "densidad_oferta" in df_barrios.columns:
        densidad_norm = normalizar_0_1(
            df_barrios["densidad_oferta"]
        )
    else:
        densidad_norm = pd.Series(
            0.5, index=df_barrios.index
        )

    # Número de restaurantes
    if "n_restaurantes" in df_barrios.columns:
        oferta_norm = normalizar_0_1(
            df_barrios["n_restaurantes"]
        )
    else:
        oferta_norm = pd.Series(
            0.5, index=df_barrios.index
        )

    # --------------------------------------------------------
    # Índice de oportunidad
    #
    # Mayor accesibilidad = mejor
    # Menor densidad = mejor
    # Menor concentración de oferta = mejor
    # --------------------------------------------------------

    df_barrios["indice_oportunidad"] = (
        0.45 * accesibilidad_norm
        + 0.35 * (1 - densidad_norm)
        + 0.20 * (1 - oferta_norm)
    ) * 100


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
    "Activos turísticos",
    options=["Restaurantes", "POI"],
    default=["Restaurantes"]
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
        "Oferta turística",
        f"{len(df_rest_filtrado):,}",
        "Activos que cumplen los filtros seleccionados",
        "🏨"
    )


# ------------------------------------------------------------
# KPI 2
# ------------------------------------------------------------

if (
    not df_barrios_filtrado.empty
    and "densidad_oferta" in df_barrios_filtrado.columns
):

    densidad_media = (
        df_barrios_filtrado["densidad_oferta"]
        .mean()
    )

else:

    densidad_media = 0


with k2:

    tarjeta_kpi(
        "Densidad media",
        f"{densidad_media:.1f}",
        "Oferta turística por superficie",
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Resumen ejecutivo",
        "🗺️ Mapa territorial",
        "🚇 Oferta y accesibilidad",
        "💡 Oportunidades",
        "🤖 Escenarios"
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

    # Centro de Madrid
    mapa = folium.Map(
        location=[40.4167, -3.7037],
        zoom_start=13,
        tiles="CartoDB positron"
    )


    # --------------------------------------------------------
    # Choropleth
    # --------------------------------------------------------

    if (
        not gdf_geo_filtrado.empty
        and "densidad_oferta" in df_barrios_filtrado.columns
    ):

        folium.Choropleth(
            geo_data=gdf_geo_filtrado,
            data=df_barrios_filtrado,
            columns=[
                "barrio",
                "densidad_oferta"
            ],
            key_on="feature.properties.barrio",
            fill_color="YlOrRd",
            fill_opacity=0.65,
            line_opacity=0.25,
            legend_name="Densidad de oferta turística"
        ).add_to(mapa)


    # --------------------------------------------------------
    # Restaurantes
    # --------------------------------------------------------

    for _, row in df_rest_filtrado.head(500).iterrows():

        if (
            "lat" not in row
            or "lon" not in row
        ):
            continue

        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue

        terraza_icon = (
            "Sí"
            if row.get("tiene_terraza", False)
            else "No"
        )

        popup_text = f"""
        <b>{row.get('nombre', 'Activo turístico')}</b><br>
        Paradas <400m:
        {row.get('n_paradas_400m', 'N/D')}<br>
        Terraza: {terraza_icon}
        """

        folium.CircleMarker(
            location=[
                row["lat"],
                row["lon"]
            ],
            radius=5,
            color="#1967D2",
            fill=True,
            fill_opacity=0.75,
            popup=popup_text
        ).add_to(mapa)


    st_folium(
        mapa,
        width=1200,
        height=600
    )

    st.caption(
        "El mapa muestra hasta 500 activos para preservar "
        "el rendimiento de la visualización."
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
            and "densidad_oferta" in df_barrios_filtrado.columns
        ):

            fig_densidad = px.bar(
                df_barrios_filtrado
                .sort_values(
                    "densidad_oferta",
                    ascending=False
                ),
                x="barrio",
                y="densidad_oferta",
                title="Densidad de oferta por barrio",
                labels={
                    "densidad_oferta":
                    "Densidad de oferta"
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
            and "densidad_oferta"
            in df_barrios_filtrado.columns
        ):

            fig_scatter = px.scatter(
                df_barrios_filtrado,
                x="accesibilidad_media",
                y="densidad_oferta",
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
                    "densidad_oferta":
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
            **Densidad de oferta**

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
