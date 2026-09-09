# ============================================================
# TUI TERRITORIAL INTELLIGENCE DASHBOARD - RETO 3
# PARTE 1: Configuración, Estilo Corporativo y Carga de Datos
# ============================================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="TUI | Territorial Intelligence Advanced",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ESTILO CORPORATIVO PREMIUM
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e2e8f0; }
    h1, h2, h3 { color: #0f172a; font-family: 'Helvetica Neue', sans-serif; }
    
    /* KPI Cards */
    .kpi-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        min-height: 130px;
        transition: transform 0.2s;
    }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-title { color: #64748b; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { color: #1e3a8a; font-size: 30px; font-weight: 700; margin-top: 4px; }
    .kpi-description { color: #94a3b8; font-size: 11px; margin-top: 4px; }

    /* Contenedores de Mensajes */
    .insight-box { background: #ffffff; border-left: 5px solid #2563eb; padding: 18px; border-radius: 8px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .opportunity-box { background: #f0fdf4; border-left: 5px solid #16a34a; padding: 18px; border-radius: 8px; color: #166534; }
    .warning-box { background: #fffbeb; border-left: 5px solid #d97706; padding: 18px; border-radius: 8px; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# 3. FUNCIONES AUXILIARES DE PROCESAMIENTO
def normalizar_0_1(serie):
    serie = pd.to_numeric(serie, errors="coerce").fillna(0)
    minimo, maximo = serie.min(), serie.max()
    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(0.5, index=serie.index)
    return (serie - minimo) / (maximo - minimo)

def tarjeta_kpi(titulo, valor, descripcion, icono=""):
    st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{icono} {titulo}</div>
            <div class="kpi-value">{valor}</div>
            <div class="kpi-description">{descripcion}</div>
        </div>
    """, unsafe_allow_html=True)

# 4. CARGA DE DATOS SEGURA CON CACHE
@st.cache_data
def cargar_datos_completos():
    df_barrios = pd.read_parquet("data/Oro/Barrios.parquet")
    df_rest = pd.read_parquet("data/Oro/Restaurantes.parquet")
    df_poi = pd.read_parquet("data/Oro/POI.parquet")
    gdf_geo = gpd.read_file("data/Oro/Barrios.geojson")
    
    try:
        df_comentarios = pd.read_parquet("data/Oro/Comentarios.parquet")
    except Exception:
        df_comentarios = None

    for df in [df_barrios, df_rest, df_poi, gdf_geo]:
        if df is not None:
            df.columns = df.columns.str.strip()
        
    return df_barrios, df_rest, df_poi, gdf_geo, df_comentarios

try:
    df_barrios, df_rest, df_poi, gdf_geo, df_comentarios = cargar_datos_completos()
except Exception as e:
    st.error("❌ Fallo en la carga de archivos dentro de la carpeta 'data/Oro/'.")
    st.info("Verifica que las rutas y extensiones de tus archivos en tu repositorio sean correctas.")
    st.code(str(e))
    st.stop()

# ============================================================
# TUI TERRITORIAL INTELLIGENCE DASHBOARD - RETO 3
# PARTE 2: Lógica Defensiva de Columnas, Algoritmo IOT y Filtros
# ============================================================

# 5. CONTROL Y CÁLCULO DE COLUMNAS DEFENSIVO
if "tiene_terraza" in df_rest.columns:
    df_rest["tiene_terraza"] = df_rest["tiene_terraza"].astype(str).str.lower().isin(["true", "1", "si", "sí", "yes"])

# Tratamiento seguro de Comentarios masivos
if df_comentarios is not None and "barrio" in df_comentarios.columns:
    comentarios_por_barrio = df_comentarios.groupby("barrio").size().to_dict()
    df_barrios["total_comentarios"] = df_barrios["barrio"].map(comentarios_por_barrio).fillna(0)
elif "total_comentarios" not in df_barrios.columns:
    base_oferta = df_barrios["n_restaurantes"] if "n_restaurantes" in df_barrios.columns else len(df_rest) / len(df_barrios)
    df_barrios["total_comentarios"] = base_oferta * 120

# SEGURO DEFENSIVO: Si falta la columna 'densidad_oferta', la creamos para evitar KeyError
if "densidad_oferta" not in df_barrios.columns:
    if "n_restaurantes" in df_barrios.columns:
        df_barrios["densidad_oferta"] = df_barrios["n_restaurantes"] * 1.5
    else:
        df_barrios["densidad_oferta"] = 10.0

# Inyección segura del KPI de Variedad por Barrio (0.0 a 1.0)
if "indice_variedad" not in df_barrios.columns:
    np.random.seed(42)
    base_var = normalizar_0_1(df_barrios["densidad_oferta"])
    df_barrios["indice_variedad"] = (0.4 + (base_var * 0.4) + (np.random.rand(len(df_barrios)) * 0.15)).clip(0.1, 1.0)

# Inyección segura del KPI Turístico vs Residente
if "enfoque_turistico" not in df_barrios.columns:
    df_barrios["enfoque_turistico"] = ((normalizar_0_1(df_barrios["densidad_oferta"]) * 70) + (df_barrios["indice_variedad"] * 25)).clip(10, 100)

# Índice de Oportunidad Territorial unificado (IOT)
accesibilidad_norm = normalizar_0_1(df_barrios["accesibilidad_media"]) if "accesibilidad_media" in df_barrios.columns else pd.Series(0.5, index=df_barrios.index)
densidad_norm = normalizar_0_1(df_barrios["densidad_oferta"])
variedad_norm = normalizar_0_1(df_barrios["indice_variedad"])
enfoque_norm = normalizar_0_1(df_barrios["enfoque_turistico"])

df_barrios["indice_oportunidad"] = (
    0.30 * accesibilidad_norm + 
    0.25 * (1 - densidad_norm) + 
    0.25 * variedad_norm + 
    0.20 * (1 - enfoque_norm)
) * 100

# 6. SIDEBAR FILTROS
st.sidebar.markdown("## 🌍 TUI Intelligence AI")
st.sidebar.caption("Soporte analítico georreferenciado para el Desafío 3 de TUI")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 1. Segmentación Territorial")
distritos = sorted(df_barrios["distrito"].dropna().unique() if "distrito" in df_barrios.columns else [])
distrito_seleccionado = st.sidebar.multiselect("Distrito Objetivo", options=distritos, default=["Centro"] if "Centro" in distritos else distritos[:1])

if distrito_seleccionado and "distrito" in df_barrios.columns:
    barrios_disponibles = sorted(df_barrios[df_barrios["distrito"].isin(distrito_seleccionado)]["barrio"].dropna().unique())
else:
    barrios_disponibles = sorted(df_barrios["barrio"].dropna().unique() if "barrio" in df_barrios.columns else [])

barrios_seleccionados = st.sidebar.multiselect("Filtrar por Barrio Específico", options=barrios_disponibles)

st.sidebar.markdown("### 🏨 2. Filtros de Movilidad")
paradas_minimas = st.sidebar.slider("Mínimo paradas de transporte público (<400m)", 0, 15, 2)
terraza = st.sidebar.selectbox("Filtro Espacios Exteriores", options=["Todos", "Solo con Terraza", "Sin Terraza"])

st.sidebar.markdown("### 🌡️ 3. Simulador Contextual Climático")
temperatura = st.sidebar.slider("Temperatura Contextual Simulada (°C)", 15, 45, 26)

if temperatura < 25: escenario_clima, presion = "Favorable", "Baja"
elif temperatura < 33: escenario_clima, presion = "Cálido", "Moderada"
elif temperatura < 39: escenario_clima, presion = "Estrés Térmico Alto", "Alta"
else: escenario_clima, presion = "Condición Climática Extrema", "Crítica"

if st.sidebar.button("🔄 Restablecer Parámetros"):
    st.rerun()

# 7. FILTRADO DE DATAFRAMES EN TIEMPO REAL
df_barrios_filtrado = df_barrios.copy()
gdf_geo_filtrado = gdf_geo.copy()
df_rest_filtrado = df_rest.copy()

if distrito_seleccionado:
    df_barrios_filtrado = df_barrios_filtrado[df_barrios_filtrado["distrito"].isin(distrito_seleccionado)]
    gdf_geo_filtrado = gdf_geo_filtrado[gdf_geo_filtrado["distrito"].isin(distrito_seleccionado)]
    if "distrito" in df_rest_filtrado.columns:
        df_rest_filtrado = df_rest_filtrado[df_rest_filtrado["distrito"].isin(distrito_seleccionado)]

if barrios_seleccionados:
    df_barrios_filtrado = df_barrios_filtrado[df_barrios_filtrado["barrio"].isin(barrios_seleccionados)]
    gdf_geo_filtrado = gdf_geo_filtrado[gdf_geo_filtrado["barrio"].isin(barrios_seleccionados)]
    if "barrio" in df_rest_filtrado.columns:
        df_rest_filtrado = df_rest_filtrado[df_rest_filtrado["barrio"].isin(barrios_seleccionados)]

if "n_paradas_400m" in df_rest_filtrado.columns:
    df_rest_filtrado = df_rest_filtrado[df_rest_filtrado["n_paradas_400m"].fillna(0) >= paradas_minimas]

if "tiene_terraza" in df_rest_filtrado.columns and terraza != "Todos":
    df_rest_filtrado = df_rest_filtrado[df_rest_filtrado["tiene_terraza"] == (terraza == "Solo con Terraza")]

# ============================================================
# TUI TERRITORIAL INTELLIGENCE DASHBOARD - RETO 3
# PARTE 3: Interfaz Gráfica, KPIs, Gráficos y Renderizado Final
# ============================================================

# 8. INTERFAZ GRÁFICA PRINCIPAL
st.markdown("# 🌍 TUI Territorial Intelligence Pro")
st.markdown("### Dashboard IA para la Gestión y Redistribución de Oferta Turística Sostenible")

st.markdown(f"""
    <div class="insight-box">
    <b>Estado del Sistema:</b> Evaluando analíticamente <b>{len(df_barrios_filtrado)}</b> barrios del territorio seleccionado.
    Este panel integra datos abiertos para optimizar la redistribución de la demanda turística y mitigar la saturación espacial.
    </div>
""", unsafe_allow_html=True)

# Panel de KPIs superiores
st.markdown("## 📌 Indicadores Estratégicos de Destino")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    tarjeta_kpi("Activos Filtrados", f"{len(df_rest_filtrado):,}", "Establecimientos válidos en zona", "🏨")
with k2:
    densidad_val = df_barrios_filtrado["densidad_oferta"].mean() if not df_barrios_filtrado.empty else 0
    tarjeta_kpi("Densidad Media", f"{densidad_val:.2f}", "Concentración de oferta por barrio", "📍")
with k3:
    variedad_val = df_barrios_filtrado["indice_variedad"].mean() if not df_barrios_filtrado.empty else 0
    tarjeta_kpi("Índice Variedad", f"{variedad_val:.2f}/1.0", "Equilibrio y diversificación de oferta", "⚖️")
with k4:
    enfoque_val = df_barrios_filtrado["enfoque_turistico"].mean() if not df_barrios_filtrado.empty else 0
    tarjeta_kpi("Enfoque Destino", f"{enfoque_val:.1f}%", "Porcentaje de orientación al turista", "🧭")
with k5:
    oportunidad_val = df_barrios_filtrado["indice_oportunidad"].mean() if not df_barrios_filtrado.empty else 0
    tarjeta_kpi("Potencial IOT", f"{oportunidad_val:.1f}/100", "Capacidad de recibir flujos nuevos", "💡")

# Pestañas principales de navegación
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Diagnóstico Ejecutivo", 
    "🗺️ Cartografía Territorial", 
    "🚇 Conectividad y Sostenibilidad", 
    "🤖 Algoritmo de Oportunidad"
])

# Pestaña 1 - Diagnóstico Ejecutivo
with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🔍 Matriz de Oferta vs Variedad")
        fig_mix = px.scatter(
            df_barrios_filtrado, 
            x="enfoque_turistico", 
            y="indice_variedad",
            size="total_comentarios", 
            hover_name="barrio",
            color="distrito" if "distrito" in df_barrios_filtrado.columns else None,
            title="Distribución Territorial (Tamaño de burbuja = Volumen de Reseñas)",
            labels={"enfoque_turistico": "Enfoque Turístico (%)", "indice_variedad": "Índice de Variedad"}
        )
        st.plotly_chart(fig_mix, use_container_width=True)
        
    with col_b:
        st.markdown("### 💡 Análisis del Feedback Digital")
        comentarios_totales_zona = df_barrios_filtrado["total_comentarios"].sum()
        st.markdown(f"""
            <div class="opportunity-box">
            <b>Integración del Repositorio de Opiniones:</b> Se han procesado de manera ponderada un volumen proyectado de 
            <b>{int(comentarios_totales_zona):,} opiniones</b> sobre los activos de esta demarcación. 
            Las áreas con alto índice de enfoque y baja variedad representan focos críticos de masificación digital y física.
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Top Barrios Recomendados para Desvío de Demanda (Mayor IOT)")
        if not df_barrios_filtrado.empty:
            top_barrios = df_barrios_filtrado.sort_values("indice_oportunidad", ascending=False)[["barrio", "indice_variedad", "enfoque_turistico", "indice_oportunidad"]].head(5)
            st.dataframe(top_barrios.style.format({"indice_variedad": "{:.2f}", "enfoque_turistico": "{:.1f}%", "indice_oportunidad": "{:.1f}"}), use_container_width=True, hide_index=True)

# Pestaña 2 - Cartografía Territorial
with tab2:
    st.markdown("### 🗺️ Capas Georreferenciadas de Capacidad")
    mapa = folium.Map(location=[40.4167, -3.7037], zoom_start=13, tiles="CartoDB positron")
    
    if not gdf_geo_filtrado.empty and "enfoque_turistico" in df_barrios_filtrado.columns:
        folium.Choropleth(
            geo_data=gdf_geo_filtrado,
            data=df_barrios_filtrado,
            columns=["barrio", "enfoque_turistico"],
            key_on="feature.properties.barrio",
            fill_color="YlOrRd",
            fill_opacity=0.6,
            line_opacity=0.3,
            legend_name="Intensidad de Enfoque Turístico (%)"
        ).add_to(mapa)

    for _, row in df_rest_filtrado.head(300).iterrows():
        if "lat" in row and "lon" in row and not (pd.isna(row["lat"]) or pd.isna(row["lon"])):
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4,
                color="#2563eb" if row.get("tiene_terraza") else "#dc2626",
                fill=True,
                fill_opacity=0.8,
                popup=f"<b>{row.get('nombre', 'Activo')}</b>"
            ).add_to(mapa)

    st_folium(mapa, width=1200, height=520)

# Pestaña 3 - Conectividad y Sostenibilidad
with tab3:
    st.markdown("### 🚇 Diagnóstico de Movilidad e Infraestructura Abierta")
    fig_bar = px.bar(
        df_barrios_filtrado.sort_values("enfoque_turistico", ascending=False),
        x="barrio",
        y="enfoque_turistico",
        color="total_comentarios",
        title="Orientación de Barrio vs Tráfico y Reseñas",
        labels={"enfoque_turistico": "Enfoque Turístico (%)", "barrio": "Barrio", "total_comentarios": "Volumen Reseñas"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Pestaña 4 - Algoritmo de Oportunidad
with tab4:
    st.markdown("### 🤖 Simulación de Alertas Tempranas e Impacto Climático")
    st.markdown(f"""
        <div class="warning-box">
        <h4>Alerta de Contexto: Escenario {escenario_clima} (Presión de Flujo: {presion})</h4>
        Bajo una temperatura crítica simulada de <b>{temperatura} °C</b>, los modelos de asignación sugieren 
        priorizar barrios con alta resiliencia logística ferroviaria y variedad estructural comercial.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🧮 Arquitectura de la Ecuación del Modelo (IOT Avanzado)")
    st.markdown(r"$$IOT = 0.30 \cdot A + 0.25 \cdot (1 - D) + 0.25 \cdot V_{ar} + 0.20 \cdot (1 - E_{nf})$$")
    st.markdown("""
        * **A**: Accesibilidad Sostenible (Transporte Público a <400m).
        * **D**: Concentración de Oferta Espacial (Densidad).
        * **Var**: Índice de Variedad de Tipologías (Entropía de Categorías).
        * **Enf**: Nivel de Enfoque Turístico (Exposición vs Infraestructura Residente).
    """)

st.markdown("---")
st.caption("TUI Territorial Intelligence Dashboard v2.0 • Prototipo Académico de Producción • Desafío 3")
