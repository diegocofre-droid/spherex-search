"""SPHEREx Motion Explorer - interfaz Streamlit (v0.3: solo datos reales de IRSA)."""
import pandas as pd
import streamlit as st

import spherex_backend as be

st.set_page_config(page_title="SPHEREx Motion Explorer", page_icon="🔭", layout="centered")

# ---------------------------------------------------------------- estilo
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(ellipse at top, #121a3a 0%, #05070f 60%); color: #e8ecff; }
    h1, h2, h3 { color: #ffffff; letter-spacing: .3px; }
    .hero { padding: 1.2rem 0 .4rem 0; }
    .hero h1 { font-size: 2.1rem; margin-bottom: .1rem; }
    .hero p { color: #9fb0ff; margin-top: 0; }
    .real-badge { background:#052e22; border:1px solid #22c993; color:#9ff0d2;
                  padding:.55rem .8rem; border-radius:10px; font-size:.9rem; margin:.4rem 0 1rem 0; }
    .card { background: rgba(255,255,255,.04); border:1px solid rgba(140,160,255,.25);
            border-radius:14px; padding:.9rem 1rem; margin:.4rem 0; }
    .muted { color:#9fb0ff; font-size:.9rem; }
    div.stButton > button { background:#5b6dff; color:white; border:0; border-radius:10px;
                            padding:.6rem 1rem; font-weight:600; width:100%; }
    div.stButton > button:hover { background:#7a89ff; color:white; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- estado
first = next(iter(be.EXAMPLES))
st.session_state.setdefault("ra", be.EXAMPLES[first]["ra"])
st.session_state.setdefault("dec", be.EXAMPLES[first]["dec"])
st.session_state.setdefault("fov", 6)          # 0.1° = 6′, como en los notebooks
st.session_state.setdefault("example", first)
st.session_state.setdefault("result", None)
st.session_state.setdefault("detalle", None)

CUSTOM = "— Coordenadas propias —"


def _on_example():
    ex = be.EXAMPLES.get(st.session_state.example)
    if ex:
        st.session_state.ra = ex["ra"]
        st.session_state.dec = ex["dec"]


def _ejemplo_para(ra, dec):
    """El objeto de ejemplo cuyas coordenadas coinciden (da movimiento propio y nombre)."""
    for ex in be.EXAMPLES.values():
        if abs(ex["ra"] - ra) < 1e-4 and abs(ex["dec"] - dec) < 1e-4:
            return ex
    return None


@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def _buscar(ra, dec, fov, n_obs, pm, nombre):
    # En caché: repetir la misma búsqueda no vuelve a descargar ni a dibujar nada.
    return be.search_observations(ra, dec, fov, n_obs, pm=pm, nombre=nombre)


# ---------------------------------------------------------------- cabecera
st.markdown(
    '<div class="hero"><h1>🔭 SPHEREx Motion Explorer</h1>'
    "<p>Mira cómo cambia el cielo infrarrojo a lo largo del tiempo.</p></div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="real-badge">🛰️ <b>DATOS REALES</b> de SPHEREx (NASA/IPAC). '
    "Visualización exploratoria, no es un análisis científico validado.</div>",
    unsafe_allow_html=True,
)

with st.expander("¿Qué es esto? (explicación sencilla)"):
    st.write(
        "**SPHEREx** es un telescopio espacial de la NASA que fotografía todo el cielo en luz "
        "infrarroja (invisible para nuestros ojos), una y otra vez. Al comparar fotos tomadas en "
        "fechas distintas podemos notar si algo **se mueve**, como una estrella cercana o un "
        "objeto frío y lejano (la búsqueda del hipotético «Planeta X» usa esta idea). "
        "Esta página te deja elegir un punto del cielo y ver esa «película»."
    )

# ---------------------------------------------------------------- 1. buscar
st.header("1 · Buscar en el cielo")

st.selectbox(
    "Objeto de ejemplo",
    [CUSTOM] + list(be.EXAMPLES.keys()),
    key="example",
    on_change=_on_example,
    help="Elige uno para rellenar las coordenadas automáticamente.",
)
ex = be.EXAMPLES.get(st.session_state.example)
if ex:
    st.caption(ex["desc"])

c1, c2 = st.columns(2)
c1.number_input("Ascensión recta RA (grados, 0–360)", 0.0, 360.0, key="ra", format="%.5f", step=0.01)
c2.number_input("Declinación Dec (grados, −90 a +90)", -90.0, 90.0, key="dec", format="%.5f", step=0.01)

c3, c4 = st.columns(2)
c3.select_slider("Campo de visión (arcmin)", options=[2, 5, 6, 10, 20, 30, 60], key="fov",
                 help="Tamaño de la zona del cielo. Los notebooks usan 6′ (0.1°). "
                      "60 arcmin = 1 grado (el Sol mide 30 arcmin).")
n_obs = c4.slider("Cantidad de observaciones", 2, 12, 6,
                  help="Se eligen repartidas a lo largo del tiempo (DR1 + QR3, detector D2).")

if st.button("🔍 Buscar y crear animación"):
    ra, dec = float(st.session_state.ra), float(st.session_state.dec)
    ejemplo = _ejemplo_para(ra, dec)
    st.session_state.detalle = None
    guardado = be.cargar_pregenerado(ejemplo["nombre"]) if ejemplo else None
    if guardado is not None:               # M101 / Barnard: se muestra el GIF real ya hecho
        st.session_state.result = guardado
        st.rerun()
    ok = False
    try:
        with st.spinner("Consultando IRSA y descargando recortes… (puede tardar 1–2 min)"):
            st.session_state.result = _buscar(
                ra, dec, float(st.session_state.fov), n_obs,
                ejemplo["pm"] if ejemplo else None,
                ejemplo["nombre"] if ejemplo else None,
            )
        ok = True
    except Exception as e:  # noqa: BLE001 - mostrar error amable en vez de romper la página
        st.session_state.result = None
        st.error(f"No se pudo completar la búsqueda: {e}")
    if ok:
        st.rerun()

def mostrar_info(res):
    obs = res.observaciones
    # ---------------------------------------------------------------- 3. información
    st.header("3 · Información científica")

    m1, m2, m3 = st.columns(3)
    m1.metric("Observaciones", f"{len(obs)}", help=f"De {res.n_encontradas} halladas en IRSA "
              f"({', '.join(f'{k}: {v}' for k, v in res.n_por_coleccion.items())}).")
    m2.metric("Periodo", f"{obs[0].obs_date:%Y-%m} → {obs[-1].obs_date:%Y-%m}")
    m3.metric("Rango de λ", f"{res.rango_lambda_um:.3f} µm")

    m4, m5, m6 = st.columns(3)
    m4.metric("Cobertura común", f"{res.cobertura_comun_pct:.0f}%")
    m5.metric("Unidad", res.unidad)
    if res.se_mueve:
        m6.metric("Desplazamiento esperado", f"{res.desplazamiento_arcsec:.2f}″",
                  help="Entre la primera y la última fecha. 1 píxel SPHEREx ≈ 6.2″.")
    else:
        m6.metric("Detector", f"D{res.detector}")

    st.markdown(
        f'<div class="card muted">Detector D{res.detector} · '
        f"{res.n_encontradas} observaciones halladas en IRSA "
        f"({', '.join(f'{k}: {v}' for k, v in res.n_por_coleccion.items())}) · "
        f"{len(obs)} usadas, repartidas a lo largo del tiempo"
        "</div>",
        unsafe_allow_html=True,
    )

    df = pd.DataFrame(
        {
            "Fecha": [o.obs_date.isoformat() for o in obs],
            "Colección": [o.coleccion for o in obs],
            "ID": [o.obs_id for o in obs],
            "λ (µm)": [o.lambda_um for o in obs],
            "Ancho de banda (µm)": [o.ancho_um for o in obs],
            "Píxel (x, y)": [f"({o.x:.1f}, {o.y:.1f})" for o in obs],
            "Comparable": ["✅ sí" if o.compatible else "❌ no" for o in obs],
            "NaN a ≤6 px": [f"{o.n_nan} de {o.n_zona}" for o in obs],
            "Rellenados": [o.n_rellenados for o in obs],
            "Bits FLAGS": [", ".join(map(str, o.bits_flags)) or "—" for o in obs],
        }
    )
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"«Comparable» = la longitud de onda difiere menos de {res.tol_um} µm de la de referencia. "
               f"Unidad de las imágenes: {res.unidad} · versión de los datos: {obs[0].version}.")

    # ---------------------------------------------------------------- 4. figuras
    st.header("4 · Figuras de análisis")

    tabs = ["Comparación alineada", "Comparación inicial"]
    if res.se_mueve:
        tabs.append("Movimiento medido vs esperado")
    for tab, nombre in zip(st.tabs(tabs), tabs):
        with tab:
            if nombre == "Comparación alineada":
                st.image(res.fig_alineada, width="stretch")
                st.caption("Todas las imágenes alineadas a la primera por coordenadas del cielo y con la misma "
                           "escala. El contorno blanco marca los píxeles interpolados."
                           + (" El círculo cian es la posición esperada del objeto." if res.se_mueve else ""))
            elif nombre == "Comparación inicial":
                st.image(res.fig_inicial, width="stretch")
                st.caption("Cada imagen en su propio sistema de coordenadas y con su propia escala; "
                           "la cruz cian marca las coordenadas buscadas.")
            else:
                mov = res.movimiento or {}
                if res.fig_movimiento is None:
                    st.info(f"No hay suficientes centroides para medir el movimiento "
                            f"({mov.get('n_medidos', 0)} de {mov.get('n_total', len(obs))}).")
                else:
                    st.image(res.fig_movimiento, width="stretch")
                    st.caption(f"Centroides medidos: {mov['n_medidos']} de {mov['n_total']}.")
                    pm_m = mov.get("pm_medido_mas")
                    if pm_m is not None:
                        st.markdown(
                            f"**Movimiento propio medido (exploratorio):** RA·cos(dec) = {pm_m[0]:.0f} mas/año, "
                            f"Dec = {pm_m[1]:.0f} mas/año"
                        )
                        if res.pm_catalogo:
                            st.markdown(
                                f"**Catálogo:** RA·cos(dec) = {res.pm_catalogo['pm_ra']:.0f} mas/año, "
                                f"Dec = {res.pm_catalogo['pm_dec']:.0f} mas/año"
                            )

    # ---------------------------------------------------------------- 5. calidad
    st.subheader("Advertencias de calidad")
    for w in res.warnings:
        st.warning(w)
    for o in obs:
        for nota in o.notas:
            st.caption(f"• {o.obs_date.isoformat()}: {nota}")

    st.markdown(
        f'<div class="card muted">Fuente de datos: <b>{res.source or "—"}</b></div>',
        unsafe_allow_html=True,
    )
    st.caption("Proyecto del NASA Space Apps Challenge 2026 · Planeta X y SPHEREx · v0.3")


res = st.session_state.result
if res is None:
    st.info("Elige un objeto (o escribe coordenadas) y toca **Buscar**.")
    st.stop()

# ---------------------------------------------------------------- GIF real ya generado
if isinstance(res, be.Pregenerado):
    st.header("2 · Animación temporal")
    st.image(res.gif, caption=f"{res.nombre} · GIF real generado con los datos de SPHEREx "
                              "(notebooks del equipo)", width="stretch")
    if res.nombre == "Estrella de Barnard":
        st.caption("⭕ El círculo cian marca dónde se espera la estrella en cada fecha (según su movimiento propio).")
    if res.figuras:
        st.header("Figuras del análisis")
        for tab, (titulo, png) in zip(st.tabs([t for t, _ in res.figuras]), res.figuras):
            with tab:
                st.image(png, width="stretch")
    ejemplo = _ejemplo_para(float(st.session_state.ra), float(st.session_state.dec))
    if st.session_state.get("detalle") is None:
        if st.button("📊 Ver información detallada (consulta IRSA en vivo, 1–2 min)"):
            try:
                with st.spinner("Consultando IRSA y descargando recortes…"):
                    st.session_state.detalle = _buscar(
                        float(st.session_state.ra), float(st.session_state.dec),
                        float(st.session_state.fov), n_obs,
                        ejemplo["pm"] if ejemplo else None,
                        ejemplo["nombre"] if ejemplo else None)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo calcular la información detallada: {e}")
    if st.session_state.get("detalle") is not None:
        mostrar_info(st.session_state.detalle)
    else:
        st.caption(f"Fuente: {be.SOURCE}")
    st.stop()

obs = res.observaciones

# ---------------------------------------------------------------- 2. animación
st.header("2 · Animación temporal")
fps = st.slider("Velocidad (cuadros por segundo)", 0.5, 6.0, 2.0, 0.5)
st.image(
    be.make_gif(res.frames_png, fps),
    caption=f"{res.nombre} · {len(obs)} observaciones · RA {res.ra:.4f}°, Dec {res.dec:+.4f}° · "
            f"campo {res.fov_arcmin:g}′ · misma escala de intensidad en todos los cuadros",
    width="stretch",
)
if res.se_mueve:
    st.caption("⭕ El círculo cian marca dónde se espera el objeto en cada fecha (según su movimiento propio).")

idx = st.slider("Ver una observación individual", 1, len(obs), 1)
st.image(res.frames_png[idx - 1], caption=obs[idx - 1].etiqueta, width="stretch")

mostrar_info(res)
