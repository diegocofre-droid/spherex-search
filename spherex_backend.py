"""
spherex_backend.py  -  CAPA DE DATOS de SPHEREx Motion Explorer
================================================================

Solo datos REALES de SPHEREx (NASA/IPAC IRSA), con el mismo procedimiento y
las mismas figuras que los notebooks del equipo:

  * GIF "blink" (misma escala de intensidad, ejes RA/Dec, círculo cian si el
    objeto se mueve),
  * comparación inicial (cada imagen en su propio WCS),
  * comparación alineada (misma escala, contorno de píxeles interpolados),
  * movimiento medido vs esperado (solo objetos con movimiento propio).

El procesamiento científico vive en spherex_core.py.
"""
from __future__ import annotations

import io
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # sin ventanas: se dibuja a memoria
import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from PIL import Image

import spherex_core as core

DETECTOR_REAL = 2  # los notebooks del equipo usan el detector D2

# Objetos de ejemplo. "pm" = movimiento propio para calcular dónde está el objeto
# en la fecha de cada observación.
EXAMPLES = {
    "M101 (galaxia espiral)": {
        "nombre": "M101",
        "ra": 210.80227, "dec": 54.34895, "pm": None,
        "desc": "Galaxia espiral lejana. No se mueve: sirve de referencia.",
    },
    "Estrella de Barnard": {
        "nombre": "Estrella de Barnard",
        "ra": 269.452075, "dec": 4.693391,
        # Gaia DR3, época J2000 (mismos valores del notebook del equipo)
        "pm": {"pm_ra": -802.803, "pm_dec": 10362.394, "parallax": 546.976,
               "rv": -110.6, "epoch": "J2000"},
        "desc": "Estrella cercana con uno de los mayores movimientos propios conocidos (~10″ por año).",
    },
    "WISE 0855-0714": {
        "nombre": "WISE 0855-0714",
        "ra": 133.79513, "dec": -7.24514,
        # VALORES APROXIMADOS escritos de memoria: el equipo debe verificarlos
        # (SIMBAD / Luhman & Esplin 2016) antes de usarlos en serio.
        "pm": {"pm_ra": -8100.0, "pm_dec": 700.0, "parallax": 439.0,
               "rv": 0.0, "epoch": "J2010.5"},
        "desc": "Objeto muy frío y cercano; casi sin luz visible, brilla en infrarrojo. Se mueve ~8″ por año.",
    },
}

SOURCE = "NASA/IPAC IRSA · SPHEREx (DR1 + QR3)"


@dataclass
class Observacion:
    obs_date: date
    coleccion: str
    obs_id: str
    lambda_um: float
    ancho_um: float
    x: float
    y: float
    unidad: str
    version: str
    compatible: bool
    notas: list[str]
    n_nan: int                   # NaN en IMAGE dentro del radio de diagnóstico
    n_zona: int                  # píxeles dentro de esa zona
    n_rellenados: int            # NaN interpolados (no son flujo medido)
    bits_flags: list[int]        # bits activos en FLAGS dentro de la zona
    etiqueta: str                # texto del título del cuadro del GIF


@dataclass
class Resultado:
    nombre: str
    ra: float
    dec: float
    fov_arcmin: float
    observaciones: list[Observacion]
    frames_png: list[bytes]      # un PNG por observación (los cuadros del GIF)
    fig_inicial: bytes           # PNG: cada imagen en su propio WCS
    fig_alineada: bytes          # PNG: alineadas, misma escala
    fig_movimiento: bytes | None # PNG: medido vs esperado (solo si se mueve)
    movimiento: dict | None
    se_mueve: bool
    desplazamiento_arcsec: float
    rango_lambda_um: float
    tol_um: float
    cobertura_comun_pct: float
    unidad: str
    n_encontradas: int
    n_por_coleccion: dict
    detector: int
    pm_catalogo: dict | None
    warnings: list[str] = field(default_factory=list)
    source: str = SOURCE


# ----------------------------------------------------------------------
# Dibujo (las mismas figuras de los notebooks)
# ----------------------------------------------------------------------
_LOCK = threading.Lock()   # pyplot no es seguro entre hilos de Streamlit


def _png(fig, dpi=150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _titulo(nombre, o) -> str:
    return f"{nombre} | {o['coleccion']} - {o['t'].strftime('%Y-%m-%d')} | {o['meta']['lambda_um']:.2f} µm"


def _grilla(n):
    ncols = min(n, 3)
    return ncols, int(np.ceil(n / ncols))


def _fig_inicial(out, nombre) -> bytes:
    """Comparación inicial: cada imagen en su propio WCS."""
    obs = out["observaciones"]
    ncols, nrows = _grilla(len(obs))
    fig = plt.figure(figsize=(4.5 * ncols, 4.5 * nrows))
    for i, o in enumerate(obs, start=1):
        meta = o["meta"]
        norm = ImageNormalize(o["imagen_original"], interval=PercentileInterval(99),
                              stretch=AsinhStretch())
        ax = fig.add_subplot(nrows, ncols, i, projection=o["wcs"])
        ax.imshow(o["imagen_original"], origin="lower", cmap="magma", norm=norm)
        ax.scatter(meta["x"], meta["y"], marker="+", color="cyan", s=130, linewidths=1.5)
        ax.set_title(f"{o['coleccion']} - {o['t'].strftime('%Y-%m-%d')}\n"
                     f"{meta['lambda_um']:.3f} µm", fontsize=9)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _png(fig, 150)


def _escala_comun(out):
    comunes = out["comunes"]
    valores = np.concatenate([o["alineada"][comunes] for o in out["observaciones"]])
    vmin, vmax = np.percentile(valores, [1, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError("No se pudo establecer una escala de intensidad válida.")
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())


def _mapa():
    mapa = plt.colormaps["magma"].copy()
    mapa.set_bad("black")
    return mapa


def _fig_alineada(out, nombre, norm) -> bytes:
    """Comparación alineada, misma escala; círculo cian y contorno de lo interpolado."""
    obs, comunes, wcs_ref = out["observaciones"], out["comunes"], out["wcs_ref"]
    n = len(obs)
    ncols, nrows = _grilla(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows),
                             subplot_kw={"projection": wcs_ref}, squeeze=False)
    axes = axes.ravel()
    im = None
    for ax, o in zip(axes, obs):
        im = ax.imshow(np.where(comunes, o["alineada"], np.nan), origin="lower",
                       cmap="magma", norm=norm)
        if out["se_mueve"]:
            px, py = o["marcador"]
            ax.scatter(px, py, marker="o", facecolors="none", edgecolors="cyan",
                       s=220, linewidths=1.2)
        if o["huecos_al"].any():
            ax.contour(o["huecos_al"].astype(float), levels=[0.5],
                       colors="white", linewidths=0.8)
        ax.set_title(f"{nombre} | {o['coleccion']} - {o['t'].strftime('%Y-%m-%d')}", fontsize=9)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
    for ax in axes[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=list(axes[:n]), shrink=0.8,
                 label=f"Intensidad ({out['unidad']})")
    return _png(fig, 150)


def _fig_movimiento(mov) -> bytes:
    """Desplazamiento medido (centroide) vs esperado (catálogo)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, k, nombre in zip(axes, (0, 1), ("RA*cos(dec)", "Dec")):
        ax.plot(mov["t"], mov["esperados"][:, k], "-", color="gray", label="esperado (catálogo)")
        ax.plot(mov["t"], mov["medidos"][:, k], "o", color="tab:orange", label="medido (centroide)")
        ax.set_xlabel("Fecha (año)")
        ax.set_ylabel(f"Desplazamiento en {nombre} (arcsec)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return _png(fig, 150)


def _frames_blink(out, nombre, norm) -> list[bytes]:
    """Los cuadros del GIF de los notebooks: figura 7x7, ejes RA/Dec, misma escala."""
    obs, comunes, wcs_ref = out["observaciones"], out["comunes"], out["wcs_ref"]
    cuadros = [np.ma.masked_where(~comunes, o["alineada"]) for o in obs]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection=wcs_ref)
    vista = ax.imshow(cuadros[0], origin="lower", cmap=_mapa(), norm=norm,
                      interpolation="nearest")
    marcador = None
    if out["se_mueve"]:
        marcador = ax.scatter([obs[0]["marcador"][0]], [obs[0]["marcador"][1]], marker="o",
                              facecolors="none", edgecolors="cyan", s=220, linewidths=1.2)
    ax.set_xlabel("Ascensión recta (RA)")
    ax.set_ylabel("Declinación (Dec)")
    titulo = ax.set_title(_titulo(nombre, obs[0]))

    frames = []
    try:
        for o, cuadro in zip(obs, cuadros):
            vista.set_data(cuadro)
            if marcador is not None:
                marcador.set_offsets([list(o["marcador"])])
            titulo.set_text(_titulo(nombre, o))
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG")
            frames.append(buf.getvalue())
    finally:
        plt.close(fig)
    return frames


def make_gif(frames_png: list[bytes], fps: float = 2.0) -> bytes:
    """Une los cuadros en un GIF animado con una paleta común (evita parpadeos de color)."""
    ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames_png]
    w, h = ims[0].size
    mosaico = Image.new("RGB", (w, h * len(ims)))
    for i, im in enumerate(ims):
        mosaico.paste(im, (0, i * h))
    paleta = mosaico.quantize(256)
    q = [im.quantize(palette=paleta, dither=Image.Dither.NONE) for im in ims]
    buf = io.BytesIO()
    q[0].save(buf, format="GIF", save_all=True, append_images=q[1:],
              duration=int(1000 / max(fps, 0.1)), loop=0)
    return buf.getvalue()


# ----------------------------------------------------------------------
# GIFs YA GENERADOS con los notebooks (se muestran tal cual, sin recalcular)
# ----------------------------------------------------------------------
# Son los archivos que guardan los notebooks. Copialos junto a app.py, en la
# carpeta 05_Analisis (los GIF) y en la raiz (los PNG), con estos nombres.
PREGENERADOS = {
    "M101": {
        "gif": "05_Analisis/M101_blink.gif",
        "figuras": [
            ("Comparación alineada", "M101_comparacion_alineada.png"),
            ("Comparación inicial", "M101_D2_comparacion_inicial.png"),
        ],
    },
    "Estrella de Barnard": {
        "gif": "05_Analisis/Barnard_blink.gif",
        "figuras": [
            ("Comparación alineada", "Barnard_comparacion_alineada.png"),
            ("Comparación inicial", "Barnard_D2_comparacion_inicial.png"),
            ("Movimiento medido vs esperado", "Barnard_movimiento_medido.png"),
        ],
    },
}


@dataclass
class Pregenerado:
    nombre: str
    gif: bytes
    figuras: list            # [(titulo, bytes_png)] solo las que existen
    archivo_gif: str


def _buscar_archivo(rel: str) -> Path | None:
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        p = base / rel
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def cargar_pregenerado(nombre: str) -> Pregenerado | None:
    """Devuelve el GIF real guardado de ese objeto, o None si el archivo no esta."""
    cfg = PREGENERADOS.get(nombre)
    if not cfg:
        return None
    gif = _buscar_archivo(cfg["gif"])
    if gif is None:
        return None
    figuras = []
    for titulo, rel in cfg["figuras"]:
        p = _buscar_archivo(rel) or _buscar_archivo(f"05_Analisis/{Path(rel).name}")
        if p is not None:
            figuras.append((titulo, p.read_bytes()))
    return Pregenerado(nombre, gif.read_bytes(), figuras, str(cfg["gif"]))


# ----------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------
def search_observations(ra: float, dec: float, fov_arcmin: float, n_obs: int = 6,
                        pm: dict | None = None, nombre: str | None = None) -> Resultado:
    """Consulta IRSA, procesa y dibuja todo. Es lo que usa la interfaz."""
    nombre = nombre or f"RA {ra:.4f}° Dec {dec:+.4f}°"
    out = core.procesar(ra, dec, fov_arcmin, n_obs, pm=pm, detector=DETECTOR_REAL)

    with _LOCK:
        norm = _escala_comun(out)
        frames = _frames_blink(out, nombre, norm)
        fig_ini = _fig_inicial(out, nombre)
        fig_al = _fig_alineada(out, nombre, norm)
        mov = out["movimiento"]
        fig_mov = _fig_movimiento(mov) if (mov and mov.get("t") is not None) else None

    observaciones = []
    for o in out["observaciones"]:
        m = o["meta"]
        observaciones.append(Observacion(
            obs_date=o["t"].to_datetime().date(), coleccion=o["coleccion"], obs_id=o["obs_id"],
            lambda_um=round(m["lambda_um"], 4), ancho_um=round(m["ancho_um"], 4),
            x=m["x"], y=m["y"], unidad=m["unidad"], version=m["version"],
            compatible=o["compatible"], notas=list(o["notas"]),
            n_nan=o["n_nan"], n_zona=o["n_zona"], n_rellenados=o["n_rellenados"],
            bits_flags=o["bits_flags"], etiqueta=_titulo(nombre, o),
        ))

    warnings = [
        "Herramienta exploratoria de inspección visual: NO es un detector validado de "
        "movimiento ni de variabilidad.",
        "No se igualan PSF, respuesta espectral, fondos ni calibraciones relativas entre observaciones.",
    ]
    if out["rango_lambda_um"] > out["tol_um"]:
        warnings.append(
            f"Las imágenes están en longitudes de onda distintas (rango {out['rango_lambda_um']:.3f} µm): "
            "el cambio de brillo puede ser espectral, no solo temporal"
            + (" ni solo movimiento." if out["se_mueve"] else ".")
        )
    if any(o.n_rellenados for o in observaciones):
        warnings.append(
            "Se interpolaron píxeles NaN junto al objeto; esos valores no son flujo medido "
            "(en la comparación alineada se marcan con un contorno blanco)."
        )
    if out["se_mueve"]:
        warnings.append(
            "El círculo cian marca la posición esperada del objeto en la fecha de cada "
            "observación, calculada con su movimiento propio de catálogo."
        )
    warnings += out["avisos"]

    return Resultado(
        nombre=nombre, ra=ra, dec=dec, fov_arcmin=fov_arcmin, observaciones=observaciones,
        frames_png=frames, fig_inicial=fig_ini, fig_alineada=fig_al, fig_movimiento=fig_mov,
        movimiento=mov, se_mueve=out["se_mueve"], desplazamiento_arcsec=out["desplazamiento_arcsec"],
        rango_lambda_um=out["rango_lambda_um"], tol_um=out["tol_um"],
        cobertura_comun_pct=out["cobertura_comun_pct"], unidad=out["unidad"],
        n_encontradas=out["n_encontradas"], n_por_coleccion=out["n_por_coleccion"],
        detector=out["detector"], pm_catalogo=pm, warnings=warnings,
    )
