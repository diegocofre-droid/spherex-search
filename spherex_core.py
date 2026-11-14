"""
spherex_core.py  -  PROCESAMIENTO CIENTÍFICO de SPHEREx Motion Explorer
=======================================================================

Es el mismo procedimiento de los notebooks del equipo (M101_SPHEREx_multiples
y Barnard_SPHEREx_autonomo) y del tutorial de IRSA:
https://caltech-ipac.github.io/irsa-tutorials/spherex-cutouts/

Es una herramienta EXPLORATORIA de inspección visual, no un detector validado
de variabilidad ni de movimiento.

Este módulo no sabe nada de Streamlit ni de matplotlib: devuelve datos.
Los gráficos y el GIF se dibujan en `spherex_backend.py`.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import requests
import astropy.units as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp

COLECCIONES = ["spherex_dr1", "spherex_qr3"]
CARPETA_FITS = Path("04_Datos_FITS")   # caché de recortes descargados
TOL_LAMBDA_UM = 0.05                   # tolerancia espectral (como en los notebooks)
RADIO_DIAGNOSTICO_PX = 6               # zona alrededor del objeto para NaN y FLAGS
PIXEL_ARCSEC = 6.2                     # tamaño aproximado de un píxel SPHEREx


# ----------------------------------------------------------------------
# Lectura de un recorte FITS
# ----------------------------------------------------------------------
def leer_recorte(ruta, ra, dec):
    """Lee un FITS y calcula la banda espectral en (RA, Dec), en grados.

    Devuelve (imagen, wcs, flags, metadatos).
    """
    ruta = Path(ruta)
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    with fits.open(ruta) as hdul:
        hdu = hdul["IMAGE"]
        imagen = np.asarray(hdu.data, dtype=float).copy()
        wcs = WCS(hdu.header).celestial
        x, y = wcs.world_to_pixel(coord)
        if not (0 <= x < imagen.shape[1] and 0 <= y < imagen.shape[0]):
            raise ValueError(f"{ruta.name}: las coordenadas no caen en el recorte.")

        # WCS-WAVE es alternativo y no debe aplicar la distorsión SIP espacial.
        wcs_espectral = WCS(hdu.header, fobj=hdul, key="W")
        wcs_espectral.sip = None
        longitud_onda, ancho_banda = wcs_espectral.pixel_to_world(x, y)
        meta = {
            "archivo": ruta.name,
            "lambda_um": float(longitud_onda.to_value(u.um)),
            "ancho_um": float(ancho_banda.to_value(u.um)),
            "x": float(x),
            "y": float(y),
            "unidad": str(hdu.header.get("BUNIT", "sin especificar")),
            "version": str(hdul[0].header.get("VERSION", "no indicada")),
        }
        # Se conserva FLAGS para el diagnóstico de calidad.
        flags = np.array(hdul["FLAGS"].data, copy=True) if "FLAGS" in hdul else None
    return imagen, wcs, flags, meta


def rellenar_huecos(imagen, x, y, radio=6, iteraciones=5):
    """Interpola los NaN cercanos a (x, y). Devuelve (imagen, mascara_huecos).

    Los valores interpolados NO son flujo medido: sirven para visualizar,
    no para fotometría.
    """
    salida = imagen.copy()
    yy, xx = np.indices(imagen.shape)
    zona = np.hypot(xx - x, yy - y) <= radio
    huecos = zona & ~np.isfinite(imagen)
    if not huecos.any():
        return salida, huecos
    kernel = Gaussian2DKernel(1.5)
    for _ in range(iteraciones):
        tmp = interpolate_replace_nans(salida, kernel, boundary="extend")
        salida[huecos] = tmp[huecos]
        if not (huecos & ~np.isfinite(salida)).any():
            break
    return salida, huecos


def diagnostico_zona(imagen, flags, x, y, radio=RADIO_DIAGNOSTICO_PX):
    """NaN y bits de FLAGS dentro de `radio` píxeles del objeto (como el notebook)."""
    yy, xx = np.indices(imagen.shape)
    zona = np.hypot(xx - x, yy - y) <= radio
    n_nan = int(np.sum(zona & ~np.isfinite(imagen)))
    bits = []
    if flags is not None:
        valores = np.unique(flags[zona])
        bits = sorted({b for v in valores for b in range(32) if int(v) >> b & 1})
    return n_nan, int(zona.sum()), bits


# ----------------------------------------------------------------------
# Objetos con movimiento propio
# ----------------------------------------------------------------------
def construir_estrella(ra, dec, pm):
    """pm = {"pm_ra": mas/año (ya incluye cos dec), "pm_dec": mas/año,
    "parallax": mas (opcional), "rv": km/s (opcional), "epoch": "J2000"}."""
    kw = dict(
        ra=ra * u.deg, dec=dec * u.deg,
        pm_ra_cosdec=pm["pm_ra"] * u.mas / u.yr,
        pm_dec=pm["pm_dec"] * u.mas / u.yr,
        frame="icrs",
        obstime=Time(pm.get("epoch", "J2000")),
    )
    if pm.get("parallax"):
        kw["distance"] = Distance(parallax=pm["parallax"] * u.mas)
        kw["radial_velocity"] = pm.get("rv", 0.0) * u.km / u.s
    return SkyCoord(**kw)


def posicion_en(estrella, tiempo):
    """Posición ICRS (sin movimiento propio adjunto) en la fecha dada."""
    p = estrella.apply_space_motion(new_obstime=tiempo)
    return SkyCoord(p.ra, p.dec, frame="icrs")


def centroide(img, x0, y0, r=4, iteraciones=3):
    """Centroide ponderado por flujo; la ventana se recentra en cada paso."""
    yy, xx = np.indices(img.shape)
    fondo = np.nanmedian(img[np.isfinite(img)])
    for _ in range(iteraciones):
        m = (np.hypot(xx - x0, yy - y0) <= r) & np.isfinite(img)
        if not m.any():
            return np.nan, np.nan
        w = np.clip(img[m] - fondo, 0, None)
        if w.sum() == 0:
            return np.nan, np.nan
        x0 = (xx[m] * w).sum() / w.sum()
        y0 = (yy[m] * w).sum() / w.sum()
    return x0, y0


def medir_movimiento(obs, wcs_ref):
    """Movimiento medido (centroides) frente al esperado por catálogo.

    Devuelve None si hay menos de 2 centroides. `pm_medido_mas` solo existe
    con 3 o más centroides y fechas distintas. Es exploratorio.
    """
    cents = [centroide(o["alineada"], *o["marcador"]) for o in obs]
    ok = [i for i, c in enumerate(cents) if np.all(np.isfinite(c))]
    if len(ok) < 2:
        return {"n_medidos": len(ok), "n_total": len(obs), "t": None}

    ref = obs[0]["pos"]

    def offsets(c):
        dra = (c.ra.deg - ref.ra.deg) * np.cos(np.radians(ref.dec.deg)) * 3600
        ddec = (c.dec.deg - ref.dec.deg) * 3600
        return dra, ddec

    medidos = np.array([offsets(wcs_ref.pixel_to_world(*cents[i])) for i in ok])
    esperados = np.array([offsets(obs[i]["pos"]) for i in ok])
    t = np.array([obs[i]["t"].jyear for i in ok])

    out = {"n_medidos": len(ok), "n_total": len(obs), "t": t,
           "medidos": medidos, "esperados": esperados, "pm_medido_mas": None}
    if len(ok) >= 3 and np.ptp(t) > 0:
        pm_ra, _ = np.polyfit(t - t[0], medidos[:, 0], 1)
        pm_dec, _ = np.polyfit(t - t[0], medidos[:, 1], 1)
        out["pm_medido_mas"] = (float(pm_ra * 1000), float(pm_dec * 1000))
    return out


# ----------------------------------------------------------------------
# Consulta y descarga (NASA/IPAC IRSA)
# ----------------------------------------------------------------------
def buscar_tabla(centro, radio, detector):
    resultados = Irsa.query_sia(pos=(centro, radio), collection=COLECCIONES)
    tabla = resultados[resultados["energy_bandpassname"] == f"SPHEREx-D{detector}"]
    tabla.sort("t_min")
    return tabla


def elegir_filas(tabla, n):
    """La observación más cercana a cada una de n fechas equiespaciadas."""
    t = np.asarray(tabla["t_min"], dtype=float)
    objetivos = np.linspace(t.min(), t.max(), min(n, len(tabla)))
    indices = sorted({int(np.argmin(np.abs(t - x))) for x in objetivos})
    return [tabla[i] for i in indices]


def descargar_recorte(fila, ra, dec, tam_deg):
    obs_id = str(fila["obs_id"])
    sub = CARPETA_FITS / ("DR1" if str(fila["obs_collection"]) == "spherex_dr1" else "QR3")
    sub.mkdir(parents=True, exist_ok=True)
    archivo = sub / f"{obs_id}_{ra:.4f}_{dec:.4f}_{tam_deg:.4f}.fits"
    if archivo.exists() and archivo.stat().st_size > 0:
        return archivo
    url = f"{fila['access_url']}?center={ra},{dec}&size={tam_deg}"
    r = requests.get(url, timeout=(20, 120))
    r.raise_for_status()
    with fits.open(io.BytesIO(r.content)) as hdul:   # comprobar que es FITS válido
        hdul.verify("exception")
    archivo.write_bytes(r.content)
    return archivo


# ----------------------------------------------------------------------
# Pipeline completo
# ----------------------------------------------------------------------
def procesar(ra, dec, fov_arcmin, n_obs, pm=None, detector=2, tol_um=TOL_LAMBDA_UM):
    """Consulta IRSA, descarga recortes, alinea y mide, igual que los notebooks.

    Todas las observaciones se alinean a la PRIMERA (la más antigua) y la
    región común se define con todas ellas. La compatibilidad espectral se
    informa por observación, pero no se descarta ninguna.
    Lanza ValueError / RuntimeError con mensajes claros si algo falla.
    """
    avisos = []
    estrella = construir_estrella(ra, dec, pm) if pm else None
    if estrella is not None:
        centro = posicion_en(estrella, Time("2026-01-01"))
        radio = 30 * u.arcsec          # amplio por el movimiento propio
    else:
        centro = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        radio = 1 * u.arcsec

    tabla = buscar_tabla(centro, radio, detector)
    if len(tabla) < 2:
        raise ValueError(
            f"Solo hay {len(tabla)} observaciones del detector D{detector} en esa posición; "
            "se necesitan al menos 2. Prueba otras coordenadas."
        )
    n_por_coleccion = {
        str(c).replace("spherex_", "").upper(): int((np.asarray(tabla["obs_collection"]) == c).sum())
        for c in COLECCIONES
    }

    tam_deg = fov_arcmin / 60.0
    obs = []
    for fila in elegir_filas(tabla, n_obs):
        t = Time(float(fila["t_min"]), format="mjd")
        pos = posicion_en(estrella, t) if estrella is not None else centro
        try:
            ruta = descargar_recorte(fila, pos.ra.deg, pos.dec.deg, tam_deg)
            imagen, wcs, flags, meta = leer_recorte(ruta, pos.ra.deg, pos.dec.deg)
        except Exception as e:  # noqa: BLE001 - una observación mala no debe tumbar todo
            avisos.append(f"Se omitió {fila['obs_id']}: {e}")
            continue
        obs.append({
            "obs_id": str(fila["obs_id"]),
            "coleccion": str(fila["obs_collection"]).replace("spherex_", "").upper(),
            "t": t, "pos": pos, "imagen": imagen, "wcs": wcs, "flags": flags, "meta": meta,
        })
    obs.sort(key=lambda o: o["t"].mjd)
    if len(obs) < 2:
        raise RuntimeError("No se pudieron descargar al menos 2 recortes válidos.")

    # --- Unidades: como en los notebooks, no se mezclan calibraciones distintas
    if len({o["meta"]["unidad"] for o in obs}) > 1:
        raise ValueError("Las imágenes tienen unidades BUNIT distintas; revisa la calibración.")
    unidad = obs[0]["meta"]["unidad"]

    # --- Compatibilidad espectral: referencia = la observación con más "vecinas"
    lambdas = np.array([o["meta"]["lambda_um"] for o in obs])
    cuenta = [(np.abs(lambdas - lam) <= tol_um).sum() for lam in lambdas]
    ref_esp = int(np.argmax(cuenta))
    for o, lam in zip(obs, lambdas):
        d_lam = abs(lam - lambdas[ref_esp])
        o["compatible"] = bool(d_lam <= tol_um)
        o["notas"] = []
        if d_lam > tol_um:
            o["notas"].append(
                f"λ difiere {d_lam:.2f} µm de la referencia ({lambdas[ref_esp]:.2f} µm): "
                "el cambio de brillo puede ser espectral, no temporal."
            )

    # --- Diagnóstico de NaN y FLAGS junto al objeto (antes de rellenar)
    for o in obs:
        o["n_nan"], o["n_zona"], o["bits_flags"] = diagnostico_zona(
            o["imagen"], o["flags"], o["meta"]["x"], o["meta"]["y"])

    # --- Rellenar NaN junto al objeto (antes de alinear; reproject los propaga)
    for o in obs:
        o["imagen_original"] = o["imagen"]
        o["imagen"], huecos = rellenar_huecos(o["imagen"], o["meta"]["x"], o["meta"]["y"])
        o["huecos"] = huecos
        o["n_rellenados"] = int(huecos.sum())
        if o["n_rellenados"]:
            o["notas"].append(
                f"{o['n_rellenados']} píxeles NaN junto al objeto fueron interpolados "
                "(no son flujo medido)."
            )

    # --- Alinear todo a la cuadrícula de la primera observación (por coordenadas del cielo)
    wcs_ref = obs[0]["wcs"]
    forma = obs[0]["imagen"].shape
    comunes = np.isfinite(obs[0]["imagen"])
    obs[0]["alineada"] = obs[0]["imagen"]
    obs[0]["huecos_al"] = obs[0]["huecos"]
    for o in obs[1:]:
        al, huella = reproject_interp((o["imagen"], o["wcs"]), wcs_ref,
                                      shape_out=forma, order="bilinear")
        hal, _ = reproject_interp((o["huecos"].astype(float), o["wcs"]), wcs_ref,
                                  shape_out=forma, order="nearest-neighbor")
        o["alineada"] = al
        o["huecos_al"] = np.nan_to_num(hal) > 0.5
        comunes &= np.isfinite(al) & (huella > 0.99)
    if comunes.sum() < 10:
        raise ValueError("Las observaciones no tienen una región común válida.")

    for o in obs:
        x, y = wcs_ref.world_to_pixel(o["pos"])
        o["marcador"] = (float(x), float(y))

    se_mueve = estrella is not None
    movimiento = medir_movimiento(obs, wcs_ref) if se_mueve else None
    desplazamiento = float(obs[0]["pos"].separation(obs[-1]["pos"]).arcsec) if se_mueve else 0.0

    return {
        "observaciones": obs,
        "wcs_ref": wcs_ref,
        "comunes": comunes,
        "n_encontradas": len(tabla),
        "n_por_coleccion": n_por_coleccion,
        "cobertura_comun_pct": float(comunes.mean() * 100),
        "unidad": unidad,
        "rango_lambda_um": float(lambdas.max() - lambdas.min()),
        "lambda_ref_um": float(lambdas[ref_esp]),
        "tol_um": tol_um,
        "detector": detector,
        "se_mueve": se_mueve,
        "desplazamiento_arcsec": desplazamiento,
        "movimiento": movimiento,
        "avisos": avisos,
    }
