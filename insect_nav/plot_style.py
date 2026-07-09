"""
plot_style.py — stile grafico condiviso per figure matplotlib publication-quality.

Adattato da un template generico (skill personale "matplotlib-publication-style",
usata anche nel repo ROS Visual-Navigation-Biorobotics, che userà insieme a
questo pacchetto gli stessi grafici in pubblicazione) per il pacchetto
`insect_nav`. Vedi ``docs/PLOTTING_STYLE_GUIDE.md`` per il "perché" di ogni
valore, in particolare la sezione 8 per le estensioni specifiche di questo
repo (colori popolazioni neurali, colormap sequenziale per valori continui).

Nessuna dipendenza dal resto del pacchetto.

Regola di thread/process-safety (vedi guida §6): tutte le funzioni qui sotto
usano solo l'API orientata a oggetti di matplotlib (``Figure``/``Axes``
espliciti, mai ``plt.figure()``/``plt.plot()``/``plt.savefig()`` globali).
``apply_style()`` è l'unica funzione che tocca stato globale (``plt.rcParams``):
va chiamata una volta sola, nel processo/thread principale, prima di generare
qualunque grafico.

Esempio d'uso — raster/traccia multi-neurone (popolazione KC):

    from insect_nav.plot_style import apply_style, new_figure, save_figure, \\
        add_legend, style_axes, POPULATION_COLORS

    apply_style()  # una volta sola, a inizio programma

    fig, ax = new_figure("error_vs_x")
    ax.plot(t, voltage, color=POPULATION_COLORS["KC"], label="KC voltage")
    style_axes(ax, xlabel="Time [ms]", ylabel="Voltage [mV]")
    add_legend(ax)
    save_figure(fig, "kc_voltage")  # scrive kc_voltage.png

Esempio d'uso — scatter PCA colorato per errore continuo (MAE):

    from insect_nav.plot_style import apply_style, new_figure, save_figure, \\
        style_axes, SEQUENTIAL_CMAP

    apply_style()
    fig, ax = new_figure("scatter")
    sc = ax.scatter(pc1, pc2, c=mae, cmap=SEQUENTIAL_CMAP, s=30, edgecolor="black", linewidth=0.3)
    fig.colorbar(sc, ax=ax, label="MAE")
    style_axes(ax, xlabel="PC1", ylabel="PC2")
    save_figure(fig, "pca_scatter")

Esempio d'uso — bar chart multi-categoria (N=10 categorie):

    from insect_nav.plot_style import apply_style, new_figure, save_figure, add_legend, \\
        get_category_style

    apply_style()

    categories = ["cat_a", "cat_b", ..., "cat_j"]  # 10 categorie
    colors, hatches, _ = get_category_style(len(categories))  # warning: N>8

    fig, ax = new_figure("bar_chart")
    x = range(len(categories))
    bars = ax.bar(x, values_mean, yerr=values_std, capsize=3,
                   color=colors, hatch=hatches, edgecolor="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=45, ha="right")
    style_axes(ax, ylabel="Metric")
    save_figure(fig, "metric_by_category")
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette colori (colorblind-safe, vedi guida §3)
# ---------------------------------------------------------------------------

#: Ruoli semantici fissi, sottoinsieme di Okabe-Ito (Okabe & Ito, 2008).
#: Stesso ruolo -> stesso colore in tutto il progetto. Non introdurre varianti locali.
COLORS = {
    "reference": "#0072B2",       # blu — riferimento/target/desiderato
    "actual": "#D55E00",          # vermiglio — eseguito/misurato
    "start": "#009E73",           # verde — marker inizio serie
    "end": "#CC79A7",             # porpora rossastro — marker fine serie
    "secondary": "#56B4E9",       # azzurro cielo — seconda metrica affiancata
    "tertiary": "#E69F00",        # arancione — terza serie quando serve
    "mean_reference": "#4D4D4D",  # grigio scuro — linee di media/soglia
    "grid": "#B0B0B0",            # grigio chiaro — griglia
}

#: Marker dedicati a start/end, indipendenti dal colore (identita' mai solo colore).
START_MARKER = "o"
END_MARKER = "s"

#: Palette estesa per N categorie (Paul Tol "muted", Tol 2021), ordine fisso:
#: la stessa categoria ha sempre lo stesso colore tra un run e l'altro, finche'
#: l'elenco di categorie non cambia. Oltre 8 colori la separazione CVD scende
#: sotto soglia sicura: usare sempre get_category_style() che aggiunge hatch.
CATEGORY_PALETTE = [
    "#332288",  # indigo
    "#88CCEE",  # ciano
    "#44AA99",  # verde acqua
    "#117733",  # verde
    "#999933",  # oliva
    "#DDCC77",  # sabbia
    "#CC6677",  # rosa
    "#882255",  # vino
    "#AA4499",  # viola
]

#: Colore per categorie "N/D" / fuori scala.
CATEGORY_PALETTE_NA = "#DDDDDD"

#: Hatch pattern per codifica secondaria nei bar chart quando N > 8 (guida §3).
HATCHES = ["", "//", "xx", "\\\\", "..", "oo", "++", "--", "**"]

#: Marker per codifica secondaria nei line/scatter chart quando N > 8.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

#: Soglia oltre la quale il colore da solo non basta piu' (vedi guida §3).
_SAFE_CATEGORICAL_LIMIT = 8

# ---------------------------------------------------------------------------
# ESTENSIONI INSECT-NAV (guida §8) — colori popolazioni neurali + colormap
# sequenziale per valori continui (scatter/heatmap colorati per errore/intensita')
# ---------------------------------------------------------------------------

#: Colore fisso per popolazione neurale, riusato in logger.py (raster, tracce
#: di tensione, correnti, spike cumulativi). Se piu' MBON vanno disegnati
#: insieme, usare CATEGORY_PALETTE/get_category_style per quelli aggiuntivi
#: invece di riusare "tertiary" per tutti (altrimenti indistinguibili).
POPULATION_COLORS = {
    "PN": COLORS["reference"],   # blu
    "KC": COLORS["actual"],      # vermiglio
    "APL": COLORS["start"],      # verde
    "MBON": COLORS["tertiary"],  # arancione
}

#: Colormap sequenziale per dati continui (es. scatter PCA colorato per MAE,
#: heatmap di intensita'). NON usare la palette categoriale per questi casi:
#: viridis e' percettivamente uniforme e colorblind-safe, standard per dati
#: continui (a differenza di Okabe-Ito/Tol che sono per categorie discrete).
SEQUENTIAL_CMAP = "viridis"


# ---------------------------------------------------------------------------
# Figsize standard per "famiglia" di grafico (guida §1: generare gia' alla
# dimensione fisica finale, mai ridimensionare dopo in LaTeX)
# ---------------------------------------------------------------------------

#: Larghezza colonna singola / doppia in pollici (IEEE/Elsevier tipico).
SINGLE_COLUMN_WIDTH_IN = 3.5
DOUBLE_COLUMN_WIDTH_IN = 7.0

FIGSIZES = {
    # Traiettoria XY: quadrata, cosi' x/y non vengono distorte visivamente.
    "trajectory": (6.0, 6.0),
    "trajectory_single_col": (3.5, 3.5),
    # Errore/metrica vs variabile indipendente (anche tracce voltage/current
    # nel tempo): panoramica, aspect ~2:1.
    "error_vs_x": (7.0, 3.5),
    "error_vs_x_single_col": (3.5, 2.6),
    # Bar chart multi-categoria: largo per ospitare N etichette ruotate.
    "bar_chart": (7.0, 4.5),
    # Multi-subplot verticale condiviso su x: altezza per-riga, vedi
    # new_figure(nrows=...).
    "multi_vertical_row_height": 2.2,
    "multi_vertical_width": 7.0,
    # Scatter quadrato (es. PCA 2D colorato per errore continuo).
    "scatter": (6.0, 6.0),
    # Heatmap panoramica (es. saveVerticalWeightingHeatmap).
    "heatmap": (7.0, 5.0),
}


# ---------------------------------------------------------------------------
# rcParams globali
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Imposta i rcParams globali secondo la guida di stile.

    Da chiamare UNA VOLTA SOLA a inizio programma, nel processo/thread
    principale, prima di generare qualunque grafico. E' l'unica funzione di
    questo modulo che tocca stato globale di matplotlib: farlo altrove o piu'
    volte da worker diversi non e' garantito sicuro (motivo per cui tutte le
    altre funzioni qui sotto lavorano solo su oggetti Figure/Axes espliciti).
    """
    matplotlib.rcParams.update({
        # Font — valori "raccomandato" della guida §1 (minimi: tick 7, label 8,
        # legenda 7, titolo 9 — usare quelli solo se lo spazio e' davvero stretto).
        "font.size": 9,
        "font.family": "sans-serif",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 11,

        # Linewidth/marker — guida §2.
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.8,
        "axes.linewidth": 0.8,
        "patch.linewidth": 0.8,
        "errorbar.capsize": 3,

        # Griglia — guida §5.
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,  # griglia sotto ai dati

        # dpi/export — guida §4.
        "figure.dpi": 100,        # dpi a schermo (interattivo); l'export usa savefig.dpi
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Legenda — guida §5.
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.3",
    })


# ---------------------------------------------------------------------------
# Creazione figure (API OO, sicura anche in multiprocessing)
# ---------------------------------------------------------------------------

def new_figure(family: str = "trajectory", nrows: int = 1, ncols: int = 1,
                single_column: bool = False, **subplots_kwargs):
    """Crea ``fig, ax`` (o ``fig, axes`` se ``nrows``/``ncols`` > 1) con figsize
    standardizzato per la famiglia di grafico richiesta.

    Usa sempre ``plt.subplots()`` (API OO) e non tocca la "current figure"
    globale in modo persistente: il chiamante riceve l'oggetto ``Figure`` e deve
    disegnarci sopra tramite gli ``Axes`` restituiti, mai tramite ``plt.plot``
    globale (vedi guida §6).

    Parameters
    ----------
    family:
        Una tra ``"trajectory"``, ``"error_vs_x"``, ``"bar_chart"``,
        ``"multi_vertical"``, ``"scatter"``, ``"heatmap"``. Determina il
        figsize di default.
    nrows, ncols:
        Passati a ``plt.subplots``. Per ``family="multi_vertical"`` l'altezza
        totale viene calcolata come ``nrows * multi_vertical_row_height``.
    single_column:
        Se True usa la variante a colonna singola (3.5in) quando disponibile
        per la famiglia scelta (trajectory/error_vs_x).
    subplots_kwargs:
        Altri kwargs passati direttamente a ``plt.subplots`` (es.
        ``sharex=True`` per un grafico multi-subplot).
    """
    if family == "trajectory":
        figsize = FIGSIZES["trajectory_single_col"] if single_column else FIGSIZES["trajectory"]
    elif family == "error_vs_x":
        figsize = FIGSIZES["error_vs_x_single_col"] if single_column else FIGSIZES["error_vs_x"]
    elif family == "bar_chart":
        figsize = FIGSIZES["bar_chart"]
    elif family == "scatter":
        figsize = FIGSIZES["scatter"]
    elif family == "heatmap":
        figsize = FIGSIZES["heatmap"]
    elif family == "multi_vertical":
        width = FIGSIZES["multi_vertical_width"]
        height = FIGSIZES["multi_vertical_row_height"] * max(nrows, 1)
        figsize = (width, height)
        subplots_kwargs.setdefault("sharex", True)
    else:
        raise ValueError(
            f"Famiglia di grafico sconosciuta: {family!r}. Attese: "
            "'trajectory', 'error_vs_x', 'bar_chart', 'multi_vertical', "
            "'scatter', 'heatmap'."
        )

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplots_kwargs)
    fig.patch.set_facecolor("white")
    return fig, axes


# ---------------------------------------------------------------------------
# Salvataggio coerente (guida §4/§5)
# ---------------------------------------------------------------------------

def save_figure(fig, path, formats: Sequence[str] = ("png",), dpi: int = 300) -> list[Path]:
    """Salva ``fig`` in uno o piu' formati con dpi/bbox/facecolor coerenti.

    Default: solo PNG a 300dpi (guida §4). ``path`` puo' avere o non avere
    estensione: viene sempre normalizzata a ``.<formato>`` per ciascun formato
    richiesto. Se per un caso specifico serve anche il vettoriale, basta
    passare ``formats=("png", "pdf")`` in quella chiamata.

    Ritorna la lista dei path effettivamente scritti.
    """
    path = Path(path)
    written = []
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # dpi si applica solo ai formati raster; e' innocuo passarlo anche ai
        # formati vettoriali (matplotlib lo ignora per pdf/svg).
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Helper assi/legenda (guida §5)
# ---------------------------------------------------------------------------

def style_axes(ax, xlabel: str | None = None, ylabel: str | None = None,
                title: str | None = None, grid: bool = True, equal: bool = False) -> None:
    """Applica label (con unita' incluse dal chiamante), griglia e aspect
    coerenti a un singolo Axes. Non chiama mai funzioni ``plt.*`` globali.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(grid, color=COLORS["grid"], linewidth=0.8, alpha=0.3)
    ax.set_axisbelow(True)


def add_legend(ax, loc: str = "best", ncol: int = 1, **kwargs):
    """Aggiunge una legenda con lo stile coerente della guida (frame, alpha,
    bordo). Da chiamare solo se l'Axes ha piu' di una serie con label (guida
    §5: mai una legenda per una singola serie).
    """
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) < 2:
        warnings.warn(
            "add_legend() chiamata con < 2 serie etichettate: la guida di "
            "stile prescrive di ometterla in quel caso (usa titolo/asse).",
            stacklevel=2,
        )
    return ax.legend(
        loc=loc,
        ncol=ncol,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.3",
        **kwargs,
    )


def add_start_end_markers(ax, x: Sequence[float], y: Sequence[float],
                           start_label: str = "Start", end_label: str = "End",
                           size: float = 70.0, zorder: int = 5):
    """Disegna i marker di inizio/fine serie con colore + forma fissi (mai
    solo colore, vedi guida §3). Utile per traiettorie o qualunque serie 2D
    dove ha senso evidenziare i punti estremi.
    """
    ax.scatter(x[0], y[0], marker=START_MARKER, s=size, color=COLORS["start"],
               edgecolor="black", linewidth=0.6, zorder=zorder, label=start_label)
    ax.scatter(x[-1], y[-1], marker=END_MARKER, s=size, color=COLORS["end"],
               edgecolor="black", linewidth=0.6, zorder=zorder, label=end_label)


# ---------------------------------------------------------------------------
# Palette per N categorie con codifica secondaria (guida §3)
# ---------------------------------------------------------------------------

def get_category_style(n: int) -> tuple[list[str], list[str], list[str]]:
    """Ritorna ``(colors, hatches, markers)``, ciascuno lungo ``n``, per un
    grafico con ``n`` categorie.

    I colori vengono ciclati su ``CATEGORY_PALETTE`` (9 colori Paul Tol
    "muted"). Se ``n > 8`` (soglia di separazione CVD sicura, guida §3) viene
    emesso un warning e vengono assegnati hatch/marker distinti a ogni
    categoria: il chiamante DEVE passarli a ``ax.bar(..., hatch=hatches[i])``
    o ``ax.plot(..., marker=markers[i])`` — il colore da solo non e' piu'
    sufficiente a quella cardinalita'.
    """
    if n <= 0:
        return [], [], []
    if n > _SAFE_CATEGORICAL_LIMIT:
        warnings.warn(
            f"get_category_style({n}): oltre {_SAFE_CATEGORICAL_LIMIT} categorie "
            "la separazione colore per daltonismo non e' piu' garantita da sola. "
            "Usa anche gli hatch/marker restituiti come codifica secondaria "
            "(vedi guida di stile §3).",
            stacklevel=2,
        )
    colors = [CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)] for i in range(n)]
    hatches = [HATCHES[i % len(HATCHES)] if n > _SAFE_CATEGORICAL_LIMIT else ""
               for i in range(n)]
    markers = [MARKERS[i % len(MARKERS)] for i in range(n)]
    return colors, hatches, markers
