# Linea guida di stile per i grafici (matplotlib, publication-quality)

Questa guida definisce lo stile grafico condiviso da usare per **tutti** i
grafici matplotlib del repo che finiscono in un paper/report. L'implementazione
concreta è in `insect_nav/plot_style.py`. Regole 1-7 identiche alla skill
personale `matplotlib-publication-style` (usata anche nel repo ROS
`Visual-Navigation-Biorobotics`, che userà insieme a questo gli stessi
grafici in pubblicazione — da qui l'importanza di restare allineati); la
sezione 8 aggiunge le estensioni specifiche di questo repo.

Motivazione: un censimento del repo (vedi `PLOTTING_INVENTORY.md` per il
dettaglio completo) ha trovato 23 funzioni di plotting in 5 moduli
(`logger.py`, `base.py`, `vision.py`, `tuning/pca_plotter.py`, `tuning/tuner.py`),
zero stile condiviso, dpi incoerente (300/200/default), due stili di API
matplotlib in competizione (OO vs globale implicita), palette diverse per
concetti analoghi, e un bug funzionale vero (legenda vuota in
`tuner.py::update_progress_plot`).

## 1. Dimensioni dei font

La figura va generata già alla dimensione fisica finale con cui comparirà nel
documento (3.5 in colonna singola / 7.0 in colonna doppia, standard IEEE/Elsevier),
mai ridimensionata dopo in LaTeX/Word.

| Elemento              | Minimo | Raccomandato (default in `apply_style()`) |
|------------------------|:------:|:-------------------------------------------:|
| Tick label (assi)      | 7 pt   | 9 pt   |
| Axis label             | 8 pt   | 10 pt  |
| Legenda                | 7 pt   | 9 pt   |
| Titolo                 | 9 pt   | 11 pt  |
| Annotazioni/testo extra| 7 pt   | 8 pt   |

## 2. Linewidth e marker

Linee dati: minimo 1.5pt, raccomandato 2.0pt. Linee ausiliarie (griglia, medie,
riferimenti): 0.8-1.0pt. Marker: minimo 6pt. Errorbar: `capsize` ≥3pt.

## 3. Palette colori (colorblind-safe)

Palette **Okabe-Ito** per i ruoli semantici fissi (`plot_style.COLORS`):

| Ruolo | Colore | Hex | Uso tipico |
|---|---|---|---|
| `reference` | blu | `#0072B2` | valore atteso/target, riferimento |
| `actual` | vermiglio | `#D55E00` | valore misurato |
| `start` | verde | `#009E73` | marker inizio serie |
| `end` | porpora rossastro | `#CC79A7` | marker fine serie |
| `secondary` | azzurro cielo | `#56B4E9` | seconda grandezza affiancata |
| `tertiary` | arancione | `#E69F00` | terza serie |
| `mean_reference` | grigio scuro | `#4D4D4D` | linee di media/soglia |
| `grid` | grigio chiaro | `#B0B0B0` | griglia |

Palette estesa **Paul Tol "muted"** (`plot_style.CATEGORY_PALETTE`, 9 colori)
per >3 categorie; oltre 8 va aggiunta codifica secondaria (hatch/marker) via
`get_category_style(n)`. Stesso ruolo → stesso colore in ogni modulo del repo.

## 4. DPI e formato di export

**Solo PNG a 300dpi** di default (mai sotto 300 — oggi `vision.py` è a 200 e
`tuner.py` non specifica affatto `dpi`, entrambi da correggere). Vettoriale
(PDF/SVG) solo per casi specifici via `formats=("png","pdf")`, non di default.

## 5. Legenda, assi, griglia, bbox, facecolor

Legenda sempre se >1 serie, mai se 1 sola. Assi sempre con unità esplicite.
Griglia sempre attiva, sotto ai dati. `bbox_inches="tight"`, `pad_inches=0.05`.
`facecolor` bianco esplicito sempre.

## 6. API sempre orientata a oggetti (mai `plt.*` globale)

`fig, ax = plt.subplots(...)`, `ax.plot(...)`, `fig.savefig(...)`,
`plt.close(fig)` — mai `plt.figure()/plt.plot()/plt.savefig()/plt.close()`
senza argomenti. Nel censimento di questo repo il rischio concreto di
corruzione di stato è **basso** (la concorrenza qui è sempre multiprocessing,
mai threading, e i worker paralleli di `Tuner`/`ParallelNavigator`/`PmTeacher`
non chiamano mai funzioni di plotting durante il training — verificato
esplicitamente), ma la regola resta non negoziabile per consistenza con la
skill generale e per prevenzione strutturale se in futuro venisse introdotto
threading (vedi `PLOTTING_INVENTORY.md` per il dettaglio dell'analisi).

## 7. Non riscrivere codice esistente senza che sia richiesto

Applicare la guida a codice nuovo è sempre corretto; migrare codice esistente
è un refactor a sé — vedi piano di migrazione §8.

## 8. Estensioni specifiche insect-nav

### 8.1 Colori per popolazioni neurali (`plot_style.POPULATION_COLORS`)

I grafici di `logger.py` (raster, tracce di tensione, correnti, spike
cumulativi) identificano sempre una delle popolazioni PN/KC/APL/MBON (o una
sinapsi tra due di esse). Oggi i colori sono hardcoded in modo incoerente
(`"green"` per KC in un punto, `"purple"` per lo spike cumulativo KC in un
altro). Mapping fisso, colorblind-safe, riusando/estendendo `COLORS`:

| Popolazione | Colore | Hex |
|---|---|---|
| `PN` | blu | `COLORS["reference"]` `#0072B2` |
| `KC` | vermiglio | `COLORS["actual"]` `#D55E00` |
| `APL` | verde | `COLORS["start"]` `#009E73` |
| `MBON` | arancione | `COLORS["tertiary"]` `#E69F00` |

Se `plot_activity_summary` deve disegnare più MBON contemporaneamente (N>1),
usa `CATEGORY_PALETTE`/`get_category_style(n)` per gli MBON aggiuntivi invece
di riusare `tertiary` per tutti (altrimenti diventano indistinguibili).

### 8.2 Colormap sequenziale per valori continui (scatter colorati per errore)

`pca_plotter.py` colora gli scatter PCA per errore (MAE) continuo — questo
NON è un caso della palette categoriale Okabe-Ito/Tol (quelle sono per
categorie discrete), è un caso di **colormap sequenziale**. Uso già corretto
di `viridis` nel codice esistente (percettivamente uniforme, colorblind-safe,
uno dei pochi standard validati per dati continui) — **mantenerlo**, non
sostituirlo con la palette categoriale. `plot_style.SEQUENTIAL_CMAP = "viridis"`
è la costante di riferimento per qualunque nuovo scatter/heatmap colorato per
valore continuo in questo repo (inclusa la heatmap di `vision.py`, che oggi
non specifica esplicitamente la colormap — verificare quale usa di default e
allinearla a `viridis` se non c'è un motivo specifico per una diversa).

### 8.3 Coppie positivo/negativo (bar chart segno-dipendenti)

`pca_plotter.py::plot_components_error_correlation`/`plot_regression_coefficients`
usano hex `#1f77b4`/`#d62728` (blu/rosso di default matplotlib, non parte di
nessuna palette dichiarata) per barre positive/negative. Allinea a
`COLORS["reference"]` (positivo) / `COLORS["actual"]` (negativo) — stessa
coppia già usata per lo stesso scopo nel repo ROS (`gini/analyze_sun2012_dataset.py`),
blu/vermiglio è una delle coppie colorblind-safe standard per dati con segno.

### 8.4 Bug funzionale da correggere durante la migrazione (non solo stile)

`tuner.py::update_progress_plot` chiama `plt.legend()` senza che nessuna serie
abbia `label=` — produce una legenda vuota/warning matplotlib. Durante la
migrazione: aggiungere `label="Best fitness"` (o simile) alla chiamata
`ax.plot(errors, ...)`, così `add_legend()` ha effettivamente qualcosa da
mostrare (con 1 sola serie però la guida §5 dice di ometterla — quindi la
correzione più coerente è probabilmente RIMUOVERE la chiamata a legend
invece di aggiungerle un'etichetta, a meno che in futuro vengano aggiunte
altre serie allo stesso asse; lasciare la decisione finale a chi esegue la
migrazione, il punto è che lo stato attuale — legenda vuota silenziosa — va
comunque eliminato).

### 8.5 Import matplotlib morti da ripulire

- `insect_nav/spiking.py:18` — `matplotlib.use("Agg")` come side-effect di
  import, `plt` mai altrimenti usato nel file. Da valutare se rimuovere del
  tutto (lasciando che sia l'applicazione a scegliere il backend) o spostare
  la scelta del backend dentro `plot_style.apply_style()` stesso, così resta
  un solo punto che tocca lo stato globale del backend invece di un
  side-effect nascosto in un modulo che non fa plotting.
- `insect_nav/parallel.py:4` — `import matplotlib.pyplot as plt` mai usato,
  da rimuovere.

### 8.6 Piano di migrazione — ESEGUITO (2026-07-09)

Tutti i punti sotto sono stati migrati a `plot_style.py`, ciascuno da un
agente indipendente su un solo file (nessuna sovrapposizione), poi verificati
con `python3 -m py_compile` su tutti e con smoke test reale (dati sintetici,
in alcuni casi con un venv temporaneo per `sklearn`/`pandas`/`scipy` mancanti
in questo ambiente) dove possibile isolando il modulo dall'import
dell'intero pacchetto (`tqdm` non installato in questo ambiente di sviluppo).

1. **`insect_nav/logger.py`** — migrate tutte e 10 le funzioni. Consolidato
   `plot_cumulative_novelty`/`plot_instant_novelty` in un helper privato
   `_plot_dual_axis_novelty()`; `plot_activity_summary` ora richiama
   direttamente `plot_currents`/`plot_raster`/`plot_voltage_traces` per i
   pannelli individuali invece di ricostruirli copiando artisti da una
   figura già renderizzata. Eccezione documentata: il pannello cumulative
   spike count nel summary usa un asse tempo relativo (da 0) mentre
   `plot_cumulative_spike_count` standalone lo usa assoluto (da `_start_step`)
   — unificarli avrebbe cambiato il dato plottato, non solo lo stile, quindi
   `_save_individual_plot` è stato mantenuto solo per questo caso specifico.
   Colori popolazione/sinapsi mappati via `POPULATION_COLORS` con due helper
   `_population_color()`/`_synapse_color()`.
2. **`insect_nav/base.py`** (`plot_test_results`) — migrata a OO;
   `"red"`/`"limegreen"` per le due linee di riferimento lasciati
   **esattamente invariati** (vincolo di coerenza con `trajectory_plot.py`
   nel repo ROS, verificato dall'agente).
3. **`insect_nav/vision.py`** (`saveVerticalWeightingHeatmap`) — dpi 200→300,
   fontsize allineato, colormap resa esplicita a `SEQUENTIAL_CMAP` (era già
   viridis di default, ora esplicita per robustezza). Import lazy dentro la
   funzione preservato come pattern deliberato del file.
4. **`insect_nav/tuning/pca_plotter.py`** — migrati tutti e 9 i metodi.
   Consolidate le 2 coppie duplicate (`_plot_loadings()` per 2d/3d,
   `_plot_signed_bar_chart()` per error_correlation/regression_coefficients).
   `SEQUENTIAL_CMAP` applicato ai 3 scatter colorati per MAE continuo;
   `COLORS["reference"]`/`COLORS["actual"]` per le coppie positivo/negativo.
5. **`insect_nav/tuning/tuner.py`** (`update_progress_plot`) — migrata a OO;
   bug della legenda vuota risolto **rimuovendo** la chiamata (conferma
   dall'agente: una sola serie, coerente con guida §5); `multiprocessing.Lock`
   lasciato intatto.
6. **Pulizia import morti** (`spiking.py`, `parallel.py`, §8.5) — fatta
   direttamente (non da agente): rimosso `import matplotlib.pyplot as plt`
   inutilizzato da entrambi; **`matplotlib.use("Agg")` in `spiking.py`
   NON rimosso** (side-effect di import probabilmente intenzionale per
   garantire un backend headless in ambienti di training senza display —
   troppo rischioso da rimuovere senza conferma esplicita, resta come
   possibile follow-up separato se si vuole centralizzare la scelta del
   backend dentro `plot_style.apply_style()`).
