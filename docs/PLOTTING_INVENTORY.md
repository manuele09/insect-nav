# Censimento grafici matplotlib nel repo `insect-nav`

Snapshot del censimento fatto il 2026-07-09, base di partenza per il restyling
descritto in `PLOTTING_STYLE_GUIDE.md`. Ricerca esaustiva su tutto il repo
(esclusi `.git`, `__pycache__`, `*.egg-info`, `.pytest_cache`, `builds_network*/`
— confermati puri artefatti di build GeNN/CUDA senza codice Python né immagini).

`insect-nav` è il pacchetto Python pip `insect_nav` (rete spiking/SNN per
navigazione visiva insect-inspired), usato come dipendenza dal progetto ROS
`Visual-Navigation-Biorobotics`. Nessuna traccia di seaborn/plotly/`pandas.plot()`
— solo matplotlib puro. Tutte le immagini salvate sono PNG (nessun PDF/SVG in
nessun punto). `matplotlib` è dipendenza obbligatoria in `pyproject.toml`.
Copertura test per il plotting: zero.

## Tabella riassuntiva (23 funzioni di plotting distinte)

| File:riga | Funzione | Cosa rappresenta | Tipo grafico |
|---|---|---|---|
| `insect_nav/logger.py:507` | `NetworkLogger.plot_raster` | Spike raster di una popolazione (PN/KC/APLN/MBON): tempo vs ID neurone | Scatter (raster) |
| `insect_nav/logger.py:526` | `NetworkLogger.plot_voltage_traces` | Tracce di potenziale di membrana (fino a 5 neuroni) di una popolazione | Line plot multi-traccia |
| `insect_nav/logger.py:545` | `NetworkLogger.plot_currents` | Correnti post-sinaptiche (fino a 5 neuroni) su una sinapsi (es. KC→MBON) | Line plot multi-traccia |
| `insect_nav/logger.py:564` | `NetworkLogger.plot_cumulative_spike_count` | Conteggio cumulativo di spike nel tempo per una popolazione | Line plot |
| `insect_nav/logger.py:584` | `NetworkLogger.plot_activity_summary` | Dashboard combinato: raster PN, raster KC, spike cumulativi KC, tensione APL, corrente KC→MBON, tensione per ogni MBON — sia come subplot combinato che come PNG individuali | Multi-subplot (scatter + line), 5+N_MBON pannelli |
| `insect_nav/logger.py:653` | `NetworkLogger._plot_neuron_spikes` | Helper interno di raster (usato solo da `plot_activity_summary`) | Scatter |
| `insect_nav/logger.py:661` | `NetworkLogger._save_individual_plot` | Ricostruisce ogni pannello del summary come figura singola, copiando a mano linee/scatter dall'axes sorgente | Line/scatter (ricostruito) |
| `insect_nav/logger.py:684` | `NetworkLogger.plot_cumulative_novelty` | Novelty cumulativa (cosine/pearson/euclidean) vs KC nuovi reclutati, doppio asse Y | Scatter, dual-axis |
| `insect_nav/logger.py:716` | `NetworkLogger.plot_instant_novelty` | Novelty per-frame + tasso di reclutamento KC, doppio asse Y | Line plot, dual-axis |
| `insect_nav/logger.py:748` | `NetworkLogger.plot_all_novelty` | Wrapper che chiama cumulative+instant per ogni metrica disponibile | (orchestratore) |
| `insect_nav/base.py:275` | `NeuralModelBase.plot_test_results` | Novelty vs angolo di shift per un frame di test, con linee verticali per direzione scelta dalla rete / direzione ottimale | Scatter + linee verticali di riferimento |
| `insect_nav/vision.py:116` | `saveVerticalWeightingHeatmap` | Heatmap diagnostica dell'effetto di `VERTICAL_WEIGHT` sul frame preprocessato, con colorbar e asse Y secondario | Heatmap (`imshow`) + colorbar + doppio asse Y |
| `insect_nav/tuning/pca_plotter.py:68` | `PCAPlotter.plot_all_individuals_2d` | Tutti gli individui della DE in spazio PCA 2D, colorati per errore (MAE) | Scatter + colorbar |
| `insect_nav/tuning/pca_plotter.py:115` | `PCAPlotter.plot_all_individuals_3d` | Come sopra ma PCA 3D | Scatter 3D + colorbar |
| `insect_nav/tuning/pca_plotter.py:176` | `PCAPlotter.plot_loadings_2d` | Loadings (contributo di ogni parametro) su PC1/PC2 | Bar chart (2 subplot) |
| `insect_nav/tuning/pca_plotter.py:232` | `PCAPlotter.plot_loadings_3d` | Come sopra ma PC1/PC2/PC3 | Bar chart (3 subplot) |
| `insect_nav/tuning/pca_plotter.py:300` | `PCAPlotter.plot_scree` | Varianza spiegata per componente + cumulativa | Bar chart + line (dual-axis) |
| `insect_nav/tuning/pca_plotter.py:375` | `PCAPlotter.plot_components_scatter_matrix` | Scatter per ogni coppia di PC (fino a 10 componenti), colorati per errore | Scatter + colorbar, N grafici separati |
| `insect_nav/tuning/pca_plotter.py:444` | `PCAPlotter.plot_components_error_correlation` | Correlazione di Pearson tra ogni PC e l'errore di rete | Bar chart |
| `insect_nav/tuning/pca_plotter.py:527` | `PCAPlotter.plot_mae_prediction` | MAE reale vs predetta da regressione lineare sui PC | Scatter + linea di identità |
| `insect_nav/tuning/pca_plotter.py:599` | `PCAPlotter.plot_regression_coefficients` | Coefficienti della regressione lineare per predire il MAE | Bar chart |
| `insect_nav/tuning/pca_plotter.py:682` | `PCAPlotter.plot_all` | Orchestratore: chiama tutti e 9 i metodi sopra | (orchestratore) |
| `insect_nav/tuning/tuner.py:320` | `Tuner.update_progress_plot` | Curva di fitness (errore del miglior individuo) per generazione della differential evolution | Line plot con marker |

### Import matplotlib morti (non plotting, ma da ripulire)

- `insect_nav/spiking.py:18` — `matplotlib.use("Agg")` eseguito come **side-effect a livello di modulo** ogni volta che `NeuralNetwork` viene importato, incondizionatamente (anti-pattern da libreria: dovrebbe scegliere il backend l'applicazione, non un modulo interno). `plt` importato ma mai usato in questo file.
- `insect_nav/parallel.py:4` — `import matplotlib.pyplot as plt` mai usato nel resto del file.

## Problemi di stile ricorrenti

1. **Nessuno stile condiviso**: zero `plt.style.use()`/`rcParams`. Ogni modulo reimposta font/colori/dpi da zero.
2. **DPI incoerente**: 300 (`logger.py`, `base.py`, `pca_plotter.py`) vs 200 (`vision.py`, unico caso) vs default/~100 (`tuner.py`, nessun `dpi=` passato).
3. **`bbox_inches="tight"` a macchia di leopardo**: presente in `vision.py`, in metà delle chiamate di `logger.py`, nella maggior parte (non tutte) di `pca_plotter.py`; assente in `base.py` e `tuner.py`.
4. **Fontsize hardcoded incoerenti tra moduli**: titoli 10 (`vision.py`), 12 (`logger.py` in alcuni punti), 14 (`logger.py`/`pca_plotter.py`), 20 (`base.py`).
5. **Colori hardcoded senza palette comune**: `"blue"/"green"/"purple"/"orange"/"red"` in `logger.py`; `"blue"/"red"/"limegreen"` in `base.py`; `'#1f77b4'/'#d62728'/'steelblue'`/viridis in `pca_plotter.py`; colore di default in `tuner.py`. Nessuna colorblind-safe per costruzione.
6. **Due stili API in competizione**: OO (`fig, ax = plt.subplots()`) in `logger.py`/`pca_plotter.py`; globale implicita (`plt.figure()`/`plt.plot()`) in `base.py`/`tuner.py`.
7. **Duplicazione di codice per grafici strutturalmente identici**: `plot_loadings_2d`/`plot_loadings_3d`; `plot_components_error_correlation`/`plot_regression_coefficients`; `plot_cumulative_novelty`/`plot_instant_novelty` (pattern twinx ripetuto); `plot_currents` standalone vs versione inline in `plot_activity_summary` (stesso dato, stile diverso, con/senza legenda).
8. **Legenda incoerente, un caso è un bug funzionale vero**: `tuner.py::update_progress_plot` chiama `plt.legend()` dopo `plt.plot(errors, ...)` **senza `label=`** — produce una legenda vuota o il warning matplotlib "No artists with labels found", non solo un problema di stile.
9. **Nessuna utility condivisa** per elementi ripetuti (colorbar, doppio asse Y/twinx, palette per errore continuo, bar chart con valore sopra le barre) — ogni occorrenza copiata a mano.

## Utility di plotting condivisa: non esiste

A differenza del repo ROS (dove `scripts/plot_style.py` esiste già), qui **non
c'è nulla da estendere**: va costruito ex novo (`insect_nav/plot_style.py`,
vedi la style guide) — verosimilmente adattando lo stesso modulo già creato
lato ROS/nella skill personale `matplotlib-publication-style`, dato che i due
repo useranno insieme gli stessi grafici in pubblicazione.

## Rischi di thread/process-safety

Il repo usa **solo multiprocessing, mai threading** per la parallelizzazione
(`parallel.py::ParallelNavigator`, `tuning/pm_teacher.py::PmTeacher`, e
indirettamente `tuner.py` via `scipy.optimize.differential_evolution(workers=-1)`).
matplotlib/pyplot non condivide stato tra processi separati, quindi l'uso
dell'API globale implicita non causa race condition tra processi come
accadrebbe con thread nello stesso processo — **rischio basso nello stato
attuale**. Punti di attenzione reali:

- `matplotlib.use("Agg")` in `spiking.py` è un side-effect di import non
  incapsulato: comportamento non deterministico se il pacchetto viene
  importato in un contesto con backend GUI già attivo nello stesso processo.
- Verificato esplicitamente: nessuna funzione di plotting è mai chiamata da
  un worker che esegue training/testing della rete durante la DE-tuning
  (`Tuner.train_network`/`test_network` non passano da `plot_*`) — lo
  scenario "plotting concorrente durante training parallelo" **non si
  verifica nel codice come scritto oggi**.
- L'unico punto con un `multiprocessing.Lock()` esplicito attorno al plotting
  è `tuner.py::update_progress_plot` — difensivo/ridondante rispetto al
  rischio reale (il callback gira solo nel processo principale), ma è
  l'unico modulo con una qualche consapevolezza esplicita del problema.

Raccomandazione: usare comunque sempre l'API OO (mai `plt.*` globale) per
consistenza con la style guide e per prevenzione strutturale, indipendentemente
dal fatto che il rischio concreto oggi sia basso — un futuro refactor che
introducesse thread (non solo processi) renderebbe il rischio reale.
