# Guida all'uso della rete spiking (`NeuralNetwork`)

Guida lato utente a `insect_nav.spiking.NeuralNetwork` — come creare, allenare
e testare la rete spiking mushroom-body (PN→KC→APL→MBON, su GeNN/pygenn), CPU
o GPU, con o senza batching. Non copre l'architettura interna della rete né i
dettagli di implementazione GeNN — solo l'API che si usa dall'esterno.

## Requisiti

`NeuralNetwork` richiede `pygenn` (`pip install insect_nav[genn]`). Su questa
macchina pygenn non è installato sull'host: va eseguito dentro il container
distrobox `insect-navContainer`:

```bash
distrobox enter insect-navContainer -- python3 il_tuo_script.py
```

## Creare una rete

```python
from insect_nav import NeuralNetwork
from insect_nav.parameters import load_parameters_from_file

params = load_parameters_from_file("percorso/parameters.json")
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True})
```

Parametri principali del costruttore:

| Parametro | Default | Significato |
|---|---|---|
| `parameters` | — | dict/`NetworkConfig` con tutti gli iperparametri (di solito caricato da `parameters.json` via `load_parameters_from_file`) |
| `load_net` | `{"pn_kc": False, "kc_mbon": False}` | quali pesi caricare da disco (`weightsPath/pn_kc_ind.npy`, `weightsPath/kc_mbon_g_0.npy`). `False` = rete nuova non allenata |
| `use_gpu` | `False` | `True` = backend `cuda`, `False` = backend `single_threaded_cpu` |
| `batch_size` | `1` | quante presentazioni processare in parallelo in una singola simulazione (solo GPU, vedi sotto) |
| `precompute_features` | `False` | precalcola le feature di tutto il dataset una volta sola al costruttore (vedi sotto) |
| `reducedNetwork` | `False` | se `True`, costruisce solo PN→KC→APL (niente MBON/plasticità) — usato per ispezionare l'attivazione dei KC senza il livello decisionale |
| `connectivity_seed` | `-1` | se `>=0` e `load_net["pn_kc"]=False`, genera una connettività PN→KC deterministica (riproducibile CPU/GPU) invece di quella casuale di GeNN |
| `tuneCurrent` | `False` | se `True`, esegue la calibrazione automatica di `INPUT_SCALE` (`tuneInputCurrent()`) subito dopo la build |
| `num_shifts` | da `params["NUM_SHIFTS"]` | override del numero di shift angolari scansionati da `testNavigation` |

Casi tipici:

```python
# Rete nuova, mai allenata, pronta per train()
nn = NeuralNetwork(params, load_net={"pn_kc": False, "kc_mbon": False})

# Rete già allenata, pronta per test()/testNavigation()
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True})

# Ispezione della sola risposta KC (nessun MBON/plasticità)
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": False}, reducedNetwork=True)
```

Quando hai finito con una rete, rilascia le risorse GPU/CPU:

```python
nn.model.unload()
```

## CPU o GPU

`use_gpu=False` (default) usa il backend `single_threaded_cpu` di GeNN —
deterministico, riproducibile bit per bit da una run all'altra, ma sequenziale
(una presentazione alla volta). `use_gpu=True` usa CUDA — molto più veloce se
combinato con `batch_size>1` e `precompute_features=True` (vedi sotto), ma
introduce un piccolo rumore floating-point rispetto alla CPU (differenze
nell'ordine di accumulo delle correnti sinaptiche tra thread paralleli):
sull'heading finale scelto (`best_degree`) l'accordo CPU/GPU è tipicamente
>99%, non il 100%.

```python
nn_cpu = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=False)
nn_gpu = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True}, use_gpu=True)
```

## `batch_size`: testare più presentazioni insieme

`batch_size` sfrutta il meccanismo nativo `model.batch_size` di GeNN per
eseguire fino a `batch_size` presentazioni (frame diversi, stesso shift
angolare) in un'unica simulazione, invece di una alla volta. Su questa rete
(piccola: 2500 KC) il guadagno è sostanziale solo a batch size grandi (es.
512) — a batch size piccoli (es. 64) l'overhead fisso per chiamata non viene
ammortizzato e può risultare persino più lento del non-batchato.

Vincoli:
- **Richiede `use_gpu=True`**: il backend CPU di GeNN rifiuta `batch_size>1`
  con un errore esplicito già al costruttore.
- **`train()` richiede `batch_size==1`**: la plasticità KC→MBON non è
  batchata (non avrebbe una semantica ben definita su lane indipendenti) —
  `train()` solleva `ValueError` se chiamato su una rete con `batch_size>1`.

```python
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True},
                    use_gpu=True, batch_size=512)

# test() accetta anche una LISTA di frame quando batch_size > 1:
counts = nn.test([frame_a, frame_b, frame_c], shift_degree=0.0)
# -> lista di conteggi spike MBON, uno per frame

# una singola frame funziona comunque (usa solo la prima lane):
count = nn.test(frame_a, shift_degree=0.0)  # -> int singolo
```

## `precompute_features`: eliminare il costo di preprocessing ripetuto

Il preprocessing di ogni immagine (crop/grayscale/shift/resize + estrazione
feature) è in pratica il vero collo di bottiglia quando si testa l'intero
dataset — non la simulazione GPU in sé. Con `precompute_features=True`, al
costruttore la rete calcola **una volta sola** le feature di tutti i frame del
dataset (`trainingDatasetPath`) per tutti gli shift angolari, e le tiene in
cache in memoria: `test()`/`testNavigation_batch()` le rileggono invece di
ricalcolarle, a patto di passare `frame_id`.

```python
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True},
                    use_gpu=True, batch_size=512, precompute_features=True)
# al costruttore: calcola le feature di TUTTO il dataset (costo una tantum,
# proporzionale al numero di frame -- per un dataset di qualche migliaio di
# frame può richiedere circa un minuto)

# poi test() può ricevere frame=None se frame_id è nella cache:
counts = nn.test([None, None], shift_degree=0.0, frame_id=[12, 13])
```

Se `frame_id` non è fornito, o non è in cache, `test()` ricade sul
preprocessing on-the-fly come sempre (richiede un `frame` vero in quel caso).

**Configurazione consigliata per testare un intero dataset il più
velocemente possibile**: `use_gpu=True, batch_size=512, precompute_features=True`.

## Allenare (`train`)

```python
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": False})  # pn_kc fisso, kc_mbon da allenare

for frame_id, frame in dataset:
    nn.train(frame, frame_id=frame_id)  # frame_id opzionale, solo per il tracking di novelty

nn.save_weights()  # scrive weightsPath/pn_kc_ind.npy e weightsPath/kc_mbon_g_0.npy
```

Oppure, per allenare su tutto il dataset in un colpo solo (via `NeuralModelBase.train_batch`, eredita da `NeuralModelBase`):

```python
nn.train_batch(frame_ids=None, plot_novelties=True)  # None = tutto trainingDatasetPath, a passo train_step
```

`train()` richiede sempre `batch_size==1` (vedi sopra) — non serve/non ha senso passare `use_gpu=True` per allenare, dato che si processa un frame alla volta comunque.

### Procedura di training estesa (`halve`)

Di default, quando un KC e l'MBON sparano in coincidenza, il peso della sinapsi KC→MBON viene azzerato (`g = 0`). Impostando `"halve": true` in `parameters.json`, `train()` usa una procedura estesa: oltre al frame stesso (azzerato, come sempre), allena anche le due presentazioni ottenute shiftando il frame di `±DEGREES_PER_SHIFT` (stesso meccanismo di shift di `testNavigation`/`_shift_degrees`), dove la coincidenza KC→MBON **dimezza** il peso (`g *= 0.5`) invece di azzerarlo — i due heading immediatamente vicini al frame di training vengono trattati come "meno familiari" invece che pienamente familiari.

```python
params["halve"] = True   # assente/False = procedura classica, invariata
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": False})
nn.train(frame, frame_id=frame_id)  # ora fa 3 presentazioni invece di 1
```

Nessuna modifica necessaria a `train_batch()`/`test_one_variant()`/`variants.py`: il parametro va letto una volta dal `parameters.json` della variante, `train()` si comporta di conseguenza per ogni frame.

## Testare un singolo frame (`test`, `testNavigation`)

```python
# spike MBON per un singolo shift angolare
count = nn.test(frame, shift_degree=9.0)

# scansione di tutti gli shift, sceglie l'heading a minima novelty
angle_rad = nn.testNavigation(frame, frame_number=42, log_path=params["plotsTestPath"])
```

## Testare un intero dataset (`testNavigation_batch`)

`testNavigation_batch` (ereditato da `NeuralModelBase`, in `base.py`) sceglie
automaticamente il percorso giusto in base a `batch_size`:

```python
# batch_size=1 (default): loop sequenziale, un frame alla volta -- comportamento invariato
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True})
error = nn.testNavigation_batch(frame_ids=None)  # None = tutto trainingDatasetPath, a passo train_step

# batch_size>1: percorso batchato automaticamente, molto più veloce sull'intero dataset
nn = NeuralNetwork(params, load_net={"pn_kc": True, "kc_mbon": True},
                    use_gpu=True, batch_size=512, precompute_features=True)
error = nn.testNavigation_batch(frame_ids=None, debug_mode=False)
```

`debug_mode` (default `True`) controlla se generare un grafico per-frame
(`plot_test_results`, matplotlib) — con migliaia di frame conviene metterlo a
`False` nel percorso batchato, dato che il rendering dei grafici può
dominare il tempo indipendentemente da quanto è veloce la rete. Il logging
CSV (`plotsTestPath/test_log.csv`) resta sempre attivo, in entrambi i casi.

## Cheatsheet

| Voglio... | Configurazione |
|---|---|
| Allenare una rete nuova | `NeuralNetwork(params, load_net={"pn_kc": False, "kc_mbon": False})`, poi `train()`/`train_batch()` |
| Testare un frame/pochi frame, in modo semplice | `NeuralNetwork(params, load_net={...: True})`, default (`use_gpu=False, batch_size=1`) |
| Testare l'intero dataset il più velocemente possibile | `use_gpu=True, batch_size=512, precompute_features=True`, poi `testNavigation_batch(debug_mode=False)` |
| Riprodurre bit-per-bit il comportamento di sempre | `use_gpu=False` (default), non passare `batch_size`/`precompute_features` |
| Ispezionare solo l'attivazione dei KC (senza MBON) | `reducedNetwork=True` |
