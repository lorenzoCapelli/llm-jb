# PROPOSALS — prima analisi vera su llm-jb

Letto: `PLAN.md`, `README.md`, `src/llm_jb/data/types.py`,
`src/llm_jb/analyses/{base,batch,residual_capture,logit_lens,
activation_patching,linear_probe,sae}.py`, `src/llm_jb/data/loaders/`,
`src/llm_jb/metrics/judge.py`, `src/llm_jb/hooks/capture.py`, `configs/`.

Vincoli confermati con te: **20-40h GPU/settimana** (moderate, non
dedicate), **nessun budget API**. Macchina: 4× A100 80GB, niente SLURM.

## Cosa ho verificato leggendo il codice (non fidarti del mio riassunto a
memoria — questo l'ho controllato riga per riga o con un import reale)

Queste cose cambiano concretamente cosa è fattibile in una settimana,
quindi le metto prima delle proposte invece che sepolte in fondo.

1. **Solo JBB ha tutte e tre le varianti.** `data/loaders/advbench.py` e
   `data/loaders/harmbench.py` popolano solo `harmful_prompt`
   (`benign_prompt=None`, `jailbroken_prompt=None` — guardali, sono
   letterali nel codice). Qualunque confronto harmful/benign/jailbroken
   **deve** usare `load_jbb()` (`configs/dataset/jbb.yaml`). AdvBench e
   HarmBench servono solo se la domanda riguarda comportamento harmful in
   generale, non il confronto fra varianti.

2. **`jailbroken_prompt` di JBB: lo span dell'istruzione è quasi sempre
   una stima, non un match esatto — e dipende dal metodo d'attacco.**
   L'ho verificato eseguendo `load_jbb()` sulla cache reale già scaricata
   in questa repo: su 100 comportamenti, **PAIR ha 82 jailbreak con
   prompt, di cui solo 1 con match esatto dell'istruzione originale**
   (`instruction_span_matched=True` in `metadata`) — PAIR parafrasa il
   goal, quindi per il 98% dei casi lo span ricade nel fallback
   "tutta la stringa è l'istruzione" (`data/loaders/jbb.py::_make_jailbroken_span`).
   **GCG ha invece 100/100 jailbreak con prompt, 99/100 con match esatto**
   (GCG appende un suffisso al goal verbatim). Conseguenza diretta:
   `AnchorMode.MEAN_INSTRUCTION_SPAN` su esempi PAIR è quasi sempre
   rumore (media su tutto il wrapper, non sull'istruzione); su GCG è
   pulito. Qualunque proposta che usa lo span dell'istruzione deve
   tenerne conto esplicitamente (vedi rischi sotto).

3. **`gpt2-small` non ha training di sicurezza.** È un base LM (nessun
   RLHF/instruction-tuning), quindi non "rifiuta" nulla in modo
   significativo — è utile per validare che il codice giri, ma un
   risultato su gpt2-small su temi di refusal/jailbreak non dice quasi
   nulla sul fenomeno reale. Le tre proposte sotto sono pensate per
   girare la parte "vera" su **`llama-3.2-1b-instruct`**
   (`configs/model/llama-3.2-1b-instruct.yaml`, config già pronta ma mai
   scaricata — gated, serve `HF_TOKEN` con licenza accettata: **primo
   blocco pratico da risolvere il giorno 0, indipendentemente dalla
   proposta scelta**). Confermato in questa sessione che TransformerLens
   supporta `meta-llama/Llama-3.2-1B-Instruct` nativamente
   (`OFFICIAL_MODEL_NAMES`). Secondo la model card pubblica: 16 layer,
   d_model 2048 — da confermare con `model.cfg` al primo caricamento
   reale, non l'ho ancora fatto io.

4. **Gli artifact JBB (PAIR/GCG) sono stati ottimizzati contro
   `vicuna-13b-v1.5`, non contro i nostri modelli.** Usarli su
   `llama-3.2-1b-instruct` significa testare **transfer** di un jailbreak
   creato per un altro modello — non è garantito che "funzioni" affatto
   su llama. `metadata["jailbroken_success"]` in `BehaviorTriple` dice
   solo se l'attacco ha funzionato su vicuna, non sul modello che
   analizzeremo. Questo è un rischio condiviso da tutte e tre le
   proposte, più acuto per la Proposta B.

5. **Non esiste generazione da nessuna parte nella repo.**
   `hooks/capture.py::capture_residual_stream` fa un solo forward pass
   (`model.run_with_hooks(tokens, ..., return_type=None)`), mai
   `.generate()`. `metrics/judge.py::Judge.judge(response: str)` si
   aspetta testo generato, ma niente nella repo lo produce oggi.
   TransformerLens espone `HookedTransformer.generate()` di serie (l'ho
   verificato via `inspect.signature`, supporta sampling/greedy/KV
   cache), quindi non serve una nuova dipendenza — ma è comunque un
   pezzo da scrivere, non uno stub già pronto. Le proposte A e C non ne
   hanno bisogno affatto; la B lo usa solo per un pilot piccolo e a
   parte (vedi sotto), non per l'esperimento principale.

6. **`Analysis.run(self, model, batch)` non ha un canale per etichette
   esterne.** Per un probe harmful-vs-benign va bene così — l'etichetta
   è già `batch.variants` (`analyses/batch.py`), niente da cambiare
   nell'interfaccia. Per un probe *refused-vs-complied* (che userebbe
   `metrics/judge.py`) servirebbe invece estendere la firma per
   accettare label esterne: **oggi non è possibile senza toccare
   `analyses/base.py`**. Lo segnalo perché nessuna delle tre proposte qui
   sotto ne ha bisogno, ma è il motivo per cui non ne propongo una
   basata sul judge.

---

## Proposta A — Geometria rappresentazionale: il jailbreak "traveste" il contenuto o lo lascia intatto?

### 1. Domanda e criterio di falsificazione

Nel residual stream, la distinzione harmful-vs-benign è linearmente
decodificabile per gran parte dei layer (atteso, ben documentato in
letteratura). La domanda vera è: **quando un jailbreak (GCG) funziona,
l'attivazione del prompt jailbroken si sposta verso il lato "benign"
della frontiera di decisione, o resta dal lato "harmful" nonostante il
comportamento del modello cambi?**

- **Ipotesi "travestimento"**: l'attivazione jailbroken proietta più
  vicino al centroide benign rispetto a quella harmful non travestita —
  il jailbreak funziona rendendo il contenuto *rappresentazionalmente*
  più simile a qualcosa di innocuo.
- **Falsificazione**: se la proiezione delle attivazioni jailbroken sul
  discriminante harmful/benign resta indistinguibile (stesso lato,
  stessa distanza media, test statistico non significativo) da quella
  harmful non travestita, ad ogni layer, l'ipotesi è falsificata — il
  jailbreak agisce altrove (causalmente più a valle, non sulla
  rappresentazione del contenuto stesso: motivo per cui la Proposta B
  esiste come alternativa, non come passo successivo automatico).

### 2. Esperimento minimo

- **Modelli**: `llama-3.2-1b-instruct` (risultato vero); `gpt2-small`
  solo come pilot di codice, non di scienza (vedi punto 3 sopra).
- **Dataset**: `configs/dataset/jbb.yaml`, `variants: [harmful, benign,
  jailbroken]`, **solo GCG** come metodo d'attacco iniziale (span
  affidabile al 99%); PAIR come secondo giro, sapendo che userà quasi
  sempre lo span "tutta la stringa" per fallback.
- **AnchorMode**: `MEAN_INSTRUCTION_SPAN` come primario (isola il
  contenuto dal wrapper); `LAST_PROMPT_POSITION` come confronto —
  separabilità sul contenuto vs. separabilità sulla posizione da cui
  parte la generazione sono domande leggermente diverse, vale la pena
  vederle entrambe.
- **Analisi**: `analyses.residual_capture.ResidualCaptureAnalysis`
  (esiste già, invariata) per estrarre le attivazioni; il probe vero e
  proprio va scritto (vedi sotto).
- **Metriche**: accuracy/AUC del probe per layer (train/test split
  sugli ~100 comportamenti JBB); posizione della proiezione jailbroken
  rispetto alla frontiera, con test statistico (Mann-Whitney, campioni
  appaiati per `behavior_id`).

### 3. Cosa implementare

- **Riuso diretto**: `data.loaders.jbb.load_jbb()`,
  `analyses.batch.build_batch()` (variants `["harmful","benign"]` per il
  training, `["jailbroken"]` per la proiezione),
  `analyses.residual_capture.ResidualCaptureAnalysis` +
  `ResidualCaptureConfig`, `data.types.AnchorMode`,
  `models.backend.load_model` + `configs/model/llama-3.2-1b-instruct.yaml`.
- **Da scrivere**: il corpo vero di
  `analyses.linear_probe.LinearProbeAnalysis.run()` (oggi
  `NotImplementedError`) — regressione logistica per layer, etichette da
  `batch.variants` (nessuna modifica a `Analysis.run`, vedi punto 6
  sopra). Serve anche un piccolo script/notebook per la proiezione degli
  esempi jailbroken sul probe già addestrato (non è "un altro giro di
  `.run()`", è un passo di analisi separato — direi in `notebooks/`,
  coerente con la convenzione della repo "solo esplorazione").
- **Dipendenze mancanti**: nessuna dipendenza pesante. Scelta
  d'implementazione da una riga: `scikit-learn` (pulito, standard, CPU,
  costo zero) vs. regressione logistica scritta a mano in `torch` (zero
  dipendenze nuove, più codice). Consiglio `scikit-learn`.

### 4. Costo e primo risultato

~100 comportamenti × 2-3 varianti × 1B parametri su 1 A100: estrazione
attivazioni in **secondi-minuti**, non ore (per confronto, 4 esempi su
gpt2-small in questa sessione: <1s). Training dei probe (regressione
logistica, ~100 esempi, per layer): trascurabile, CPU. Stima:
**< 2h GPU totali**, **2-3 giorni-persona** (implementare il probe reale,
girare, un primo grafico "accuracy del probe per layer + proiezione
jailbroken"). Primo risultato osservabile: quel grafico, entro 3 giorni,
ben dentro la settimana.

### 5. Rischio principale e come lo scopro presto

**Rischio**: lo span PAIR inaffidabile (punto 2 sopra) contamina
silenziosamente i risultati se non lo si esclude esplicitamente.
**Scoperta precoce**: il giorno 1, prima di qualunque probe, controllo
`metadata["instruction_span_matched"]` sulla popolazione usata — se lavoro
solo su GCG questo rischio è già chiuso per costruzione. Rischio
secondario, più serio: `llama-3.2-1b-instruct` è gated — se `HF_TOKEN`
non è ancora configurato con licenza accettata, lo scopro nei primi 5
minuti (`load_model` fallisce subito con un errore HF esplicito), non a
metà settimana.

---

## Proposta B — Localizzazione causale: dove "vive" l'effetto del jailbreak?

### 1. Domanda e criterio di falsificazione

Con activation patching layer-per-layer: **se sostituisco, a un singolo
layer e alla posizione dell'ultimo token del prompt, l'attivazione del
run "harmful pulito" con quella del run "jailbroken" (o viceversa), il
modello passa da una tendenza al rifiuto a una tendenza alla compliance
(misurata come differenza di logit fra un token indicativo di rifiuto e
uno indicativo di compliance)? A quale layer l'effetto è massimo?**

- **Falsificazione**: se nessun singolo layer produce uno spostamento di
  logit-diff statisticamente distinguibile dal rumore (patching a
  qualunque layer ≈ nessun patching), l'effetto non è localizzato nel
  residual stream a quella posizione — risultato negativo reale, non un
  fallimento dell'esperimento (magari l'effetto è distribuito, o vive
  nei pattern di attenzione: qui la Proposta C diventa il passo
  naturale, non ridondante).

### 2. Esperimento minimo

- **Modelli**: solo `llama-3.2-1b-instruct` — su gpt2-small non c'è
  refusal da spostare, l'esperimento sarebbe privo di senso, buono solo
  per testare che il codice non esploda.
- **Dataset**: `configs/dataset/jbb.yaml`, coppie harmful/jailbroken
  filtrate su `metadata["jailbroken_success"] is True` (attenzione: vero
  "su vicuna", non su llama — vedi punto 4 sopra, è il rischio
  principale di questa proposta, non un dettaglio).
- **AnchorMode**: `LAST_PROMPT_POSITION` (il patching causale classico
  agisce alla posizione decisionale, non mediata sull'istruzione).
- **Analisi**: `analyses.activation_patching.ActivationPatchingAnalysis`
  — oggi stub puro (`NotImplementedError`, nessuna logica), va scritta
  da zero.
- **Metriche**: differenza di logit fra un token "rifiuto" e uno
  "compliance" al passo successivo al prompt (es. " I" vs " Sure"/" Here"
  — **da confermare empiricamente su llama-3.2-1b**, non assumerlo dai
  pattern GPT-generici). Nessuna generazione di testo necessaria per
  l'esperimento principale — è tutto leggibile dai logit di un singolo
  forward pass.

### 3. Cosa implementare

- **Riuso diretto**: `data.loaders.jbb.load_jbb()`,
  `analyses.batch.build_batch()`, `models.backend.load_model`,
  `hooks.capture.resid_post_hook_name()` (stesso naming helper),
  `data.alignment.anchor_range()` (per l'indice di
  `LAST_PROMPT_POSITION` per riga), `HookedTransformer.run_with_hooks`
  (stesso meccanismo di `capture_residual_stream`, ma con un hook che
  **scrive** invece di solo leggere).
- **Da scrivere**: il corpo vero di `ActivationPatchingAnalysis.run()`
  più, verosimilmente, un nuovo modulo tipo `hooks/patch.py` (hook che
  sovrascrive `hook_resid_post` con un'attivazione cached da un altro
  run, speculare a `capture_residual_stream` come struttura). Più un
  pilot **piccolo e separato** (in `notebooks/`, non nella pipeline
  principale) per: (a) verificare su ~20 esempi reali se PAIR/GCG
  transferiscono affatto su llama-3.2-1b (usando
  `HookedTransformer.generate()` + `metrics.judge.SubstringRefusalJudge`
  — questo è l'unico punto di tutta la proposta che tocca
  generazione/judge, ed è un controllo, non l'esperimento); (b)
  scegliere empiricamente la coppia di token proxy per il logit-diff.
- **Dipendenze mancanti**: nessuna nuova dipendenza pip. Il costo è
  interamente codice nuovo (patch-in hook + pilot), non librerie.

### 4. Costo e primo risultato

Sweep di patching: ~16 layer × ~50-80 coppie harmful/jailbroken-riuscito
(da verificare quante coppie sopravvivono al filtro success) × 2
direzioni ≈ 1600-2600 forward pass extra, ciascuno rapido (modello da 1B
su A100, frazioni di secondo) → **sotto l'ora di GPU** anche stimando
largo. Persona: **4-5 giorni** — il pilot di transferability (mezza
giornata) e la scelta dei token proxy (mezza giornata) vanno fatti
*prima* di scrivere `hooks/patch.py`, non dopo. Primo risultato
osservabile: la curva "effetto causale per layer", entro la settimana ma
con meno margine delle altre due proposte.

### 5. Rischio principale e come lo scopro presto

**Rischio principale, e il più serio delle tre proposte**: PAIR/GCG
potrebbero non funzionare affatto su llama-3.2-1b-instruct (ottimizzati
per vicuna-13b-v1.5, modello diverso, tokenizer diverso, alignment
diverso). Se non transferiscono, l'intero esperimento di patching parte
da premesse false. **Come lo scopro presto**: il pilot (a) sopra — 20
generazioni reali + judge a substring — è il primissimo passo, va fatto
*prima* di scrivere una riga di `hooks/patch.py`. Costa meno di un'ora
di GPU e mezza giornata-persona. Se il tasso di successo reale su llama
è vicino a zero, questa proposta va scartata o ridotta a un dataset più
piccolo di jailbreak che effettivamente transferiscono, non proseguita
alla cieca.

---

## Proposta C — Routing dell'attenzione: il jailbreak "diluisce" l'attenzione sul contenuto dannoso?

### 1. Domanda e criterio di falsificazione

**Quando il modello sta per generare (posizione dell'ultimo token del
prompt), quanta massa di attenzione destina ai token dell'istruzione
originale — e questa massa cala quando l'istruzione è avvolta in un
jailbreak riuscito, rispetto a quando è presentata da sola?**

- **Ipotesi "diluizione"**: il wrapper del jailbreak non cambia cosa
  "significa" l'istruzione per il modello, ma ne riduce il peso
  attenzionale relativo semplicemente aggiungendo molto altro testo
  intorno — il contenuto dannoso diventa meno "salente" nel calcolo
  della risposta, non meno riconosciuto.
- **Falsificazione**: se la massa di attenzione (normalizzata, non
  grezza — vedi rischio sotto) sui token dell'istruzione è
  statisticamente indistinguibile fra harmful e jailbroken a parità di
  lunghezza dell'istruzione, l'ipotesi cade — il meccanismo non è
  diluizione attenzionale (coerente con un effetto rappresentazionale,
  Proposta A, o causale-non-attenzionale, Proposta B).

### 2. Esperimento minimo

- **Modelli**: `gpt2-small` come pilot **metodologicamente informativo**
  (il routing attenzionale verso uno span marcato è una proprietà
  architetturale generale, non richiede training di sicurezza — più
  utile qui di quanto lo sia per A/B, ma l'interpretazione "dannoso" resta
  valida solo su `llama-3.2-1b-instruct`).
- **Dataset**: `configs/dataset/jbb.yaml`, harmful + jailbroken (GCG
  primario per lo stesso motivo della Proposta A — span affidabile).
- **AnchorMode**: usa `data.alignment.anchor_range()` con
  `AnchorMode.MEAN_INSTRUCTION_SPAN` per ottenere il range
  `[instruction_token_start, instruction_token_end)` — qui non per
  mediare attivazioni, ma come range su cui sommare i pesi di
  attenzione in arrivo dalla query in `LAST_PROMPT_POSITION`.
- **Analisi**: non esiste — nessuno dei quattro stub attuali copre
  pattern di attenzione. Va scritta da zero, sia il capture sia la
  classe `Analysis`.
- **Metriche**: massa di attenzione sull'istruzione, **normalizzata**
  (frazione della massa totale non-sink, non valore grezzo — vedi
  rischio); confronto appaiato per `behavior_id`, harmful vs jailbroken.

### 3. Cosa implementare

- **Riuso diretto**: `data.loaders.jbb.load_jbb()`,
  `analyses.batch.build_batch()`, `data.alignment.anchor_range()` (solo
  per calcolare gli indici, non per l'estrazione — quella è nuova),
  `models.backend.load_model`.
- **Da scrivere** (tutto nuovo, non uno stub esistente):
  - un modulo tipo `hooks/capture_attention.py`, stessa filosofia
    selettiva di `hooks/capture.py` (mai salvare il pattern intero
    `(batch, n_heads, seq, seq)` — enorme e inutile — riduci **dentro
    l'hook** a `(batch, n_heads)` sommando solo sulla colonna
    dell'istruzione), agganciato a `blocks.{layer}.attn.hook_pattern`
    (nome hook standard di TransformerLens, diverso da
    `hook_resid_post`);
  - `analyses/attention_routing.py::AttentionRoutingAnalysis`, nuova
    classe che implementa `Analysis`.
- **Dipendenze mancanti**: nessuna nuova dipendenza pip. Tutto il costo
  è codice nuovo, e più di quanto serva per A (che riusa
  `residual_capture` così com'è) — paragonabile a B in quantità di
  codice nuovo, ma senza il rischio di transferability così acuto.

### 4. Costo e primo risultato

Costo GPU paragonabile alle altre due (**< 1h**, un forward pass a
comportamento, nessuna generazione). Persona: **4-5 giorni** — il
capture selettivo sui pattern di attenzione è infrastruttura nuova, va
testato con la stessa cura di `hooks/capture.py` (shape/dtype espliciti,
niente cache totale) prima di fidarsi dei numeri. Primo risultato
osservabile: distribuzione appaiata "massa di attenzione sull'istruzione,
harmful vs jailbroken" su GCG, entro la settimana, con margine più
stretto di A.

### 5. Rischio principale e come lo scopro presto

**Rischio**: "attention sink" — è documentato che molti transformer
destinano una frazione sproporzionata di attenzione al primissimo token
indipendentemente dal contenuto, il che può far sembrare "diluita" ogni
altra regione semplicemente per posizione, non per il jailbreak.
**Come lo scopro presto**: prima di costruire tutta la pipeline, un
controllo di 30 minuti su gpt2-small — plot grezzo della massa di
attenzione su *tutte* le posizioni (non solo l'istruzione) per una
manciata di esempi, per vedere se il sink è già dominante lì. Se sì, la
metrica va normalizzata escludendo il primo token o usando una massa
relativa, non assoluta — decisione da prendere il giorno 1, non dopo
aver girato tutto l'esperimento.

---

## Confronto rapido

| | A — Geometria | B — Causale (patching) | C — Attenzione |
|---|---|---|---|
| Rischio implementativo | Basso (riusa `residual_capture` invariata) | Alto (hook di scrittura nuovo, nessun precedente in repo) | Medio-alto (infrastruttura di capture nuova, nessuno stub esistente) |
| Rischio scientifico | Medio (probabile *qualche* separabilità: la domanda vera è dove/quanto) | Alto (rischio concreto di risultato nullo o non-transfer) | Medio (rischio di confondere sink attenzionale con effetto reale) |
| Dipendenze mancanti | Nessuna (o `scikit-learn`, opzionale) | Nessuna | Nessuna |
| Girabile su gpt2-small oggi | Sì (solo come pilot di codice) | Solo come test "non esplode" | Sì (pilot metodologicamente utile, non solo di codice) |
| Primo risultato | ~3 giorni | ~4-5 giorni, meno margine | ~4-5 giorni, meno margine |
| GPU stimata | < 2h | < 1h | < 1h |

## Raccomandazione

**Parti da A.** È l'unica delle tre che riusa `residual_capture` così
com'è (zero rischio infrastrutturale nuovo), ha il criterio di
falsificazione più pulito da misurare, e con 20-40h GPU/settimana il
collo di bottiglia reale sei tu, non la macchina — A minimizza il tempo
fra "inizio" e "primo numero vero" (~3 giorni vs ~4-5). In parallelo, nei
ritagli della prima settimana, fai **solo il pilot di transferability
della Proposta B** (punto 5 di B: 20 generazioni + judge a substring,
mezza giornata) — è economico, e se il risultato è che PAIR/GCG *non*
transferiscono affatto su llama-3.2-1b, è un fatto che ti serve comunque
per decidere se B ha senso come seconda proposta o va scartata subito. C
la lascerei per una terza iterazione: è la più costosa in infrastruttura
nuova e la sua domanda è più complementare che urgente rispetto ad A.

Se invece l'obiettivo di questa fase è avere presto qualcosa di
riconoscibile come "mechanistic interpretability causale" da mostrare
(più vicino a cosa un relatore si aspetterebbe come metodo), B vale la
pena anche con il rischio più alto — ma allora il pilot di
transferability non è più opzionale-in-parallelo, è letteralmente lo
step 0.

## Domande aperte

- `HF_TOKEN` per `llama-3.2-1b-instruct` è già configurato su questa
  macchina con la licenza accettata, o è un passo ancora da fare? Blocca
  tutte e tre le proposte allo stesso modo.
- Per la Proposta A: preferisci `scikit-learn` (nuova dipendenza minima,
  codice più pulito) o regressione logistica scritta a mano in `torch`
  (zero dipendenze nuove)?
- Sull'uso di PAIR: va bene scartarlo per ora (solo GCG, span
  affidabile) e riprenderlo solo se serve un secondo metodo d'attacco
  più avanti, o è importante per te confrontare i due metodi fin da
  subito nonostante il problema di span-matching?
- Il relatore/gruppo ha una preferenza metodologica dichiarata
  (rappresentazionale vs causale vs strutturale) che dovrebbe pesare più
  del profilo di rischio puro nella scelta?
