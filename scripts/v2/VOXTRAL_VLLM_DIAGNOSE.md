# Voxtral LoRA → vLLM: Vollständige Diagnose (2026-07-20)

## Ziel
Voxtral-Mini-3B-2507 mit LoRA auf medizinischen Transkriptionen feintunen → nativ in vLLM deployen (kein LoRA-Runtime, kein transformers, Mistral-Format `consolidated.safetensors`).

---

## Architektur

### Modell: Voxtral-Mini-3B-2507
- **30 LM-Layer** (Llama-style): n_heads=32, n_kv_heads=8, hidden=3072, intermediate=8192
- **32 Audio-Whisper-Encoder-Layer**: hidden=1280, heads=20, intermediate=5120
- **Projector**: audio_language_projection (2-layer MLP, gelu)
- **Format**: Mistral-nativ — eine `consolidated.safetensors`, 761 Tensoren, 8.71 GB
  - LM: `layers.N.attention.{wq,wk,wv,wo}.weight`, `layers.N.feed_forward.{w1,w2,w3}.weight`
  - Audio: `mm_whisper_embeddings.whisper_encoder.transformer.layers.N.attention.{wq,wk,wv,wo}.{weight,bias}`
  - Audio: `mm_whisper_embeddings.whisper_encoder.transformer.layers.N.feed_forward.{w1,w2}.{weight,bias}`
  - Norms: `layers.N.{attention_norm,ffn_norm}.weight`

### LoRA-Adapter
- **Training**: V100 (CC 7.0), FP16 AMP, GradScaler, SDPA-Attention
- **LoRA-Config**: r=32, alpha=64, dropout=0.05, scaling = alpha/r = 2.0
- **Targets**: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **612 Tensoren** (306 A,B-Paare): 30 LM × 7 Module + 32 Audio × 3 Module (q/k/v, kein out_proj)
- **PEFT-Key-Format**: `base_model.model.model.language_model.layers.N.self_attn.{mod}.lora_{A,B}.weight`

### vLLM-Container
- vLLM 0.17.1+a03ca76a (NVIDIA 26.03), transformers 4.57.5
- `voxtral-vllm-dgx:latest`, DGX Spark (192.168.188.185)
- Modell-Logik: `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/voxtral.py`
- Whisper-Encoder: `/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/whisper.py`

---

## Versuchte Lösungswege (chronologisch)

### ❌ Weg 1: LoRA-Runtime via vLLM
- LoRA-Adapter direkt in vLLM laden (`--lora-modules`)
- **Fehler**: vLLM erwartet HuggingFace-LoRA-Key-Namen, aber das Modell ist Mistral-Format.
- **Key-Mismatch**: vLLM sucht `language_model.model.layers.*` aber Modell hat `layers.N.*`.

### ❌ Weg 2: merge_and_unload() → HF-Format
- `model.merge_and_unload()` → `model.save_pretrained()` → `model.safetensors`
- **Fehler**: vLLM's `voxtral.py` load_weights erwartet `consolidated.safetensors` im Mistral-Format.
- 0/761 Keys überlappen zwischen HF-Format und Original-Mistral-Format.
- Zusätzlich brauchte es `voxtral_patched.py` mit falschen Remapping-Regeln für HF-Keys.

### ✅ Weg 3: Native Mistral Merge (merge_lora_native_mistral.py)
- Lädt LoRA-A/B-Paare, mapped auf Mistral-safetensors-Keys, wendet Delta an: `W += scaling * (B @ A)`
- **Ergebnis**: 761 Keys, 306 mit Nicht-Null-Delta, MD5 ≠ Original
- **LoRA-Mapping korrekt**: Alle 306 Ziel-Matrizen haben Delta (max ~0.007, mean ~0.0002)
- **Verifiziert**: Shape-Matching 100%, keine Missing- oder Extra-Keys
- **vLLM lädt ohne Crash** — nach Entfernen von `voxtral_patched.py` Mount

### ❌ Problem: Output identisch mit Base-Modell
- Trotz korrektem Merge und erfolgreichem vLLM-Start: **Transkriptionen sind exakt dieselben wie Base**
- Der LoRA-Effekt geht im vLLM-Inferenz-Pfad **komplett verloren**
- Transformers-Test (direkt) zeigt: LoRA-Modell transkribiert ALLE trainierten Paare korrekt

---

## Gewichts-Lade-Kette in vLLM (Container-Original)

### Schritt 1: `VoxtralForConditionalGeneration.load_weights()`
```python
remapping_rules = [
    (r"mm_whisper_embeddings\.(.*)", r"\1"),  # Strip mm_whisper_embeddings. prefix
    ...
]
```
→ `mm_whisper_embeddings.whisper_encoder.transformer.layers.0.attention.wq.weight`
→ `whisper_encoder.transformer.layers.0.attention.wq.weight`

### Schritt 2: `VoxtralEncoderModel.mistral_remapping` (lines 746-795)
```python
mistral_remapping = [
    # attention
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.w([qkv])\.(weight|bias)",
     r"whisper_encoder.layers.\1.self_attn.\2_proj.\3"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo\.(weight|bias)",
     r"whisper_encoder.layers.\1.self_attn.out_proj.\2"),
    # feed_forward
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1\.(weight|bias)",
     r"whisper_encoder.layers.\1.mlp.fc1.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2\.(weight|bias)",
     r"whisper_encoder.layers.\1.mlp.fc2.\2"),
    # norms
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm\.(weight|bias)",
     r"whisper_encoder.layers.\1.self_attn_layer_norm.\2"),
    (r"whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm\.(weight|bias)",
     r"whisper_encoder.layers.\1.final_layer_norm.\2"),
]
```
→ `whisper_encoder.transformer.layers.0.attention.wq.weight`
→ `whisper_encoder.layers.0.self_attn.q_proj.weight`

### Schritt 3: `VoxtralEncoderModel.load_weight()` (lines 916-963)
```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
]
```
Ersetzt `q_proj` → `qkv_proj` und nutzt `param.weight_loader(param, loaded_weight, shard_id)`.

**Achtung**: vLLM's WhisperEncoder **fusioniert QKV** in eine einzige Linear-Schicht (`qkv_proj`). Die separaten `wq`, `wk`, `wv` werden über den weight_loader mit shard_id in die richtigen Slices der fusionierten Matrix geschrieben.

### Schritt 4: `WhisperEncoder` (aus whisper.py)
- `WhisperAttention` hat: `self.qkv_proj` (ColumnParallelLinear)
- `WhisperMLP` hat: `self.fc1` (ColumnParallelLinear), `self.fc2` (RowParallelLinear)
- `WhisperEncoderLayer` hat: `self.self_attn`, `self.self_attn_layer_norm`, `self.mlp`, `self.final_layer_norm`

---

## Mögliche Ursachen für den LoRA-Effekt-Verlust

### Hypothese 1: Audio-Tower-Mapping auf falsche vLLM-Module
- LoRA targetiert `audio_tower.layers.N.self_attn.{q,k,v}_proj` (PEFT-Namen)
- Unser Merge schreibt auf `mm_whisper_embeddings.whisper_encoder.transformer.layers.N.attention.{wq,wk,wv}.weight`
- vLLM's `VoxtralForConditionalGeneration.load_weights()` strippt `mm_whisper_embeddings.` prefix
- Dann greift `VoxtralEncoderModel.mistral_remapping`
- Dann `load_weight()` mapped `q_proj` → `qkv_proj` (fused)
- ⚠️ **Audio feed_forward (w1/w2) wird NICHT von LoRA targetiert!** LoRA targetiert nur attention-Module im Audio-Encoder. Das ist korrekt so (nur q/k/v trainiert), ABER:

### Hypothese 2: Fused QKV loading zerstört Deltas
- Der `weight_loader` für `qkv_proj` nutzt `shard_id` um das geladene Weight in den richtigen Slice der fusionierten Matrix zu schreiben
- **Frage**: Korrumpiert der weight_loader die Reihenfolge oder überschreibt er Q/K/V-Slices falsch?
- **Test**: Direkter Forward-Pass-Vergleich auf Audio-Encoder-Ebene nötig

### Hypothese 3: vLLM verwendet andere numerische Pfade
- vLLM castet BF16 → FP16 beim Laden (`Casting torch.bfloat16 to torch.float16`)
- vLLM nutzt `enforce_eager` (kein CUDA Graph, kein torch.compile)
- Die LoRA-Deltas sind sehr klein (max 0.007 auf Gewichte im Bereich ±0.1 = ~7%)
- FP16 hat nur ~3.3 Dezimalstellen Präzision → kleine Deltas könnten durch FP16-Rundung verloren gehen
- ⚠️ **Das Training lief auf V100 in FP16!** Die LoRA-Matrizen selbst sind FP32, aber das Mergen castet sie auf BF16 (Original-Datentyp)

### Hypothese 4: Audio-Preprocessing unterscheidet sich
- vLLM nutzt `mistral_common.audio.mel_filter_bank` + `torch.stft` (in `VoxtralEncoderModel.compute_whisper_melspec`)
- Transformers nutzt `WhisperFeatureExtractor` (HF)
- Unterschiede in Mel-Spektrogramm-Berechnung könnten den Input so verändern, dass der LoRA-Effekt nicht greift

### Hypothese 5: vLLM lädt doch das falsche File
- Ausgeschlossen: MD5 von `/home/owc/voxtral-consolidated/consolidated.safetensors` ≠ Original-MD5
- vLLM mountet `/home/owc/voxtral-consolidated:/models/merged:ro` → lädt korrekt unser merged File

---

## Was funktioniert (Transformers)
- `VoxtralForConditionalGeneration.from_pretrained(merged_dir)` lädt das gemergte Modell korrekt
- `model.generate()` mit LoRA-Adapter (vor merge) oder mit gemergtem Modell produziert korrekte, vom Base-Modell verschiedene Transkriptionen
- Alle trainierten Test-Paare werden korrekt transkribiert

## Was nicht funktioniert (vLLM)
- vLLM lädt das gemergte `consolidated.safetensors` ohne Fehler
- Health-Check 200 OK, Chat-Completion funktioniert, Audio-Transkription funktioniert
- **Aber**: Output ist bit-identisch oder nahezu identisch zum Base-Modell

---

## Nächste Schritte / Offene Fragen

1. **Forward-Pass-Vergleich**: Gleicher Audio-Input → WhisperEncoder output embeddings → vergleichen Base vs Merged in vLLM
2. **FP16-Präzisionstest**: Mergen in FP32 statt BF16, dann Vergleich
3. **vLLM WhisperEncoder isoliert testen**: `WhisperEncoder.forward()` mit separaten q/k/v weights vs fused qkv
4. **Recherche**: Gibt es bekannte Voxtral+LoRA+vLLM Deployment-Erfolge im Netz?

---

## Dateien
| Datei | Zweck | Status |
|-------|-------|--------|
| `scripts/v2/merge_lora_native_mistral.py` | LoRA→Mistral-Format Merge | ✅ Funktioniert (306 Matrizen) |
| `training/train_voxtral_lora.py` | LoRA-Training auf V100 | ✅ Trainiert (r=32, alpha=64) |
| `training/models/voxtral-mini-finetuned/lora-final/` | LoRA-Adapter (237 MB) | ✅ |
| `/home/owc/voxtral-consolidated/` | Gemergtes Modell (8.71 GB) | ✅ vLLM lädt es |
| `/home/owc/voxtral-vllm/run_vllm.sh` | vLLM Launch-Script | ✅ (ohne patched.py) |
| `/home/owc/voxtral-setup/voxtral_patched.py` | Alter HF-Format Patch | ❌ DEPRECATED |
| `scripts/v2/deep_delta_analysis.py` | Delta-Analyse | ✅ 306 nonzero |
| `scripts/v2/ab_compare.py` | A/B-Vergleich Base vs Merged | ⏸️ Noch nicht ausgeführt |
| `scripts/v2/compare_forward.py` | Forward-Pass-Vergleich | ⏸️ Noch nicht ausgeführt |

---

## 🔍 Recherche-Ergebnisse (2026-07-20)

### GitHub vLLM Issues & PRs

#### PR #45697 — "Enable LoRA support for tower and connector in Voxtral" 
- **Autor**: anshulkulhari7, eröffnet 15. Juni 2026
- **Status**: Offen, Review pending (patrickvonplaten, DarkLight1337, ywang96, AndreasKaratzas)
- **Inhalt**: Fügt `get_num_mm_encoder_tokens` / `get_num_mm_connector_tokens` für Voxtral hinzu
- **Wichtigster Satz**: *"Tower (`whisper_encoder`) is vLLM's native `WhisperEncoder` (QKV/Column/Row-parallel, including the explicit `MergedColumnParallelLinear` K/V split the code comments call out as enabling LoRA), so tower LoRA attaches"*
- **ABER**: *"Connector (`audio_language_adapter`) currently uses plain `nn.Linear` (`w_in`/`w_out`). `from_layer` only wraps vLLM-native linear types, so connector LoRA is inert today"*
- **Fazit für uns**: vLLM's WhisperEncoder **kann LoRA** (weil native Linears), aber AudioLanguageAdapter **kann kein LoRA** (nn.Linear). Das betrifft aber nur **Runtime-LoRA**, nicht unseren Merge-Ansatz.

#### RFC #45771 — "Enable connector LoRA for audio multimodal models"
- Gleicher Autor, 16. Juni 2026
- Bestätigt: **ALLE Audio-MM-Modelle** (Voxtral, Ultravox, Qwen2-Audio, AudioFlamingo3) haben dasselbe Problem: Connector-Projectoren sind `nn.Linear` → können nicht von vLLM's `from_layer` mit LoRA gewrappt werden
- Lösung: `nn.Linear` → `ReplicatedLinear` konvertieren in der Modelldefinition
- **NICHT UNSER PROBLEM** — wir mergen, nicht runtime LoRA

#### Issue #31479 — "Enable LoRA support for tower and connector in more MM models"
- jeejeelee, Dezember 2025
- Tracking-Issue für alle MM-Modelle
- Voxtral ist **nicht** auf der "done" Liste (nur Pixtral von Mistral-Familie ist merged)

### HuggingFace Discussions
- Discussion #5 ("Colab notebook request"): User berichten von Problemen mit `vllm serve` für Voxtral
- Keine bekannten erfolgreichen Voxtral-LoRA-vLLM-Deployments dokumentiert

### Key Takeaway aus der Recherche
**Niemand hat bisher erfolgreich ein LoRA-fingetuntes Voxtral auf vLLM deployed.** 
- PR #45697 ist der ERSTE Versuch, überhaupt Tower/Connector-LoRA-Support für Voxtral zu implementieren
- Der PR ist noch nicht gemerged (seit 5 Wochen offen)
- Selbst wenn gemerged, betrifft er nur **Runtime-LoRA**, nicht unseren Merge-Ansatz

### Neue Hypothesen basierend auf Recherche

#### Hypothese 6: vLLM's WhisperEncoder nutzt MergedColumnParallelLinear für K/V
- Der PR-Kommentar erwähnt explizit *"MergedColumnParallelLinear K/V split"*
- Das könnte bedeuten: K und V sind in EINER fusionierten Linear-Schicht, Q separat
- Oder: Q, K, V sind ALLE in QKVParallelLinear fusioniert
- Die `stacked_params_mapping` in `load_weight()` zeigt: `q_proj`, `k_proj`, `v_proj` → `qkv_proj`
- **Wenn die Fusionierung die Weight-Layouts ändert**, könnten unsere separat geschriebenen Deltas (auf `attention.wq`, `attention.wk`, `attention.wv`) falsch im fusionierten Tensor landen

#### Hypothese 7: vLLM's Mel-Spectrogram unterscheidet sich von HF's WhisperFeatureExtractor
- vLLM: `torch.stft` + `mel_filter_bank` (aus `mistral_common`)
- HF: `WhisperFeatureExtractor` (aus `transformers`)
- Selbst kleine Unterschiede im Audio-Frontend könnten den LoRA-Effekt neutralisieren
- **Test**: Audio-Features aus beiden Pipelines extrahieren und vergleichen

#### Hypothese 8: Der LoRA-Effekt ist real, aber betrifft nur den LANGUAGE MODEL Teil
- LoRA targetiert nur attention + feed_forward im LM (210 Matrizen) + attention im Audio-Encoder (96 Matrizen)
- Die Audio-Features (conv layers, norms, embeddings) sind UNVERÄNDERT
- Wenn der LoRA-Effekt hauptsächlich im LM-Teil liegt, würde er bei Transkription weniger sichtbar sein als bei Text-Generation
- **ABER**: Der User berichtet, dass Transformer-Test ALLE trainierten Paare korrekt transkribiert → Effekt MUSS auch in Transkription sichtbar sein

---

# 📎 ADDENDUM (2026-07-21): Ursachenanalyse, Eliminierung, neue Wege

## TL;DR der neuen Analyse

Die meisten bisherigen Hypothesen (1, 2, 4, 6, 7) sind durch ein **Symmetrie-Argument** widerlegbar. Die zwei wahrscheinlichsten Ursachen liegen **nicht in der Gewichts-Mathematik**, sondern:
1. **H9 — Serving-Pfad**: Der befragte Server hat die gemergte Datei gar nicht ausgeliefert (systemd-Kollision oder falscher Modellpfad im Entrypoint). Konkreter Verdacht liegt im Repo begründet (s.u.).
2. **H10 — Evaluationslücke**: Ein kontrollierter A/B-Vergleich wurde nie ausgeführt (`ab_compare.py`, `compare_forward.py` sind ⏸️). Die Beobachtung "identisch zu Base" ist methodisch nicht abgesichert.

Das **Canary-Merge-Experiment** (s.u.) entscheidet in einem einzigen Schritt zwischen "Datei wird nicht gelesen" und "Effekt zu klein".

---

## 1. Logische Eliminierung: Was die Ursache NICHT sein kann

Kernbeobachtung: Die gemergte Datei hat **exakt dieselben 761 Keys** wie das Original — nur andere Werte in 306 Matrizen. Jeder Fehler in der Key-Remapping-/Fusions-Kette würde Base und Merged **identisch treffen**, denn die Keys sind dieselben.

| Hypothese | Verdikt | Begründung |
|-----------|---------|------------|
| H1 (falsches Audio-Mapping) | ❌ widerlegt | Wäre das Mapping falsch, läge auch das **Base**-Modell falsch/kaputt. vLLM transkribiert aber mit Base-Qualität. |
| H2 (Fused-QKV zerstört Deltas) | ❌ widerlegt | Der `weight_loader` mit `shard_id` schreibt Base- und Merged-Werte über denselben Pfad. Ein Bug dort müsste die Base-Transkription sichtbar brechen. |
| H6 (MergedColumnParallelLinear K/V) | ❌ widerlegt | Gleiches Symmetrie-Argument wie H2. |
| H3 (bf16→fp16 Cast) | ⬇️ stark abgeschwächt | bf16→fp16 ist für |x| ∈ [6e-5, 65504] **exakt** (fp16 hat 11 Mantissen-Bits vs. bf16 8, kein Overflow bei Gewichten ~1e-2). Die Deltas (mean 2e-4, max 7e-3) überleben den Cast verlustfrei. FP16-Rundung im Forward könnte den Effekt *abschwächen*, aber nicht über alle Tokens/Samples **bit-identisch** auslöschen — die HF-Seite zeigt den Effekt ja in fp16/bf16 deutlich. Größenordnung: mean\|ΔW\|/rms(W) ≈ 1–2 % pro Matrix × 30 LM-Layer → Logit-Änderungen weit oberhalb der fp16-Auflösung (ulp bei 1.0 ≈ 1e-3). |
| H4 / H7 (Mel-Frontend) | ❌ für das Kernsymptom | Frontend-Unterschiede treffen Base und Merged **innerhalb von vLLM gleichermaßen**. Sie können vLLM≠HF insgesamt erklären, nicht aber vLLM-Merged==vLLM-Base. (Für die HF↔vLLM-Parität bleiben sie relevant.) |
| H5 (falsches File) | ⬆️ **befördert → H9** | "MD5 ≠ Original" beweist nur, dass die Datei *auf Disk* anders ist — nicht, dass vLLM sie *geöffnet* hat. |
| H8 (Effekt nur im LM) | ➖ irrelevant fürs Symptom | Erklärt kein bit-identisches Verhalten. |

**Übrig bleiben nur Erklärungen, die vor der Gewichts-Mathematik ansetzen.**

---

## 2. Neue Hypothesen (nach Wahrscheinlichkeit)

### ⭐ H9 — Serving-Pfad: Der befragte Server hat die gemergte Datei nie ausgeliefert

"Bit-identisch zu Base" ist die exakte Signatur von "vLLM liest andere Bytes". **Konkreter, im Repo belegter Verdacht:**

- `scripts/04_install_voxtral_dgx_spark_container.sh` richtet einen **enabled** systemd-Dienst `voxtral-vllm` ein mit `Restart=always`, `RestartSec=5` (Zeilen 206–233), der `vllm serve mistralai/Voxtral-Mini-3B-2507` **aus dem HF-Cache** servt (Zeilen 192–202) — Container-Name `voxtral-vllm`, Host-Port **8000**.
- `scripts/v2/run_vllm_fixed.sh` nutzt **denselben Container-Namen** (`--name voxtral-vllm`) und **denselben Port** (`-p 8000:8000`).
- Kollisions-Szenario: War der Dienst nicht per `systemctl disable --now` gestoppt, belebt systemd den **Base-servenden** Container innerhalb von 5 s nach jedem `docker stop/rm`. Das Spike-`docker run` schlägt dann fehl (Name/Port belegt) — und jeder `curl`/Health-Check trifft den **alten Dienst**. Das erklärt *alle* drei Beobachtungen gleichzeitig: Health 200 ✓, Transkription funktioniert ✓, Output == Base ✓. Selbst "vLLM lädt ohne Crash" kann aus den Logs des alten Dienstes stammen (`journalctl -u voxtral-vllm` zeigt dann den Base-Start).
- Alternativ-Szenario: `entrypoint_merged.sh` (liegt **nur auf dem DGX**, nicht im Repo!) servt die HF-Modell-ID statt `/models/merged` — der Mount allein beweist nichts. Auch möglich: Fallback im Entrypoint, falls z.B. `params.json` in `/home/owc/voxtral-consolidated/` fehlt (`--config-format mistral` braucht es).

**Checks (5 Minuten, kein GPU-Work):**
```bash
sudo systemctl status voxtral-vllm --no-pager -l   # lief der alte Dienst?
docker ps --format '{{.Names}} {{.Image}} {{.Ports}}'
ss -tlnp | grep -E '8000|8001'                      # wer lauschte wann auf 8000?
cat /home/owc/voxtral-vllm/entrypoint_merged.sh     # welcher Modellpfad?
journalctl -u voxtral-vllm --since "2026-07-20" | grep -iE 'model|safetensors|merged'
curl -s http://127.0.0.1:8000/v1/models             # welche Modell-ID antwortet?
ls -la /home/owc/voxtral-consolidated/              # params.json vorhanden?
```

### H10 — Evaluationslücke: "Identisch zu Base" ist methodisch nicht abgesichert

- `ab_compare.py` und `compare_forward.py` wurden **nie ausgeführt** und sind nicht mal im Repo — es gibt keinen dokumentierten kontrollierten Vergleich.
- Unklar: Wurden die **trainierten Paare** durch vLLM geschickt, oder neues Audio? Bei einem kleinen LoRA (r=32), das die Trainingspaare weitgehend **memorisiert**, ist Base==Merged auf neuem/unähnlichem Audio in *jeder* Engine erwartbar.
- Unklar: Gab es überhaupt eine **vLLM-Base-Messung**, oder wurde gegen alte HF-Transkripte verglichen?
- Nötig: Vier-Quadranten-Vergleich {vLLM, HF} × {base, merged}, **gleiche Audios (trainierte Paare)**, gleiche Sprache, greedy.

### H11 — Prompt-/Frontend-Konditionierung (schwächer, aber prüfbar)

- Der LoRA wurde mit HF-`apply_transcription_request`-Prompts trainiert. vLLMs `/v1/audio/transcriptions` baut den Prompt über die eigene IO-Pipeline; der Dockerfile-cuFFT-Patch verlegt die STFT zusätzlich auf CPU/float32 (`04_install_...sh`, Zeilen 112–158). Features und Prompt-Assemblierung weichen also von HF ab.
- Kann **nicht** bit-identische Outputs innerhalb von vLLM erklären, wohl aber eine *abgeschwächte* LoRA-Wirkung und die HF↔vLLM-Paritätslücke.
- Check: HF-`inputs.input_ids` eines Trainingspaars dumpen und vLLM über `/v1/chat/completions` mit identischem Prompt + Audio füttern; außerdem Mel-Features beider Frontends für dieselbe WAV direkt vergleichen.

---

## 3. 🐤 Das entscheidende Experiment: Canary-Merge

Trennt in **einem Schritt** "Datei wird nicht gelesen" (H9) von "Effekt zu klein" (H3/H10/H11):

1. **Canary-Datei erzeugen** — eine dieser Varianten:
   - Merge erneut laufen lassen mit `scaling × 50` (eine Zeile in `merge_lora_native_mistral.py`), oder
   - chirurgisch: `output.weight` (LM-Head) in einer Kopie mit 0.01 multiplizieren → garantiert sichtbarer, deterministischer Output-Kollaps, keine NaNs.
2. Canary-Datei serven (gleicher Pfad, gleiches Script).
3. **Output unverändert** → *bewiesen*: vLLM liest die Datei nicht → H9, Serving-Pfad fixen.
4. **Output anders/kaputt** → Datei wird gelesen → mit kontrolliertem A/B (H10) und Prompt-Test (H11) weitermachen.

**Danach, in dieser Reihenfolge:**
1. Serving-Audit (Befehle unter H9) — inkl. MD5-Abgleich: V100-Merge-Output ↔ DGX-Datei ↔ HF-getestetes Verzeichnis.
2. Canary-Merge.
3. Vier-Quadranten-A/B auf trainierten Paaren (`ab_compare.py` endlich ausführen; greedy, `language=de`).
4. Prompt-Identitäts-Test + Mel-Parität (H11).
5. Erst falls alles unauffällig: `compare_forward.py` (Encoder-Outputs Base vs Merged in vLLM, identischer Mel-Input).

---

## 4. Merge-Script-Hygiene (unabhängig vom Befund)

Aktuell: `delta = scaling * (B[k].to(tensor.dtype) @ A[k].to(tensor.dtype))` — A/B werden **vor** dem Matmul auf bf16 (8 Mantissen-Bits) gecastet; das Delta selbst verliert ~2 Dezimalstellen.

Besser (kein kausaler Fix nötig, aber billig):
```python
delta = scaling * (B[k].float() @ A[k].float())          # Matmul in FP32
merged = tensor.float().add_(delta).to(tensor.dtype)     # ein einziger Final-Cast
```
Zusätzlich: Merge-Report (JSON) mit pro-Tensor |Δ| min/mean/max + MD5 von Input/Output; MD5 nach dem Kopieren auf den DGX erneut verifizieren (Datei-Identität V100↔DGX belegen).

---

## 5. Alternative Wege zum Ziel

### Weg A — Runtime-LoRA mit umbenanntem Adapter (Hybrid)
Weg 1 scheiterte nur an **Key-Namen** — das ist reine Umbenennung in `adapter_model.safetensors`:
- LM: nach Strip von `base_model.model.` liefert PEFT `model.language_model.layers.N...`; vLLM erwartet `language_model.model.layers.N...` (Fehlermeldung aus Weg 1). → Prefix-Regel `model.language_model.` → `language_model.model.`.
- Zielnamen verifizieren via `named_parameters` des geladenen vLLM-Modells.
- Audio-Tower: Runtime-LoRA braucht PR #45697 (in 0.17.1 nicht enthalten) → **Hybrid**: Tower-Deltas (96 Matrizen) statisch einmergen (Weg 3 behalten), LM-LoRA (210 Matrizen) zur Laufzeit via `--enable-lora --lora-modules`.

### Weg B — Gemergtes Modell im vorhandenen Transformers-Server serven ⭐ pragmatisch
- Der Beweis "Merged funktioniert" existiert bereits unter Transformers. `voxtral_server.py` per `VOXTRAL_LOCAL_MODEL=/pfad/zu/merged` auf das gemergte Verzeichnis zeigen lassen.
- Skalierung ohne vLLM: Worker-Pool nach dem `whisperx_spark`-Muster (2–4 fp16-Worker à ~8 GB — auf 128 GB Unified Memory problemlos) hinter dem FastAPI-Router. Verliert Continuous Batching, behält aber exakt die HF-Numerics, unter denen der LoRA-Effekt nachgewiesen ist. Für kurze VAD-Chunks ist Parallelität ≈ Worker-Zahl.

### Weg C — Engine-Achse
NVIDIA-Build 0.17.1+a03ca76a ist ein Fork; upstream-vLLM-Release oder neueren NVIDIA-Container testen, PR #45697 (Tower-LoRA) beobachten. Nach der Eliminierung oben **niedrige Priorität** — ein Loading-Bug, der nur Merged aber nicht Base trifft, ist logisch kaum möglich.

### Weg D — Domänen-Adaption in die Correction-Stage verlegen
Wenn das eigentliche Ziel "bessere medizinische Transkripte" ist (nicht "LoRA in vLLM" um seiner selbst willen): Base-Voxtral in vLLM lassen (funktioniert, schnell, parallel-fähig) und die medizinische Domänenanpassung ins Korrektur-LLM (Gemma 4, Port 9000) legen — Fine-Tune, Glossar oder Prompting. Engine-unabhängig, kein Audio-Modell-Chirurgie-Risiko, der Stack routet schon so. Vorab messen, ob die Endqualität damit den LoRA-Gewinn erreicht oder übertrifft.

---

## 6. Offene Fragen (blockieren die endgültige Ursache)

1. Inhalt von `/home/owc/voxtral-vllm/entrypoint_merged.sh` + vLLM-Startlog: Welcher Modellpfad und welche safetensors-Dateien wurden tatsächlich geladen?
2. War der systemd-Dienst `voxtral-vllm` vor dem Spike-Start per `systemctl disable --now` gestoppt? Wer hat während des Tests auf Port 8000 gelauscht (`docker ps`, `ss -tlnp`)?
3. Mit welchen Audios wurde "identisch zu Base" festgestellt — **trainierte Paare** oder neues Material? Wie viele Samples? Gab es eine vLLM-Base-Referenzmessung auf derselben Box?
4. Lief der Transformers-Referenztest in fp16 oder bf16 — auf der V100 oder dem Spark?
5. Wurde `deep_delta_analysis.py` auf der **DGX-Datei** oder dem V100-Output ausgeführt? Sind die MD5 von V100-Merge-Output, DGX-Datei und HF-getestetem Verzeichnis identisch?
6. Liegt ein `params.json` in `/home/owc/voxtral-consolidated/` (`--config-format mistral`)?
7. Welche Sampling-Parameter/Endpoint beim vLLM-Test (temperature=0? Proxy :8000 oder direkt :8001? `/v1/audio/transcriptions` oder Chat)?
