# Reasoning-Effort Proxy

Transparent proxy **Open WebUI → proxy → llama.cpp** qui, à chaque prompt :
1. demande à un petit **sidecar** (LLM OpenAI-compatible) de **classifier**
   la dernière message — type de tâche (`creative` / `reasoning`) **et** niveau
d'effort ;
2. **route** la requête vers le bon backend (modèle de réflexion/agentique
ou modèle créatif) ;
3. **injecte** l'effort dans `chat_template_kwargs.reasoning_effort`.

```
                                    ┌──► LLM réflexion/agentique (backend principal)
 Open WebUI ───► PROXY ──► sidecar ─┤
 (port 8080)     (FastAPI)   (task +   └──► LLM créatif (CREATIVE_BACKEND)
                     effort)              (si configuré)
```

Sans `CREATIVE_BACKEND`, toutes les tâches vont au backend principal.
Le routage créatif est donc optionnel : le projet fonctionne exactement
comme avant avec un seul LLM.

## Comportement

| Situation | Action du proxy |
|---|---|
| `POST */chat/completions` **sans** `reasoning_effort` | Envoie les derniers tours de conversation au sidecar (LLM) avec `enable_thinking: false` (voir ci-dessous), reçoit une réponse JSON `{"task": "creative|reasoning", "effort": "..."}`, **route** la requête vers le bon backend, **injecte** `chat_template_kwargs.reasoning_effort` puis transfère |
| `reasoning_effort` déjà présent (dans `chat_template_kwargs` **ou** au top-level — c'est ce qu'envoie Open WebUI quand l'utilisateur remplit le champ *Advanced options*) | **Transfert byte-par-byte, sidecar non consulté** |
| `chat_template_kwargs` contient `"enable_thinking": false` | L'effort est sans objet : le prompt est transféré **tel quel** au LLM principal, sidecar non consulté, aucune injection |
| Réponse **streaming** (les cas ci-dessus + effort fourni par le client) | Le flux SSE est préfixé par des chunks `delta.reasoning` : « Détermination de l'effort de raisonnement… » puis « Effort: {effort} » (ou un simple écho « Effort: {effort} » quand le client a fourni sa valeur). Open WebUI les affiche dans la **boîte de réflexion** repliable, sans polluer le message ni l'historique. Désactivable via `EFFORT_NOTIFY=0` |
| Sidecar HS / timeout | Repli sur `DEFAULT_REASONING_EFFORT` (si défini), sinon transfert tel quel |
| `GET /v1/models` et `GET /models` (listes de modèles) | Chaque modèle est renvoyé avec `"loaded": true` et `"status": {"value": "loaded"}` (sauf si le backend fournit déjà un statut) → **point vert « chargé »** dans la liste des modèles OWUI (voir ci-dessous) |
| `/v1/models`, `/v1/completions`, audio, tools, SSE… | Transfert **100 % transparent** (body, headers, status, streaming SSE byte-par-byte) |

Le proxy n'altère **jamais** la réponse ni l'usage des tools : il ne touche
que la requête, et uniquement pour ajouter le `reasoning_effort` qui en
manquait.

## Configuration (variables d'environnement)

| Variable | Défaut | Description |
|---|---|---|
| `PROXY_HOST` / `PROXY_PORT` | `0.0.0.0` / `8080` | Bind du proxy |
| `LLAMA_BACKEND` | — (**requis**) | URL de base de llama.cpp **sans** `/v1`, ex. `http://127.0.0.1:8081` |
| `LLAMA_BACKEND_KEY` | — | API key optionnelle pour llama.cpp |
| `SIDECAR_BASE_URL` | — (**requis**) | Base OpenAI-compatible du sidecar, ex. `http://127.0.0.1:8082/v1` |
| `SIDECAR_API_KEY` | — | API key optionnelle du sidecar |
| `SIDECAR_MODEL` | — (**requis**) | Nom du modèle sidecar |
| `SIDECAR_TIMEOUT` | `10` | Timeout (s) de la requête au sidecar |
| `CREATIVE_BACKEND` | *(vide)* | Base URL du backend **créatif** dédié (llama.cpp ou autre endpoint OpenAI-compatible). Vide = les tâches créatives vont au backend principal |
| `CREATIVE_BACKEND_KEY` | — | API key optionnelle du backend créatif |
| `CREATIVE_MODEL` | *(vide)* | Nom du modèle à utiliser sur le backend créatif. Vide = on garde le modèle demandé par le client (si le même modèle est chargé des deux côtés) |
| `CREATIVE_EFFORTS` | `low medium high` | Échelle d'effort **des tâches créatives** (peut différer de l'échelle de raisonnement, ex. `high` au lieu de `xhigh`) |
| `EFFORTS` | `low medium xhigh` | Valeurs autorisées. Pilote le parsing de la réponse du sidecar **et** le remplissage du placeholder `{efforts}` du prompt |
| `SIDECAR_PROMPT_FILE` | `sidecar_prompt.txt` | Fichier du prompt système du sidecar, à personnaliser librement (langue, critères, valeurs hardcodées). `{efforts}` est remplacé par la liste de `EFFORTS` |
| `DEFAULT_REASONING_EFFORT` | *(vide)* | Repli si sidecar HS / absent (vide = transfert tel quel) |
| `CONTEXT_TURNS` | `2` | Nb de tours de fin de conversation envoyés au sidecar |
| `MAX_PROMPT_CHARS` | `6000` | Budget de caractères envoyé au sidecar (le dernier message user est prioritaire) |
| `EFFORT_NOTIFY` | `1` | Annoncer l'effort en streaming (`delta.reasoning`). `0` = désactiver l'annonce |
| `LOG_LEVEL` | `INFO` | `DEBUG` pour voir les flux complets |

## Démarrage

```bash
pip install -r requirements.txt

LLAMA_BACKEND=http://127.0.0.1:8081 \
SIDECAR_BASE_URL=http://127.0.0.1:8082/v1 \
SIDECAR_MODEL=qwen3-1.7b \
DEFAULT_REASONING_EFFORT=medium \
python3 proxy.py
# si les efforts de ton modèle principal diffèrent :
EFFORTS="low medium xhigh" python3 proxy.py
# ou un prompt de sidecar entièrement personnalisé :
EFFORTS="low medium xhigh" SIDECAR_PROMPT_FILE=/etc/proxy/sidecar_prompt.txt python3 proxy.py

# Routage 2 LLM : réflexion/agentique (principal) + créatif (dedicated backend)
LLAMA_BACKEND=http://127.0.0.1:8081 \
CREATIVE_BACKEND=http://127.0.0.1:8082 \
CREATIVE_EFFORTS="low medium high" \
CREATIVE_MODEL=mon-llm-creatif \
python3 proxy.py
```

Docker :

```bash
docker run -d --name reasoning-proxy -p 8080:8080 \
  -e LLAMA_BACKEND=http://host.docker.internal:8081 \
  -e SIDECAR_BASE_URL=http://host.docker.internal:8082/v1 \
  -e SIDECAR_MODEL=qwen3-1.7b \
  -e DEFAULT_REASONING_EFFORT=medium \
  reasoning-proxy:latest
```

## Côté Open WebUI

Admin → **Settings → Connections** → nouveau backend :

* **API base** : `http://<hote-proxy>:8080/v1`
* Type : *OpenAI*

Le test de connexion passe (`/v1/models` est transféré tel quel).

### Point vert « modèle chargé »

Comme avec une connexion directe à llama.cpp (où le statut est fourni
par llama.cpp lui-même, sans aucune configuration), le proxy signale
les modèles comme chargés :

* `GET /v1/models` et `GET /models` : chaque modèle est renvoyé avec
  `"loaded": true` et `"status": {"value": "loaded"}` (sauf si le
  backend fournit déjà un statut, auquel cas il est respecté — ex.
  `unloaded` après un sleep).
* Le point vert apparaît donc **avec n'importe quel Provider**
  ("Défaut" inclus), sans rien changer dans la connexion OWUI.
* Optionnel : **Provider = `llama.cpp`** dans la connexion OWUI fait
  de plus respecter finement les états `loading`/`unloaded` et active
  les boutons charger/éjecter du panneau admin (`/models/load`,
  `/models/unload`, bien servis par llama.cpp).

## Côté llama.cpp (modèle principal)

Le modèle doit avoir une *chat template* Jinja qui consomme la variable
`reasoning_effort` (ex. modèles Qwen3/R1 style GGUF avec template
paramétrable). llama.cpp transmet `chat_template_kwargs` au template, donc
`reasoning_effort` injecté par le proxy devient une variable du template —
c'est exactement le même canal que le champ *Advanced options →
reasoning_effort* d'Open WebUI (qui passe dans `chat_template_kwargs`).

## Côté sidecar

N'importe quel endpoint OpenAI-compatible :

* une seconde instance `llama-server` sur un petit modèle rapide
  (Qwen3 1-4B, Gemma 3 1-4B…) :

  ```bash
  llama-server -m qwen3-1.7b-instruct-q4_k_m.gguf -a 127.0.0.1:8082 \
    --jinja --alias sidecar-model
  ```

* ou une API externe (Ollama `http://host:11434/v1`, API Anthropic via
  compatibilité OpenAI, etc.) via `SIDECAR_BASE_URL` / `SIDECAR_API_KEY`.

### Prompt du sidecar

Le prompt système est lu depuis `sidecar_prompt.txt` (ou `SIDECAR_PROMPT_FILE`).
Le placeholder `{efforts}` y est remplacé à l'arrivée avec la liste de
`EFFORTS` ; les critères par effort sont dans le fichier, à adapter à tes
vrais efforts. Le fichier est optionnel : s'il est absent, un prompt
intégré est utilisé (le parsing, lui, ne suit que `EFFORTS`).

Le proxy lui envoie :

```json
{
  "model": "<SIDECAR_MODEL>",
  "messages": [
    {"role": "system", "content": "…classifie low/medium/xhigh…"},
    {"role": "user", "content": "Conversation (dernier message EN DERNIER) :\n…"}
  ],
  "max_tokens": 16, "temperature": 0, "stream": false
}
```

et attend un seul mot parmi `EFFORTS` : `low`, `medium` ou `xhigh` par défaut.

## Tests

```bash
python3 tests/run_tests.py
```

Lève un backend mock, un sidecar mock et le proxy, puis vérifie :

* effort injecté quand le client n'en donne pas (stream + non-stream) ;
* transfert byte-par-byte + sidecar **non consulté** quand le client fournit
  `reasoning_effort` (dans `chat_template_kwargs` ou au top-level) ;
* transfert tel quel + sidecar **non consulté** quand
  `chat_template_kwargs` contient `"enable_thinking": false` ;
* `GET /models` et `GET /v1/models` → modèles signalés `loaded: true`
  + `status.value = "loaded"` (point vert OWUI, provider « Défaut »
  inclus), statuts explicites du backend respectés ;
* préservation des `tools`/`tool_choice`, des autres `chat_template_kwargs` ;
* `/v1/models` transparent ;
* repli `DEFAULT_REASONING_EFFORT` quand le sidecar est mort.

## Notes

* `GET /proxy-health` : état du proxy (backend, sidecar, efforts).
* L'**alias** du modèle principal est bien vu par Open WebUI : `/v1/models`
  est transféré sans modification, donc le nom listé est celui que publie
  llama.cpp (`--alias` ou nom du GGUF), et le champ `model` de la requête
  est transmis tel quel.
* Chaque décision est journalisée : `sidecar decided effort=high (0.41s)`,
  `client supplied reasoning_effort='low' -> forwarding as-is`,
  `sidecar unavailable (...)` + repli.
* Le proxy n'ajoute **aucun** champ à la réponse et laisse passer les
  erreurs du backend avec leur status code d'origine.
