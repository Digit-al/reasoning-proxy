# Reasoning-Effort Proxy

Transparent proxy **Open WebUI → proxy → llama.cpp** qui décide, à chaque
prompt, combien d'effort de raisonnement le modèle principal doit fournir,
et le lui communique via `chat_template_kwargs.reasoning_effort`.

```
                 ┌────────────────────────┐
 Open WebUI ───► │        PROXY           │ ───► llama.cpp (modèle principal)
 (port 8080)     │  (FastAPI, ce projet)  │        ex. port 8081
                 │                        │
                 │  si pas d'effort fourni│──► SIDECAR : petit LLM OpenAI-compatible
                 │  → demande au sidecar  │    qui répond low / medium / xhigh
                 └────────────────────────┘
```

## Comportement

| Situation | Action du proxy |
|---|---|
| `POST */chat/completions` **sans** `reasoning_effort` | Envoie les derniers tours de conversation au sidecar (LLM), reçoit `low`/`medium`/`xhigh`, **injecte** `chat_template_kwargs.reasoning_effort` puis transfère |
| `reasoning_effort` déjà présent (dans `chat_template_kwargs` **ou** au top-level — c'est ce qu'envoie Open WebUI quand l'utilisateur remplit le champ *Advanced options*) | **Transfert byte-par-byte, sidecar non consulté** |
| Sidecar HS / timeout | Repli sur `DEFAULT_REASONING_EFFORT` (si défini), sinon transfert tel quel |
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
| `EFFORTS` | `low medium xhigh` | Valeurs autorisées. Pilote le parsing de la réponse du sidecar **et** le remplissage du placeholder `{efforts}` du prompt |
| `SIDECAR_PROMPT_FILE` | `sidecar_prompt.txt` | Fichier du prompt système du sidecar, à personnaliser librement (langue, critères, valeurs hardcodées). `{efforts}` est remplacé par la liste de `EFFORTS` |
| `DEFAULT_REASONING_EFFORT` | *(vide)* | Repli si sidecar HS / absent (vide = transfert tel quel) |
| `CONTEXT_TURNS` | `2` | Nb de tours de fin de conversation envoyés au sidecar |
| `MAX_PROMPT_CHARS` | `6000` | Budget de caractères envoyé au sidecar (le dernier message user est prioritaire) |
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
