# 08 — Orquestração, Turn-Taking e Barge-in

Este documento define a **máquina de estados da conversa** e o **barge-in**, um dos requisitos
mais sensíveis do sistema (NFR-L1: parar em ≤ 200 ms).

## 1. Framework: Pipecat

Usamos **Pipecat** como framework de orquestração de agente de voz (asyncio-native, com
suporte de primeira classe a pipelines de fala, interrupção e turn-taking).

- Docs: [Pipecat](https://docs.pipecat.ai/)
- Referências de interrupção/turn-taking para embasar o design:
  [Voice AI Barge-In and Turn-Taking (2026)](https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/),
  [Adaptive interruption handling — LiveKit](https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/),
  [Barge-in for on-device voice agents — EdgeAI](https://www.runedge.ai/blog/barge-in-interruption-handling-on-device-voice).

Podemos usar os componentes de pipeline do Pipecat plugando nossos serviços locais (VAD,
faster-whisper, cliente do vLLM/LFM2.5-VL-3B, XTTS-v2) como *services* custom. A máquina de
estados abaixo é o contrato de comportamento, independentemente de quanto do Pipecat reusamos.

## 2. Máquina de estados

```
        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
     ┌──────┐  speech_start        ┌───────────┐  speech_end       │
     │ IDLE │─────────────────────▶│ LISTENING │──────────┐        │
     └──────┘                      └───────────┘          │        │
        ▲                               ▲                 ▼        │
        │                               │            ┌──────────┐  │
        │ resposta                      │ barge-in   │ THINKING │  │
        │ terminada +                   │ (cancela)  └──────────┘  │
        │ silêncio                      │                 │        │
        │                          ┌──────────┐  1º áudio  │        │
        └──────────────────────────│ SPEAKING │◀───────────┘        │
                                   └──────────┘                     │
                                        │                           │
                                        └── speech_start (VAD) ─────┘
                                            = BARGE-IN
```

### Estados
- **IDLE:** sem fala; VAD ativo aguardando `speech_start`. Vídeo continua sendo amostrado.
- **LISTENING:** usuário falando; áudio → STT (parciais). No `speech_end` (endpoint), pega a
  transcrição final → `THINKING`.
- **THINKING:** monta prompt (histórico + janela de vídeo + texto) e roda o LLM em streaming.
  Pode incluir tool calls. Assim que sai o 1º trecho falável → `SPEAKING`.
- **SPEAKING:** TTS sintetiza e toca em streaming. **VAD continua ativo** para detectar
  barge-in. Ao terminar a fala e haver silêncio → `IDLE`.

### Transição especial: BARGE-IN
`speech_start` detectado em **THINKING** ou **SPEAKING** ⇒ barge-in:
1. Setar `cancel_token` → **aborta a requisição HTTP em streaming ao vLLM** (fecha a conexão
   SSE) e para a síntese do TTS.
2. Esvaziar a fila de playback (corta o áudio no meio).
3. Cancelar/descartar tool calls em voo (ver [`06`](06-llm-brain-generation.md)).
4. **Não commitar** a resposta parcial nem o bloco de frames do turno na sessão — o histórico
   de mensagens permanece no último estado consolidado (ver [`09`](09-kv-cache-generation-loop.md)
   §5 "Barge-in").
5. Ir para `LISTENING` e começar a transcrever a nova fala.

## 3. Orçamento de barge-in (NFR-L1 ≤ 200 ms)

A latência de barge-in é a soma de:

```
detecção VAD (≈30–50 ms de janela)  +
sinalização do cancel_token (≈1–5 ms) +
parada do TTS + flush da fila de áudio (≈20–80 ms, depende do buffer de playback)
────────────────────────────────────────────────────
alvo total ≤ 120 ms p50 / ≤ 200 ms p95
```

Regras de implementação:
- **Buffers de playback pequenos** (ex.: 20–40 ms) para que o flush corte o som quase
  instantaneamente. Buffer grande = barge-in "lento" mesmo com detecção rápida.
- **cancel_token compartilhado** (um `asyncio.Event` / token de cancelamento) que TTS, LLM e
  ferramentas checam entre unidades de trabalho.
- **VAD dedicado de baixa latência** rodando durante SPEAKING, separado do VAD de endpointing.

## 4. AEC — cancelamento de eco (crítico!)

Se o microfone capta o próprio TTS, o VAD dispara barge-in falso a cada resposta. Mitigações,
em ordem de robustez:
1. **AEC (Acoustic Echo Cancellation)** no pipeline de áudio de entrada (referência: o sinal
   que está sendo tocado). É a solução correta.
2. **Half-duplex gating** como fallback: durante SPEAKING, exigir do VAD um limiar mais alto
   e/ou casar com energia que **não** corresponde ao sinal do TTS.
3. Fone/echo cancel de hardware quando o form factor permitir.

Sem AEC, barge-in e SPEAKING brigam. Tratar AEC como requisito de M3, não "nice to have".

## 5. Distinção barge-in vs. backchannel (M5, FR-18)

Nem toda fala durante SPEAKING é interrupção — "aham", "sei", "certo" são *backchannels* que
não devem parar o assistente. Estratégia:
- Janela de confirmação curta: só tratar como barge-in se a fala do usuário **persistir** além
  de X ms (ex.: 300–500 ms) ou passar de um limiar de energia/duração.
- Opcional: classificador leve de backchannel. Fica para M5; no MVP, **qualquer** fala
  sustentada = barge-in (mais seguro que ignorar o usuário).

## 6. Vídeo durante todos os estados

A amostragem de vídeo e o encoder **rodam continuamente**, independentemente do estado da
conversa, atualizando a "janela de vídeo" (buffer compartilhado). No momento da montagem do
prompt (entrada em THINKING), o LLM lê o **snapshot atual** dessa janela. Detalhes de
sincronização e de como isso entra no KV cache em [`09`](09-kv-cache-generation-loop.md).

## 7. Contrato do orquestrador

```python
# orchestrator/state_machine.py (esboço)
class Turn:
    state: Literal["IDLE","LISTENING","THINKING","SPEAKING"]
    cancel_token: asyncio.Event
    session: Session              # doc 09 — histórico consolidado; NÃO recebe o turno abortado

async def on_vad_event(ev: VadEvent):
    if ev.kind == "speech_start" and state in ("THINKING","SPEAKING"):
        await barge_in()        # NFR-L1
    elif ev.kind == "speech_start" and state == "IDLE":
        enter("LISTENING")
    elif ev.kind == "speech_end" and state == "LISTENING":
        enter("THINKING")

async def barge_in():
    turn.cancel_token.set()     # fecha o stream HTTP do vLLM + aborta TTS/tools
    playback.flush()            # corta o áudio já enfileirado
    # nada a "rebobinar" explicitamente: turn.session.history só é atualizado no commit
    # de fim de turno (doc 09 §5) — como o turno atual nunca chega lá, o histórico já
    # está no estado correto.
    enter("LISTENING")
```

## 8. Checklist de aceitação (barge-in)
- [ ] Interromper no meio de uma frase corta o áudio em ≤ 200 ms (medido).
- [ ] Após barge-in, a nova fala do usuário é transcrita corretamente (sem lixo do turno
      anterior).
- [ ] Nenhuma tool call com efeito colateral é aplicada em duplicado após barge-in.
- [ ] O próprio TTS não dispara barge-in (AEC funcionando).
- [ ] KV cache consistente após barge-in (sem crescimento indefinido — NFR-R3).
