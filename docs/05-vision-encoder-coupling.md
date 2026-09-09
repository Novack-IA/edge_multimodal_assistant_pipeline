# 05 — Capacidades Multimodais Nativas do LFM2.5-VL-3B (sem treino no MVP)

> **Este documento foi reescrito em 2026-09-03.** A versão anterior descrevia como *treinar* um
> conector para acoplar um encoder de visão a um backbone LFM2.5-2.6B text-only, replicando o
> padrão LFM2-VL com pesos diferentes. Pesquisa mostrou que a Liquid AI já publica esse exato
> acoplamento **pronto**: o **LFM2.5-VL-3B**. Este documento agora descreve o que o modelo já
> faz nativamente, como consumimos isso, e o que sobra como trabalho opcional (M6).

## 1. O que é o LFM2.5-VL-3B

Publicado pela Liquid AI em **12/ago/2026**. Três partes, todas já treinadas e integradas pela
Liquid:

1. **Backbone de linguagem:** **LFM2.5-2.6B** (o mesmo backbone agentic considerado no plano
   original — arquitetura híbrida de convoluções curtas com double-gating + grouped-query
   attention).
2. **Vision encoder:** **SigLIP2 NaFlex 400M**.
3. **Projetor multimodal:** MLP integrado, já treinado.

~3,1B parâmetros totais, ~3 GB de memória, **contexto de 32.768 tokens** (menor que os 128K do
2.6B puro-texto — ver [`09`](09-kv-cache-generation-loop.md) §6), não-reasoning (responde direto,
latência baixa por design — ao contrário de variantes "Thinking" da família LFM2.5 que **não**
usamos aqui).

Fontes: [LFM2.5-VL-3B — Liquid Docs](https://docs.liquid.ai/lfm/models/lfm25-vl-3b),
[LFM2.5-VL-3B: A Better and Faster VLM for the Edge — Liquid AI Blog](https://www.liquid.ai/blog/lfm2-5-vl-3b),
[LiquidAI/LFM2.5-VL-3B — Hugging Face](https://huggingface.co/LiquidAI/LFM2.5-VL-3B).

## 2. Como a imagem entra (para entendimento — não implementamos isso)

Internamente (documentado para referência, não para reimplementar):
1. Imagens grandes são divididas em **tiles não-sobrepostos de 512×512**, mais um **thumbnail**
   global para contexto de imagem inteira.
2. O SigLIP2 NaFlex codifica cada tile em resolução/aspecto nativos.
3. O projetor MLP alinha os embeddings de visão ao espaço de tokens do backbone.
4. Tudo isso roda **dentro do forward do modelo servido pelo vLLM** — nosso código nunca vê um
   tensor de embedding de visão.

## 3. Como nós consumimos isso

Enviamos a imagem como **conteúdo de mensagem**, no formato multimodal OpenAI-compatible que o
vLLM expõe (`{"type": "image_url", "image_url": {...}}` ou equivalente do processor do modelo),
dentro da mensagem `role="user"` do turno atual — junto com o texto transcrito pelo STT. O
`chat template` do modelo (ChatML-like, com tokens especiais `<|im_start|>`/`<|im_end|>` e
marcação de imagem) é aplicado automaticamente pelo vLLM/processor; **não construímos esse
template manualmente**.

```python
# services/llm/client.py (esboço — ver contrato completo em doc 06 §4)
messages = [
    {"role": "system", "content": system_prompt},
    *history,  # texto apenas — ver doc 09
    {
        "role": "user",
        "content": [
            *[{"type": "image_url", "image_url": {"url": frame_to_data_url(f)}} for f in window_frames],
            {"type": "text", "text": user_text},
        ],
    },
]
```

## 4. Capacidades nativas relevantes ao produto

Números publicados pela Liquid (comparar/validar no device durante M2 — ver [`11`](11-testing-eval.md)):

| Capacidade | Benchmark | Resultado |
|---|---|---|
| Screen/document understanding | ScreenSpot-v2 (desktop/mobile/web) | 78.7 / 81.2 / 82.2 (média 80.7) |
| Grounding (bounding box/coordenadas) | RefCOCO-avg precision@1 | 87.9 (predecessor: 57.1) |
| Function/tool calling | ToolSandbox | 59.5 (predecessor: 26.4) |
| Function/tool calling | BFCL v4 | 32.5 (predecessor: 20.5) |
| Multi-imagem | BLINK / MuirBench | 61.5 / 58.3 |
| Vídeo curto (multi-frame) | 5-frame clip, TTFT | 34 ms (H100) |

Estas capacidades **não custam treino adicional** — a decisão de produto é só sobre expor ou não
cada uma (ver [`02`](02-requirements.md) §5/FR-8b/FR-8c).

## 5. Vídeo em tempo real = ainda amostragem de frames

Não há encoder de vídeo temporal nativo — vídeo continua sendo tratado como **amostragem de
frames**, agora anexados como itens de imagem na mensagem do turno em vez de embeddings
pré-computados. As regras da "janela de vídeo" (quantos frames, com que fps, como isso interage
com o orçamento de contexto) estão em [`09-kv-cache-generation-loop.md`](09-kv-cache-generation-loop.md).

Pontos de projeto (mudaram pouco em espírito, mudou o mecanismo):
- **Orçamento de tokens continua sendo o gargalo**, mas agora é o *processor* do modelo (tiling
  512×512 + thumbnail) quem decide quantos tokens cada imagem consome — não um
  `downsample_factor` que configurávamos nós. Medir empiricamente por imagem/resolução em M1/M2
  e usar isso para calibrar `video_token_budget` (config).
- **Amostragem esperta:** fps fixo baixo (2) por padrão; seleção por keyframe fica para M5,
  como no plano original.

## 6. Fallback de tamanho: LFM2.5-VL-1.6B

Existe uma variante menor, **LFM2.5-VL-1.6B**, publicada pela Liquid com o mesmo padrão de
acoplamento. Se o footprint/latência do 3B não couber no orçamento em algum cenário de
degradação (FR-21), trocar para o 1.6B é uma opção documentada — mesma interface de serving
(vLLM), mesma família de tool-call parser.

## 7. Upgrade opcional (M6): fine-tuning de domínio via LoRA

Só entra em escopo se a avaliação de M2 (VQA/grounding/tool-calling no domínio real do produto)
mostrar necessidade — por exemplo, vocabulário/objetos muito específicos que o modelo base erra
consistentemente. Se necessário:
- **LoRA/QLoRA**, nunca full fine-tune (mesmo cuidado do plano original: preservar a capacidade
  agentic é crítico).
- Mistura de dados deve incluir exemplos agentic/tool-calling (visuais e não-visuais) para não
  regredir o que já vem forte no checkpoint base.
- Medir contra o **próprio checkpoint base do LFM2.5-VL-3B** como baseline (não mais contra um
  "2.6B sem visão" — esse baseline não existe mais no nosso pipeline).

Isso é trabalho de pesquisa opcional e fora do caminho crítico do MVP — diferente do plano
original, onde era a **peça central e obrigatória** de M2.

## 8. Entregáveis deste componente (revisado)

- Validação de quantização (FP8/NVFP4/GGUF) vs. baseline BF16 em VQA/grounding/tool-calling —
  ver [`11`](11-testing-eval.md) §2.2.
- Confirmação de que o parser `lfm2` do vLLM extrai `tool_calls` corretamente ponta a ponta,
  inclusive quando a chamada é motivada pelo conteúdo da imagem (FR-11b).
- Calibração de `video_token_budget`/`window_frames` com medições reais de tokens-por-imagem no
  processor do modelo.
- (Opcional, M6) Adapter LoRA de domínio + relatório comparando contra o baseline do checkpoint.
