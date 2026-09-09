# 00 — Visão Geral do Projeto

## 1. Visão

Construir um **assistente multimodal conversacional que roda 100% on-device** em uma
plataforma NVIDIA Jetson (AGX Thor), capaz de ver o mundo por uma câmera em tempo real, ouvir e
responder por voz em português do Brasil, e executar ações via tool calling — sem
dependência de nuvem, preservando privacidade e eliminando latência e custo de rede.

## 2. Objetivos

- **G1.** Diálogo por voz natural em pt-BR com latência conversacional (round-trip alvo
  < 1 s; ver [`02-requirements.md`](02-requirements.md)).
- **G2.** Percepção visual em tempo real: o assistente responde sobre o que a câmera vê
  *agora*, com atualização contínua de frames.
- **G3.** **Um único modelo pronto** faz percepção visual + linguagem + tool calling (Opção A
  revisada): o **LFM2.5-VL-3B**, publicado já treinado pela Liquid AI — não construímos nem
  treinamos o acoplamento visão↔linguagem nós mesmos.
- **G4.** **Barge-in**: o usuário interrompe a fala do assistente naturalmente, como numa
  conversa humana.
- **G5.** **Voz clonada**: a voz de saída é clonada de uma amostra de referência via XTTS-v2.
- **G6.** Máximo de compute em **GPU local** (AGX Thor).

## 3. Não-objetivos (nesta fase)

- Treinar um LLM ou VLM do zero, ou treinar um conector visão↔linguagem próprio. Partimos dos
  pesos abertos e já-multimodais do LFM2.5-VL-3B (ver [`05`](05-vision-encoder-coupling.md)
  para o porquê desta mudança de plano).
- Compreensão temporal profunda de vídeo (reconhecer ações complexas ao longo do tempo).
  Começamos com **percepção per-frame** (o modelo suporta clipes curtos de poucos frames
  nativamente); temporalidade mais longa é um upgrade opcional (M6).
- Multi-idioma. Foco em **pt-BR** na saída; STT pode aceitar pt-BR (e opcionalmente en).
- Nuvem/telemetria remota.
- Fine-tuning/LoRA de domínio no caminho crítico do MVP — só entra se a avaliação de M2 mostrar
  necessidade real (M6, opcional).

## 4. Personas de uso

- **Robô/quiosque assistivo:** usuário conversa e aponta objetos; o assistente descreve,
  responde e executa comandos (tool calls) sobre dispositivos locais.
- **Companheiro embarcado:** rodando em robótica física (o alvo "Physical AI" da linha Thor).

## 5. Princípios de design

1. **Latência acima de tudo.** Cada estágio é streaming e mede seu próprio p50/p95.
2. **Um cérebro, não um pipeline de LLMs.** O LFM2.5-VL-3B já nasce multimodal — percepção de
   imagem entra nativamente como conteúdo de mensagem na API do modelo (servido via vLLM), não
   por um estágio de encoder/conector que nós construímos e treinamos.
3. **Estado explícito.** O contrato de turno/sessão (histórico, janela de vídeo do turno,
   commit/descarte em barge-in) é um objeto com ciclo de vida documentado — nunca efeito
   colateral de biblioteca, mesmo que o KV cache de baixo nível seja gerido pelo vLLM.
4. **Degradação graciosa.** Se a GPU saturar, reduzimos fps/resolução de vídeo, quantização do
   modelo e tamanho de beam antes de derrubar o diálogo.
5. **Tudo reproduzível.** Pesos versionados, versão do vLLM fixada, config única.

## 6. Glossário

| Termo | Significado |
|-------|-------------|
| **STT** | Speech-to-Text (reconhecimento de fala) |
| **TTS** | Text-to-Speech (síntese de fala) |
| **VAD** | Voice Activity Detection (detecção de atividade de voz) |
| **VLM** | Vision-Language Model — modelo que aceita imagem e texto no mesmo espaço de tokens |
| **Barge-in** | Usuário interrompe a fala do assistente; o turno volta pra ele |
| **Turn-taking** | Alternância de quem "fala" na conversa |
| **Endpointing** | Decidir quando o usuário terminou de falar |
| **KV cache** | Cache de chaves/valores da atenção, reutilizado a cada token gerado — gerido internamente pelo vLLM (ver [`09`](09-kv-cache-generation-loop.md)) |
| **PagedAttention** | Técnica do vLLM que gerencia o KV cache em páginas, como memória virtual |
| **Automatic Prefix Caching (APC)** | Reuso automático de KV cache do vLLM para prefixos de prompt idênticos entre requisições |
| **TTFT** | Time To First Token (latência até o 1º token do modelo) |
| **TTFB (áudio)** | Time To First Byte/chunk de áudio do TTS |
| **Grounding** | Capacidade do VLM de apontar coordenadas/bounding boxes de objetos na imagem |
| **Tool call parser** | Componente do vLLM que converte a saída pythonic do modelo em `tool_calls` estruturados (aqui, o parser `lfm2`) |
| **Backbone** | O modelo de linguagem base dentro do VLM (aqui, LFM2.5-2.6B, parte do LFM2.5-VL-3B) |
| **NaFlex** | Variante do SigLIP2 com resolução/aspecto nativos e flexíveis, usada como encoder de visão do LFM2.5-VL-3B |
| **NVFP4** | Formato de ponto flutuante 4-bit nativo do Blackwell (Jetson Thor), usado para quantização agressiva |

## 7. Componentes e por que cada um

- **Silero VAD** — endpointing e gatilho de barge-in leves e precisos.
- **faster-whisper** — STT robusto em pt-BR, implementação CTranslate2 ~4× mais rápida que o
  Whisper de referência, com quantização INT8.
- **LFM2.5-VL-3B (via vLLM)** — o cérebro único: agentic, tool calling nativo (texto **ou**
  imagem), grounding, leitura de tela/documento. ~3,1B parâmetros, ~3 GB em BF16 (menos
  quantizado), contexto de 32K tokens. Publicado já treinado pela Liquid AI — ver
  [`05`](05-vision-encoder-coupling.md) e [`06`](06-llm-brain-generation.md).
- **vLLM** — runtime de serving do LFM2.5-VL-3B, com suporte oficial da NVIDIA para Jetson
  Thor, PagedAttention, continuous batching, Automatic Prefix Caching e parser de tool calling
  dedicado à família LFM2 (ver [`13`](13-vllm-deployment.md)).
- **XTTS-v2** — TTS multilíngue com **clonagem de voz** a partir de uma única amostra e
  **streaming < 200 ms** de latência inicial; suporta português.
- **Pipecat** — framework de orquestração de agentes de voz asyncio-native, com suporte
  nativo a interrupção/barge-in e turn-taking.

Detalhes e citações de documentação em cada spec de componente e em
[`12-references.md`](12-references.md).
