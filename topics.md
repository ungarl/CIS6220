---
layout: default
title: Topics & Readings
---

## Course Topics & Readings

This is the week-by-week curriculum: topics, readings, and key concepts. For the actual
calendar — dates, holidays, midterms, and project deadlines — see the [Schedule](schedule.html).

This content is tentative and may be adjusted based on class progress and interests. Readings
should be completed before the listed class session.

The readings listed under each week are the assigned reading — read these before class.
Deeper/more current/background material used in preparing each lecture lives separately in
the instructor's own lecture notes, not here, so this list doesn't grow into more than
students can reasonably read.

---

### Week 0: Course Introduction, Deep Learning Review & Attention

This week sets the shared vocabulary and mental models — architectures, losses, optimizers,
regularization, and attention — that every later week assumes. Deeper question: what are the
basic building blocks of a neural network, and why do these particular choices (not others)
tend to work?

**Tuesday: Course Overview; How to Read a Paper; Deep Learning Review**

- Course overview and introductions: structure, infrastructure, grading
- How to read a paper: the C-C-C framework (Context-Content-Conclusion), plus a working set of
  questions for every paper this semester — what problem is being addressed, why it matters,
  (optionally) how would *you* address it before seeing the paper's answer, what task they apply
  it to, how they evaluate it, and the key concepts/terms
- Deep learning review: learning paradigms (supervised, unsupervised/semi-supervised,
  reinforcement), data modalities (tabular, language, vision, speech, video, multimodal), and
  where retrieval/tool use fit in (RAG, MCP, harnesses)
- Core concepts survey — model form (activations, CNN vs. RNN/LSTM vs. Transformer, architecture
  depth), loss function and regularization (L1/L2, dropout, early stopping), and optimization
  (SGD, minibatch size, gradient clipping, Adam = Adagrad + RMSProp)
- Three concepts get a problem → example → solution treatment, the same lens used for reading
  papers all semester:
  - **Adagrad**: problem (SGD's one global learning rate doesn't fit all parameters,
    especially with sparse features) → example (rare-word embeddings barely update under a
    fixed LR while common-word embeddings update fine) → solution (adapt the learning rate
    per-parameter from historical gradient magnitude)
  - **Skip connections / ResNet**: problem (plain deep networks perform *worse* than shallower
    ones — the degradation problem — despite being able to represent everything a shallow net
    can) → example (an 18-layer plain CNN beating a 34-layer plain CNN on ImageNet) → solution
    (residual connections: each block only learns a residual from the identity mapping)
  - **BatchNorm**: problem (each layer's input distribution keeps shifting as earlier layers'
    weights update, forcing tiny learning rates and careful initialization) → example (a
    moderately large learning rate causing activations to blow up or vanish layer-by-layer as
    training progresses) → solution (normalize each layer's activations per mini-batch, then
    learn a scale/shift)
- Minibatch size, revisited twice: first as a statistical tradeoff (larger batches → smoother
  gradients and better GPU utilization but need learning-rate scaling and can generalize worse;
  smaller batches → noise acts as a regularizer but hardware efficiency and wall-clock time
  suffer), then as a hard memory constraint (worked example: an 8B-parameter model's
  mixed-precision optimizer state alone is ~128GB before a single activation — more than a
  single 80GB GPU holds — so minibatch size is often just whatever's left over after the model's
  own bookkeeping)

**Thursday: Attention**

- Why RNN/LSTM sequence transduction is inherently sequential — can't parallelize across time
  steps — and still struggles with long-range dependencies, motivating "Attention Is All You
  Need"
- Self-attention (Query, Key, Value); multi-head attention
- Attention applications: machine translation, question answering, speech recognition
- Positional encoding; masking
- Transformer architecture overview — sets up Week 1's deeper treatment (tokenization, scaling
  laws)

**Readings:**

- None assigned — Thursday works through Vaswani et al., "Attention Is All You Need" (2017)
  live rather than assigning it as a separate reading

**Key Concepts:**

- How to read a paper: Context-Content-Conclusion; problem/importance/approach/evaluation
- Architecture, loss function, and optimization choices; inductive bias and cost tradeoffs
- Adagrad, skip connections/ResNet, BatchNorm as worked problem/example/solution demonstrations
- Minibatch size as both a statistical tradeoff and a memory-budget constraint
- Self-attention, multi-head attention, positional encoding, masking — foundation for Week 1

---

### Week 1: NLP and Attention Mechanisms

Week 0 already covered how attention works — self-attention, multi-head attention, positional
encoding, masking, the core "Attention Is All You Need" mechanics. This week goes deeper on what
breaks when you actually try to run that mechanism at scale: how do you make attention cheaper
over long documents, and does the resulting model actually *use* everything in a long context
well? Deeper question: attention's basic mechanics don't change with scale — what does?

**Tuesday: Long Context — Sparse Attention and Whether It's Used**

- **HW0 recap**: what the minibatch-size experiment showed about compute time and test accuracy
  as batch size varies, and whether AdamW is the same as plain Adam
- Standard self-attention's quadratic cost in sequence length, and what that forces you to trade
  off to handle long documents cheaply
- Longformer: sliding-window (local) attention + global attention on a small set of task-chosen
  tokens; dilated windows to grow the receptive field across layers without added compute;
  separate Q/K/V projections for local vs. global attention
- "Lost in the Middle": even when a model's context window comfortably covers the test sequence
  length, performance is U-shaped — best near the start/end, worst in the middle — indicating a
  learned/training-data effect, not a length limitation
- Bridge to Thursday: between 2019-2021 the field chased new attention *patterns* (Longformer,
  Performer, Reformer) to cut FLOPs; today most frontier models use standard full attention with
  FlashAttention instead — why?

**Thursday: FlashAttention — Making Exact Attention Fast**

- IO-awareness: the real bottleneck in attention isn't FLOPs, it's data movement between GPU HBM
  (large, slow) and SRAM (small, fast) — standard attention materializes and repeatedly
  reads/writes the full N×N score matrix
- Tiling + online/streaming softmax: compute an exact softmax incrementally, block by block,
  without ever materializing the full attention matrix in memory
- Recomputing the attention matrix during the backward pass instead of storing it — more FLOPs,
  but far less memory traffic, and still a net win
- Why an exact, IO-aware method beats approximate/sparse methods (Tuesday's readings) on
  wall-clock time despite doing "more" math
- Bridge forward: FlashAttention fixes the *training*-time bottleneck; during generation the
  analogous bottleneck becomes the KV cache
- **KV-cache efficient attention**: MHA (baseline) → MQA (share K/V across all heads — large
  memory win, quality cost) → GQA (grouped sharing — production standard: Llama 2/3, Mistral) →
  MLA (DeepSeek-V2/V3: compress K/V into a shared low-rank latent — smaller cache than GQA,
  near-MHA quality) → sparse indexers (shrink *how many* tokens are attended to, not what's
  stored per token; composes with MLA)

**Ongoing Theme: Evaluation** (introduced here; revisited for reasoning models in Week 8 and
for agents in Week 11)

- Does a longer context window actually get used well, or does it just exist? "Lost in the
  Middle": models attend unevenly across long contexts even when the encoding/window supports
  the length
- How do we know a model is actually good more generally? Benchmarks and held-out test sets,
  perplexity vs. downstream task performance
- Why this matters more as models get harder to evaluate by inspection (scale, reasoning, agency)

**Readings:**

- Beltagy, Peters, Cohan, "Longformer: The Long-Document Transformer" (2020)
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023)
- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (2022)

**Key Concepts:**

- Quadratic attention cost; sparse/local attention (sliding window, dilation, global attention)
  as one architectural route to cheaper long context
- Evaluation as a recurring theme; "Lost in the Middle" as a concrete test of whether long
  context is actually used well, independent of window size
- IO-awareness, tiling, online softmax; why exact + hardware-aware can beat approximate
- FLOPs vs. wall-clock time; why the field converged on FlashAttention over custom sparse
  attention patterns

**Not covered this week** (moved out of the old plan — see `lecture_notes/supplemental_papers.md`
for where this content is parked, pending a decision on which week it belongs to): RoPE/YaRN
positional-encoding extrapolation, Kaplan/Chinchilla scaling laws, and the MHA→MQA→GQA→MLA
KV-cache-compression lineage. **Flagged for Lyle** — these were the prior plan's Week 1 content
but don't match what's actually been assigned to students; they need a new home in the schedule.

---

### Week 2: Computer Vision and Segmentation

Object detection and segmentation are about getting a model to answer "what is where."
YOLO and U-Net solved this efficiently for a fixed, closed vocabulary of object classes; SAM
asked a harder question — can one model segment *anything*, given just a point or a box, with
no retraining for new categories? Cool application: SAM-family models now power real-time video
editing, robotics perception, and medical-image annotation pipelines.

**Tuesday: Object Detection**

- YOLO (You Only Look Once) - architecture and evolution (v1 to v11)
- Loss functions: focal loss, anchor boxes
- U-Net architecture for medical image segmentation
- Skip connections, transposed convolutions
- Data augmentation strategies

**Thursday: Segmentation**

- Segment Anything Model (SAM, SAM2, SAM3)
- Memory attention, prompt encoders, mask decoders
- Video Object Segmentation (VOS)
- SAM3's shift: from promptable *visual* segmentation (click a point) to promptable *concept*
  segmentation (describe it in text, find every instance)
- Data curation and quality (Llama 3, Olmo, DeepSeek)

**Readings:**

- Redmon et al., "You Only Look Once" (YOLO, 2015)
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Kirillov et al., [Segment Anything](https://arxiv.org/abs/2304.02643) (SAM, 2023); Explained: [SAM - The Complete Guide](https://viso.ai/deep-learning/segment-anything-model-sam/)

**Key Concepts:**

- Object detection, segmentation, scene understanding
- Foundation models, zero-shot transfer
- Data curation process

---

### Week 3: VLMs, multimodal models, diffusion models

How do you get a single model to connect a caption to a picture? CLIP's contrastive trick —
pull matching image/text pairs together in a shared embedding space, push mismatched pairs apart
— became the substrate for nearly every text-to-image system since. Deeper question: how do you
use language to control image generation, and where does "meaning" actually live in that shared
embedding space? Aside: most of the current image-generation frontier (FLUX.2, Stable Diffusion
3.5) is still built on the same latent-diffusion + text-conditioning recipe taught here — the
advances since 2022 are mostly in fidelity and control, not a new paradigm. Complication: Google's
"Nano Banana" line briefly tested the alternative — the original Nano Banana (Gemini 2.5 Flash
Image, Aug 2025) generated images autoregressively, as LLM-style tokens, before Nano Banana 2
reverted to a diffusion decoder — a real, if short-lived, run at a genuinely different paradigm.

**Tuesday: Multimodal Architectures**

- CLIP (Contrastive Language-Image Pre-training)
- Early vs late (deep) fusion
- ViLBERT and co-attention mechanisms
- Q-Former / Resampler (BLIP-2)
- Contrastive learning

**Thursday: Diffusion Models**

- Variational Autoencoders (VAE)
- Diffusion process: forward (noising) and reverse (denoising)
- Latent Diffusion Models
- DALL-E 2 architecture: prior networks, decoders
- Noise scheduling
- Reparameterization trick

**Readings:**

- Lu et al., [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265) (2019)
- Radford et al., [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021)
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion, 2022); Explained: [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- Ramesh et al., [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) (DALL-E 2, 2022)

**Key Concepts:**

- U-Net, early/late fusion
- Contrastive pre-training
- VAE, Diffusion models, latent space

---

### Week 4: MCP, Harnesses; Agentic AI

How should an AI system get access to knowledge and capabilities it wasn't trained on? RAG
answers this for knowledge — retrieve, then generate. MCP and tool use answer it for
capabilities — a model that can act, not just recite. Deeper question: what should live inside
the model's weights vs. outside it in the surrounding system, and who controls that boundary?

**Tuesday: RAG, Tool Use, and Harnesses**

- RAG architecture: retriever + generator; vector databases and document chunking;
  limitations and alternatives (memory networks) — covered in lecture, no assigned reading
- MCP learning to use APIs (Toolformer): self-supervised training for tool use, no
  hand-labeled API-call examples needed
- Harnesses, introduced: what surrounds the model (tool permissions, context management) —
  the basic vocabulary Thursday and Week 11 both build on
- Design choices: knowledge location (in the weights vs. the prompt vs. the harness), control
  flow

**Thursday: Agentic AI — Single Agents and Multi-Agent Systems**

- What is an agent? The reason-act loop (ReAct): interleaving reasoning traces with tool
  calls, the base pattern nearly every agent framework wraps something around
- Single agent vs. multiple agents: when does splitting a task across agents help (parallel,
  independent sub-tasks) vs. hurt (added coordination cost, token cost, new failure modes)?
- A taxonomy of how multi-agent systems actually fail in practice (coordination breakdowns,
  not just model capability limits) — the design-level question introduced here; Week 11
  covers the orchestration mechanics and evaluation benchmarks in depth

**Readings:**

- (Tuesday) Schick et al., [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (2023); Explained: [How does AI learn to use tools? - Toolformer explained](https://www.youtube.com/watch?v=hI2BY7yl_Ac)
- (Tuesday) Zhang, Wang, Ge, Xu, Hamm, and Reddy, ["Stop Comparing LLM Agents Without Disclosing the Harness"](https://arxiv.org/abs/2605.23950) (2026) — the Binding Constraint Thesis: for long-horizon agent tasks, harness configuration (context construction, tool routing, orchestration, error recovery) explains more performance variance than model choice does
- (Thursday) Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023) — the origin of the reason+act loop every harness wraps around
- (Thursday) Zhuge et al., [GPTSwarm: Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) (ICML 2024) — represents multi-agent systems as computational graphs and optimizes both node prompts and the graph's own wiring

RAG is covered in lecture (retrieve-then-generate architecture, vector DBs/chunking, alternatives) but has no assigned reading this year.

**See also (optional, background):**

- (Tuesday) Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, 2020) — the lecture's own reference for the architecture, not an assigned reading
- (Tuesday, harness) Kim et al., "The Interplay of Harness Design and Post-Training in LLM Agents" (2026) — treats the harness (tool exposure, tool descriptions) as a design dimension separate from training; a distinct paper from this week's required harness reading above
- (Tuesday) Anthropic, [Model Context Protocol specification](https://modelcontextprotocol.io) (2024) — the actual MCP spec this topic is named for
- (Tuesday) Anthropic, ["Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents) (Dec 2024) — the canonical workflows-vs-agents framework
- (Tuesday) Bai et al., [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (2022) — training principles into the weights instead of the harness enforcing them; the "inside the model" counterpart to this week's harness material
- (Thursday) Cemri et al., [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) (2025) — a taxonomy of 14 real multi-agent failure modes across 3 categories, from 150 annotated production traces
- (Thursday) Anthropic, ["Effective harnesses for long-running agents"](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) (Nov 2025) — direct engineering treatment of harness design (context management, sandboxing, assumptions going stale as models improve); assigned in full at Week 11 once multiple agents are coordinating
- (Thursday) Anthropic, ["How we built our multi-agent research system"](https://www.anthropic.com/engineering/multi-agent-research-system) (June 2025) — the orchestrator-worker architecture behind Claude's Research feature, with real cost numbers
- (Thursday) Anthropic/Claude, ["When to use multi-agent systems (and when not to)"](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) — short practitioner framing of the single-vs-multi-agent design question

**Key Concepts:**

- RAG architecture; API integration with LLMs via MCP
- Harnesses vs. models: what goes inside the weights vs. outside in the surrounding system
- Agentic AI systems: the reason-act loop
- Single-agent vs. multi-agent design tradeoffs (parallelism and cost vs. coordination risk)

---

### Week 5: Midterm 1

Only session this week is **Midterm 1, Tuesday** (covers everything through Week 4). Thursday
is Fall Term Break — no class. No new content this week.

---

### Week 6: Mechanistic Interpretability and Steering

Once a model can do something, how do you make it do a *specific* thing reliably — and how do
you know what's happening inside when you try? Deeper question: how should AI be steered — by
changing its weights (fine-tuning), by changing its input (prompting/in-context learning), or by
directly editing its internal activations?

**Tuesday: Five Ways to Steer, and Looking Inside**

- The steering landscape: what you change — all the weights (SFT, RLHF/DPO), a low-rank slice
  of them (LoRA, prefix tuning), the input (prompting, in-context learning), the surrounding
  harness, or the activations themselves
- LoRA vs. full fine-tuning: why a low-rank update learns less but forgets less; in-context
  learning as implicit fine-tuning
- Probing: linear probes and what they can and can't show; the linear representation hypothesis
- Superposition and sparse autoencoders: why neurons are polysemantic, and how dictionary
  learning recovers interpretable features
- Transformer circuits: the residual stream as an additive channel, QK/OV circuits, induction
  heads
- Attribution graphs in a production model: multi-step reasoning, planning, and why models
  hallucinate

**Thursday: Steering with Activations**

- Contrastive Activation Addition: steering vectors from contrastive pairs, which layers work,
  and flipping a behavior with a sign change
- Feature steering with SAEs (Golden Gate Claude) vs. activation addition
- AxBench: detecting a concept vs. steering it, and why prompting still wins at steering
- Emergent misalignment: found by probing, controlled by steering
- When to steer instead of prompt or fine-tune

**Readings:**

- Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021) — Tuesday; paired with [Neel Nanda's walkthrough video](https://www.youtube.com/watch?v=KV5gbOmHbjU)
- Lindsey et al., [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (2025) — Tuesday; attribution graphs
- Panickssery et al., [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) (2024) — Thursday
- Wu et al., [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders](https://arxiv.org/abs/2501.17148) (2025) — Thursday

**See also (optional, background):**

- Lindsey et al., "Circuit Tracing: Revealing Computational Graphs in Language Models" (2025) — the methods companion to "On the Biology"
- Bricken et al., "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning" (2023) — sparse autoencoders
- Templeton et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (2024)
- Alain & Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes" (2016) — the classic linear-probing paper
- Turner et al., "Activation Addition: Steering Language Models Without Optimization" (2023) — the original activation-steering paper
- Zou et al., "Representation Engineering: A Top-Down Approach to AI Transparency" (2023)
- Dai et al., "Why Can GPT Learn In-Context?" (2023) — in-context learning
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — fine-tuning, covered in lecture

**Key Concepts:**

- Ways to adapt model behavior: fine-tuning (SFT, LoRA), prompting/in-context learning, harness, activation steering
- Probes; linear representation hypothesis; superposition; sparse autoencoders
- Residual stream; QK/OV circuits; induction heads; attribution graphs
- Steering vectors (difference of means); reading vs. writing a concept

---

### Week 7: Distillation; Efficient Architectures: MoE and Quantization

Distillation asks: can a small model learn to imitate a big one's behavior, capturing most of
its capability at a fraction of the cost? Cool current example: DeepSeek's R1-distilled
models — small models trained purely by fine-tuning on reasoning *traces* generated by R1, with
no RL of their own — beat other similarly-sized models on math benchmarks. It's a direct,
current illustration of the distillation idea taught here, and a callback to Week 8's reasoning
models.

Distillation makes a big model tractable *after* training, for deployment. The other half of
"making big models tractable" happens *before* training even starts: how do you train a model
too large to fit on one device in the first place? Two complementary strategies: shard the
model itself across devices (tensor/model parallelism), or shard the optimizer state/gradients
instead of replicating them everywhere (ZeRO/FSDP-style memory sharding). This is the technical
foundation behind several of the "Training Efficiency" and "Memory Optimization" final project
ideas.

As models get too big to run cheaply, two more questions emerge: how do you activate only
*part* of a giant model per token instead of all of it (MoE), and how do you represent its
weights with fewer bits without breaking it (quantization)? Cool application: DeepSeek-V3's
combination of MoE and low-precision training is a big part of why it could be trained for a
fraction of the compute of comparably capable dense models.

**Tuesday: Knowledge Distillation; Training Efficiency at Scale**

- Distillation: teacher-student models
- Soft softmax with temperature
- Distilling Step-by-Step
- Reduced precision models (Deepseek example)
- DeepSeek-R1-Distill: distilling reasoning traces (not just outputs) into small dense models
- Training efficiency: tensor/model parallelism (Megatron-LM) vs. memory-efficient sharding of
  optimizer state and gradients (ZeRO/FSDP) — training-time tractability, distillation's
  deployment-time counterpart

**Thursday: Mixture of Experts and Quantization**

- Mixture of Experts (MoE) architectures
- Expert routing and load balancing
- Llama-3, 4 architecture: MoE, Grouped-query attention, RoPE
- GLaM, Switch Transformers
- DeepSeek-V3: MoE + multi-head latent attention + auxiliary-loss-free load balancing, at 671B
  total / 37B activated parameters
- Post-training quantization (PTQ) vs. quantization-aware training (QAT)
- GPTQ, LLM.int8() — 8-bit/4-bit weight and activation quantization
- QLoRA: quantization + LoRA for efficient fine-tuning

**Readings:**

- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015); Explained: [Distilling the Knowledge in a Neural Network](https://www.youtube.com/watch?v=EK61htlw8hY)
- Sanh et al., [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) (2019)
- DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024) — §3.2.3 (MoE routing) and §3.3 (FP8 quantization)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022/ICLR 2023)

**See also (optional, background):**

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019) — training-time parallelism, covered in lecture
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019) — covered in lecture
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022) — alternate quantization angle
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- Fedus et al., "A Review of Sparse Expert Models in Deep Learning" (2022) — last year's MoE background reading, now dated

**Key Concepts:**

- Distillation; reduced precision models; distilling reasoning traces vs. distilling outputs
- Tensor/model parallelism; memory-efficient optimizer/gradient sharding (ZeRO/FSDP)
- Mixture of Experts (MoE)/Sparse models
- Quantization: PTQ vs. QAT, bit-width tradeoffs

---

### Week 8: Reinforcement Learning - Policy Optimization

How do you get a model to do the right thing when "right" can only be judged after the fact,
not labeled in advance? RLHF and its descendants (DPO, GRPO) turn human or automated preference
judgments into a training signal — and the same machinery, pointed at *verifiable* rewards
instead of human preferences, is what trains today's reasoning models.

**Tuesday: RLHF and Policy Gradients**

- Reinforcement learning fundamentals (V, Q, policy)
  - On-policy vs off-policy learning
  - Model-based vs model-free RL
- RLHF (Reinforcement Learning from Human Feedback)
- PPO (Proximal Policy Optimization)
- Actor-critic methods
- Generalized Advantage Estimation (GAE)
- REINFORCE for LLMs

**Thursday: Direct Preference Optimization; Reasoning Models & Evaluation**

- DPO (Direct Preference Optimization); GRPO — the algorithm behind DeepSeek-R1
- RLVR (RL with Verifiable Rewards): how reasoning models (o1/o3, DeepSeek-R1) are trained
- Test-time compute in practice: "thinking tokens," inference-time scaling (cashing out the
  preview from Week 1)
- Evaluation revisited: LLM-as-judge, benchmark contamination and saturation — how do we know a
  reasoning model is actually better?

**Readings:**

- Ross, Gordon, Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (DAgger, 2011)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO, 2017) - Explained: [Preference Tuning LLMs with DPO](https://huggingface.co/blog/pref-tuning)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO, 2023)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025), with Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024) for the GRPO mechanics

**See also (optional, background):**

- Schulman et al., "Trust Region Policy Optimization" (TRPO, 2015) — background for PPO's clipping
- Christiano et al., "Deep Reinforcement Learning from Human Preferences" (RLHF, 2017)
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (2023) — LLM-as-judge, moved out of required reading this year
- Zheng et al., [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) (GSPO, 2025) — direct fix to GRPO's instability; trains Qwen3
- "DAPO: An Open-Source LLM Reinforcement Learning System at Scale" (2025) — similar ground to GSPO, more ablation-heavy

**Key Concepts:**

- Imitation learning, distribution shift, "why RL"
- Policy gradients, advantage functions
- Trust regions, clipping, Alignment tax
- DPO, PPO, GRPO
- RLVR; test-time/inference-time compute scaling
- LLM-as-judge; benchmark contamination and saturation

---

### Week 9: Reinforcement Learning - Deep Q-Networks

Q-learning asks a basic RL question with a neural net standing in for the lookup table: can an
agent learn a value for every action in every state purely from trial-and-error reward, and act
greedily on it? Self-play pushes the same trial-and-error idea further — no human
demonstrations, no labeled data, just an agent improving against itself. That mechanism isn't
only for games: Week 8's reasoning models already train on verifiable rewards, and Absolute
Zero closes the loop, using self-play to generate a model's own training curriculum with zero
external data at all.

**Tuesday: Q-Learning with Neural Networks**

- Deep Q-Networks (DQN): approximating the Q-function with a neural net, experience replay,
  target networks
- Double DQN: overestimation bias in Q-learning's max operator, and the two-network fix
- Brief mention only: classic game-playing systems built on this lineage (AlphaGo, AlphaZero,
  MuZero, Monte Carlo Tree Search) — optional background, see readings below

**Thursday: Beyond Value-Based RL — Maximum Entropy and Self-Play**

- Soft Actor-Critic (SAC): maximum-entropy RL, continuous control, off-policy actor-critic
- Self-play as a general RL mechanism, not just a game-playing trick
- Absolute Zero: self-play RL for LLM reasoning with zero external data — closes the loop back
  to Week 8's RLVR content
- Brief mention only: Cicero (Diplomacy: self-play plus natural-language negotiation) —
  optional background, see readings below

**Readings:**

- Mnih et al., [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (2013); journal version: ["Human-level control through deep reinforcement learning"](https://doi.org/10.1038/nature14236) (DQN, Nature 2015)
- Van Hasselt et al., [Double DQN](https://arxiv.org/abs/1509.06461) (2016) - Explained: [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course)
- Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (SAC, 2018) - Explained: [Soft Actor-Critic - Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- Zhao et al., [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) (2025)

**See also (optional, background):**

- Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning" (2017)
- Silver et al., "Mastering the Game of Go without Human Knowledge" (AlphaZero, 2017)
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2019)
- FAIR/Meta AI, ["Human-level play in the game of Diplomacy by combining language models with strategic reasoning"](https://www.science.org/doi/10.1126/science.ade9097) (Cicero, Science 2022)

**Key Concepts:**

- Q-learning with neural networks
- Maximum-entropy RL; continuous control
- Self-play as a general RL mechanism — for games, and (Absolute Zero) for LLM reasoning with
  no external data

---

### Week 10: Speech and Audio Processing

Speech recognition is "just" sequence-to-sequence transduction, but Whisper's real lesson was
that weak supervision at massive scale beat carefully-curated, strongly-supervised pipelines — a
pattern that recurs throughout this course (more noisy data + scale often beats cleverness).
Aside: by 2026 several open models (Qwen3-ASR, NVIDIA Canary) have overtaken Whisper on raw
accuracy leaderboards, but the architectural lesson taught here — weak supervision at scale — is
the point, not who currently tops a leaderboard.

**Tuesday: Speech Models**

- Speech recognition fundamentals: Mel-spectrograms and MFCCs
- Whisper architecture
- Conformers: combining convolutions and attention
- Brief mention only: WaveNet/dilated convolutions (legacy architecture) and full-duplex
  speech-to-speech systems (frontier aside, not a core topic)

**Thursday:** No new topic assigned (open/flex slot — see Schedule for how it's used).

**Readings:**

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022)
- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)

**Key Concepts:**

- Conformer architecture

---

### Week 11: LLMs for Research; Agentic AI — Orchestration & Evaluation

If an AI can read papers and write code, can it also do science — generate hypotheses, run
experiments, and write them up? Deeper question: as AI-assisted (or AI-generated) papers flood
the literature, how do we evaluate contributions when authorship and originality themselves
become harder to pin down? This closes the loop on the course's evaluation theme (Weeks 1, 8)
at the level of science itself.

On the agentic side, the deeper question shifts once one agent can call tools: who decides what
to delegate to which sub-agent, and how do you know the resulting swarm actually did the right
thing?

**Tuesday: AI for Scientific Research**

- AI for reviewing papers
- AI-generated scientific papers

**Thursday: Agentic AI — Orchestration, Harnesses, and Evaluation** (a current-state-of-the-art
follow-up to Week 4's RAG/MCP/agents material)

- Multi-agent orchestration: subagent/worker delegation and coordination, vs. single-agent
  ReAct-style loops
- Harness design: context management across long sessions, sandboxing and tool permissions,
  human-in-the-loop approval flows
- Coding agents as a case study (e.g. Claude Code, Cursor, Devin-style architectures) —
  code as a shared, inspectable state for multi-agent coordination
- Current agent evaluation: GAIA, SWE-bench (Verified), τ-bench, OSWorld, WebArena, METR
  time-horizon benchmarks

**Readings:**

- Lu et al., "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (2024)
- Kusumegi et al., "Scientific production in the era of large language models" (Science, 2025)
- Mialon et al., [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983) (2023)
- Yao et al., [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) (2024)
- Jimenez et al., [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770) (2023/ICLR 2024)
- Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering" (NeurIPS 2024) — the interface/harness around an agent matters as much as the model itself, with real ablations proving it
- Anthropic, ["Effective harnesses for long-running agents"](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) (Nov 2025) — direct engineering follow-up on harness design (context management, sandboxing, assumptions going stale as models improve); optional background at Week 4, assigned in full here

**Key Concepts:**

- Agent harnesses vs. models; multi-agent orchestration and delegation
- Agent evaluation and benchmarking

---

### Week 12: Video Understanding and Generation; Course Wrap-up

Video generation is diffusion plus one more axis of complexity: time. Deeper question: how do
you keep a scene physically and character-consistent across hundreds of frames when you're
generating pixels, not simulating physics? Aside: Sora itself has already been superseded (Sora
2, then deprecated in 2026); current leaders (Veo 3.1, Seedance 2.0) are judged as much on
audio/dialogue sync as visual fidelity — the frontier has moved from "can it generate video" to
"can it generate a consistent audiovisual scene." Closing the course here: this is the last new
technical content before presentations, followed by a recap of the course's major themes.

**Tuesday: Video Understanding and Generation** (merged into one session — the diffusion
fundamentals were already covered in Week 3, so this is an application/extension, not a new
foundation)

- Large Vision Models (LVM); video tokenization with VQGAN
- Stable Video Diffusion; latent diffusion for video
- Diffusion Transformers (DiT), spacetime patches (Sora); character consistency
- Data curation for video (cuts, fades, optical flow); frame interpolation

**Course Wrap-up**: recap of major course themes (architecture/data/loss choices, evaluation,
scale) — see Key Takeaways below.

**Readings:**

- Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (2023)
- OpenAI, [Sora: Video Generation Models as World Simulators](https://openai.com/research/video-generation-models-as-world-simulators) (2024) — historical: still the clearest explanation of DiT/spacetime patches, even though the product has moved on

**Key Concepts:**

- Diffusion transformers for video; multi-stage training pipelines

---

## Key Takeaways

- **Build in inductive bias**: invariances, architectural choices (U-Nets; early/late fusion; causal models)
- **Loss function and regularization matter**: SGD as regularization; "keep policy close" penalties in RL
- **Training data quality is crucial**: Use LLMs to curate or generate better data
- **Invert processes to learn**: e.g., denoising for generation
- **Pretrain; then tune**: RLHF, SFT/LORA, ...
