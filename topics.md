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

### Week 0: Course Introduction & Deep Learning Review

This week sets the shared vocabulary and mental models — architectures, losses, optimizers,
regularization — that every later week assumes. Deeper question: what are the basic building
blocks of a neural network, and why do these particular choices (not others) tend to work?

**Topics:**

- Course overview
- How to read a paper
- Deep learning review
  - Model architectures, loss functions, optimization
  - Core concepts: ReLU/SwigLU, CNN, RNN/LSTM, Transformers
  - Regularization: L1/L2, dropout, early stopping
  - Optimization: SGD, minibatch, Adam, Adagrad
  - Learning paradigms: supervised, unsupervised, semi-supervised, reinforcement

**Readings:**

- [Role-Playing Paper-Reading Seminars](https://colinraffel.com/blog/role-playing-paper-reading-seminars.html)

**Key Concepts:**

- Architecture, training data, and loss function choices, inductive bias, and costs
- How to read a paper

---

### Week 1: NLP and Attention Mechanisms

The transformer solved a concrete engineering problem — let every token attend to every other
token, in parallel, without the sequential bottleneck of RNNs — and that one architectural
choice now underlies almost everything else in this course. Deeper question: how much of what a
model can do comes from architecture, vs. simply from how much data and compute you throw at it?

**Tuesday: Transformers**

- Tokenization: WordPiece, Byte Pair Encoding (BPE)
- Encoder vs decoder architectures
- Masking strategies
- Positional encoding methods
- **Scaling laws — two axes:**
  - *Training-time*: model size, data size, and loss (Kaplan) — and compute-optimal allocation
    between the two for a fixed training budget (Chinchilla)
  - *Inference-time*: spending more compute per query at test time, rather than more compute at
    training time — previewed here; covered in depth with reasoning models in Week 8

**Thursday: Attention Mechanisms**

- Attention for machine translation, Q&A, speech recognition
- Self-attention (Query, Key, Value)
- Multi-head attention
- How to handle long context windows? RoPE-scaling methods (YaRN, LongRoPE2) have pushed
  production context windows past 1M tokens (Gemini, Llama 4 Scout) — but by 2026 the field is
  shifting focus from "how long a window" to "how well the model uses what's in it" (recall
  Thursday's "Lost in the Middle" reading) and toward inference-time compute instead

**Ongoing Theme: Evaluation** (introduced here; revisited for reasoning models in Week 8 and
for agents in Week 11)

- How do we know a model is actually good? Benchmarks and held-out test sets, perplexity vs.
  downstream task performance
- Why this matters more as models get harder to evaluate by inspection (scale, reasoning, agency)

**Readings:**

- Vaswani et al., "Attention Is All You Need" (2017)
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)

**Key Concepts:**

- Encoder-decoder architectures
- Transformers, Self-attention, Multi-head attention
- Context windows and embedding dimensions
- Scaling laws: training-time (data/model size) vs. inference-time (compute per query)
- Evaluation as a recurring theme

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
embedding space? Aside: the current image-generation frontier (FLUX.2, Stable Diffusion 3.5) is
still built on the same latent-diffusion + text-conditioning recipe taught here — the advances
since 2022 are mostly in fidelity and control, not a new paradigm.

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

- Radford et al., [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021)
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (Stable Diffusion, 2022); Explained: [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)

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

**Tuesday: RAG (Retrieval-Augmented Generation)**

- Architecture: retriever + generator
- Vector databases and document chunking
- Limitations and alternatives to RAG
- Memory networks

**Thursday: MCP, Tool Use, Harnesses, and Agents**

- MCP learning to use APIs (Toolformer)
- Self-supervised training for tool use
- Kani, langchain, hooks, skills and agent frameworks
- Design choices: knowledge location, control flow
- Harnesses, introduced: what surrounds the model (tool permissions, context management) —
  the basic vocabulary that Week 11's orchestration/evaluation session builds on

**Readings:**

- Schick et al., [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (2023); Explained: [How does AI learn to use tools? - Toolformer explained](https://www.youtube.com/watch?v=hI2BY7yl_Ac)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, 2020)
- Anthropic, [Model Context Protocol specification](https://modelcontextprotocol.io) (2024) — the actual MCP spec this topic is named for
- Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023) — the origin of the reason+act loop every harness wraps around
- Anthropic, ["Building Effective Agents"](https://www.anthropic.com/engineering/building-effective-agents) (Dec 2024) — the canonical workflows-vs-agents framework

**Key Concepts:**

- API integration with LLMs via MCP
- Agentic AI systems; harnesses vs. models
- Controlling what goes into the context

---

### Week 5: Distillation; Training Efficiency at Scale

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

**Tuesday: Knowledge Distillation; Training Efficiency at Scale**

- Distillation: teacher-student models
- Soft softmax with temperature
- Distilling Step-by-Step
- Reduced precision models (Deepseek example)
- DeepSeek-R1-Distill: distilling reasoning traces (not just outputs) into small dense models
- Training efficiency: tensor/model parallelism (Megatron-LM) vs. memory-efficient sharding of
  optimizer state and gradients (ZeRO/FSDP) — training-time tractability, distillation's
  deployment-time counterpart

**Readings:**

- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015); Explained: [Distilling the Knowledge in a Neural Network](https://www.youtube.com/watch?v=EK61htlw8hY)
- Shoeybi et al., [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) (2019)
- Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (2019)

**Key Concepts:**

- Distillation
- Reduced precision models
- Distilling reasoning traces vs. distilling outputs
- Tensor/model parallelism; memory-efficient optimizer/gradient sharding (ZeRO/FSDP)

---

### Week 6: Mechanistic Interpretability and Steering

Once a model can do something, how do you make it do a *specific* thing reliably — and how do
you know what's happening inside when you try? Deeper question: how should AI be steered — by
changing its weights (fine-tuning), by changing its input (prompting/in-context learning), or by
directly editing its internal activations?

**Tuesday, first half: Probing**

- Linear probes
- The linear representation hypothesis: why concepts tend to be encoded as directions in
  activation space — the theoretical basis both probing and steering rely on
- Sparse autoencoders, and the Anthropic case: from "Towards Monosemanticity" to "Scaling
  Monosemanticity" — sparse probes finding interpretable features (golden gate bridge)

**Tuesday, second half + Thursday: Steering — In-Context Learning, Activation Steering, and LoRA Fine-Tuning**

- In-context learning / prompting: adapt behavior via the input, no weight or activation changes
- Activation steering: adapt behavior by directly editing internal activations (activation
  addition — prompt-pair activation differences)
- LoRA fine-tuning: adapt behavior by updating (a low-rank slice of) the weights
- Comparing the three: how AxBench benchmarks steering vs. fine-tuning vs. in-context learning

**Readings:**

- Alain & Bengio, [Understanding Intermediate Layers Using Linear Classifier Probes](https://arxiv.org/abs/1610.01644) (2016) — the classic linear-probing paper
- Anthropic, ["A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) (2021) — the linear representation hypothesis; paired with [Neel Nanda's walkthrough video](https://www.youtube.com/watch?v=KV5gbOmHbjU)
- Anthropic, ["Towards Monosemanticity: Decomposing Language Models with Dictionary Learning"](https://transformer-circuits.pub/2023/monosemantic-features) (2023) — sparse autoencoders
- Dai et al., [Why Can GPT Learn In-Context?](https://arxiv.org/abs/2212.10559) (2023) — the in-context-learning method
- Turner et al., [Activation Addition: Steering Language Models Without Optimization](https://arxiv.org/abs/2308.10248) (2023) — the core activation-steering paper
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — the fine-tuning method

**Key Concepts:**

- Probes; linear representation hypothesis; sparse autoencoders
- Three ways to adapt model behavior: in-context learning/prompting, activation steering, fine-tuning

---

### Week 7: Efficient Architectures: MoE and Quantization

As models get too big to run cheaply, two questions emerge: how do you activate only *part* of
a giant model per token instead of all of it (MoE), and how do you represent its weights with
fewer bits without breaking it (quantization)? Cool application: DeepSeek-V3's combination of
MoE and low-precision training is a big part of why it could be trained for a fraction of the
compute of comparably capable dense models.

**Tuesday: Mixture of Experts**

- Mixture of Experts (MoE) architectures
- Expert routing and load balancing
- Llama-3, 4 architecture: MoE, Grouped-query attention, RoPE
- GLaM, Switch Transformers
- DeepSeek-V3: MoE + multi-head latent attention + auxiliary-loss-free load balancing, at 671B
  total / 37B activated parameters

**Thursday: Quantization**

- Post-training quantization (PTQ) vs. quantization-aware training (QAT)
- GPTQ, LLM.int8() — 8-bit/4-bit weight and activation quantization
- QLoRA: quantization + LoRA for efficient fine-tuning

**Readings:**

- Fedus et al., [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/pdf/2209.01667.pdf) (2022); Explained: [Sparse Expert Models (Switch Transformers, GLaM, and more)](https://www.youtube.com/watch?v=U5mhpKkOzKs)
- Frantar et al., [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) (2022/ICLR 2023)
- Dettmers et al., [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) (2022)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

**Key Concepts:**

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

- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO, 2017) - Explained: [Preference Tuning LLMs with DPO](https://huggingface.co/blog/pref-tuning)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO, 2023)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
- Zheng et al., [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) (2023)

**Key Concepts:**

- Policy gradients, advantage functions
- Trust regions, clipping, Alignment tax
- DPO, PPO, GRPO
- RLVR; test-time/inference-time compute scaling
- LLM-as-judge; benchmark contamination and saturation

---

### Week 9: Reinforcement Learning - Deep Q-Networks

Game-playing AI is a clean testbed for a bigger question: can an agent learn good strategy
purely from trial-and-error and self-play, with no human demonstrations at all? Cicero pushed
this further, combining strategic planning with natural-language negotiation — the first system
to reach human-level play in a game that's fundamentally about talking, not just moving pieces.

**Tuesday: Q-learning; Decision Transformers**

- Soft Actor-Critic (SAC)
- Maximum entropy RL
- Conservative Q-learning

**Thursday: Game Playing AI**

- Deep Q-Networks (DQN)
- Double DQN
- Experience replay
- AlphaGo, AlphaZero, MuZero
- Monte Carlo Tree Search (MCTS)
- Cicero: Diplomacy with language and strategy

**Readings:**

- Haarnoja et al., [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (SAC, 2018) - Explained: [Soft Actor-Critic - Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- Van Hasselt et al., [Double DQN](https://arxiv.org/abs/1509.06461) (2016) - Explained: [HuggingFace Deep RL Course](https://huggingface.co/learn/deep-rl-course)
- FAIR/Meta AI, ["Human-level play in the game of Diplomacy by combining language models with strategic reasoning"](https://www.science.org/doi/10.1126/science.ade9097) (Cicero, Science 2022)

**Key Concepts:**

- Q-learning with neural networks
- Self-play training
- Combining language models with strategic reasoning

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
- Anthropic, ["Effective harnesses for long-running agents"](https://www.anthropic.com/engineering) (Nov 2025) — direct engineering follow-up on harness design (context management, sandboxing, assumptions going stale as models improve)

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
