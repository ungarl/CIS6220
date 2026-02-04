---
layout: default
title: Schedule
---

## Course Schedule

This schedule is tentative and may be adjusted based on class progress and interests. Required readings should be completed before the listed class session.

---

### Week 1: Introduction & Foundations

**Tuesday** — Course Overview & Why Scale Matters
- Course logistics and expectations
- The case for scale in deep learning

**Thursday** — Scaling Laws
- *Required*: [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
- *Required*: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022) — "Chinchilla"

---

### Week 2: Hardware & Systems Foundations

**Tuesday** — GPU Architecture and Programming Model
- *Required*: [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
- Key concepts: Memory hierarchy, tensor cores, throughput vs latency

**Thursday** — Distributed Computing Primitives
- *Required*: [Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) (NCCL Documentation)
- Key concepts: AllReduce, AllGather, ReduceScatter, ring topology

---

### Week 3: Data Parallelism

**Tuesday** — Distributed Data Parallel Training
- *Required*: [PyTorch Distributed](https://arxiv.org/abs/2006.15704) (Li et al., 2020)
- Key concepts: Gradient synchronization, bucket fusion

**Thursday** — Large Batch Training
- *Required*: [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677) (Goyal et al., 2017)
- *Required*: [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888) (You et al., 2017) — LARS

---

### Week 4: Model Parallelism

**Tuesday** — Tensor Parallelism
- *Required*: [Megatron-LM](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019)
- Key concepts: Column/row parallel linear layers, attention parallelism

**Thursday** — Pipeline Parallelism
- *Required*: [GPipe](https://arxiv.org/abs/1811.06965) (Huang et al., 2019)
- *Required*: [PipeDream](https://arxiv.org/abs/1806.03377) (Narayanan et al., 2019)
- Key concepts: Micro-batching, pipeline bubbles, 1F1B schedule

---

### Week 5: Memory Optimization

**Tuesday** — Activation Checkpointing & Memory-Efficient Training
- *Required*: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (Chen et al., 2016)
- Key concepts: Recomputation vs storage tradeoffs

**Thursday** — ZeRO and Fully Sharded Data Parallel
- *Required*: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2020)
- Key concepts: Optimizer state sharding, gradient sharding, parameter sharding

---

### Week 6: Mixed Precision & Quantization

**Tuesday** — Mixed Precision Training
- *Required*: [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (Micikevicius et al., 2017)
- Key concepts: FP16, BF16, loss scaling, master weights

**Thursday** — Quantization-Aware Training
- *Required*: [LLM.int8()](https://arxiv.org/abs/2208.07339) (Dettmers et al., 2022)
- *Required*: [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)

---

### Week 7: Transformer Architecture Deep Dive

**Tuesday** — Attention Mechanisms
- *Required*: [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- *Required*: [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)

**Thursday** — Efficient Transformers
- *Required*: [FlashAttention-2](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- *Recommended*: [GQA: Grouped Query Attention](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)

---

### Week 8: Mixture of Experts

**Tuesday** — MoE Fundamentals
- *Required*: [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)
- *Required*: [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2022)

**Thursday** — Modern MoE Systems
- *Required*: [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (Jiang et al., 2024)
- Key concepts: Expert routing, load balancing, capacity factors

---

### Week 9: Training Infrastructure

**Tuesday** — Fault Tolerance and Checkpointing
- *Required*: [Check-N-Run](https://www.usenix.org/conference/nsdi24/presentation/wang-zhuang) (Wang et al., 2024)
- Key concepts: Asynchronous checkpointing, recovery strategies

**Thursday** — Cluster Scheduling and Resource Management
- *Required*: [Pollux](https://www.usenix.org/conference/osdi21/presentation/qiao) (Qiao et al., 2021)
- Key concepts: Elastic training, goodput optimization

---

### Week 10: Optimization Techniques

**Tuesday** — Optimizers for Large-Scale Training
- *Required*: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (Loshchilov & Hutter, 2017) — AdamW
- *Recommended*: [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (Muennighoff et al., 2023)

**Thursday** — Learning Rate Schedules and Warmup
- *Required*: [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) (Liu et al., 2019) — RAdam
- Key concepts: Linear warmup, cosine decay, restarts

---

### Week 11: Pre-training at Scale

**Tuesday** — Language Model Pre-training
- *Required*: [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020)
- *Required*: [LLaMA](https://arxiv.org/abs/2302.13971) (Touvron et al., 2023)

**Thursday** — Data Curation and Quality
- *Required*: [The Pile](https://arxiv.org/abs/2101.00027) (Gao et al., 2020)
- *Required*: [Dolma](https://arxiv.org/abs/2402.00159) (Soldaini et al., 2024)

---

### Week 12: Post-Training

**Tuesday** — Instruction Tuning
- *Required*: [FLAN: Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Wei et al., 2021)
- *Required*: [Self-Instruct](https://arxiv.org/abs/2212.10560) (Wang et al., 2022)

**Thursday** — RLHF and Alignment
- *Required*: [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022) — InstructGPT
- *Required*: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) — DPO

---

### Week 13: Inference Optimization

**Tuesday** — Efficient Inference
- *Required*: [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180) (Kwon et al., 2023)
- Key concepts: PagedAttention, KV cache management, continuous batching

**Thursday** — Speculative Decoding and Beyond
- *Required*: [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)
- Key concepts: Draft models, verification, parallel decoding

---

### Week 14: Project Presentations

**Tuesday** — Project Presentations (Group A)

**Thursday** — Project Presentations (Group B)

---

### Week 15: Project Presentations & Wrap-up

**Tuesday** — Project Presentations (Group C)

**Thursday** — Course Wrap-up & Future Directions
- Emerging trends and open problems
- Course feedback and discussion
