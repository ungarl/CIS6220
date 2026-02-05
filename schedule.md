---
layout: default
title: Schedule
---

## Course Schedule

This schedule is tentative and may be adjusted based on class progress and interests. Readings should be completed before the listed class session.

---

### Week 0: Course Introduction & Deep Learning Review

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

**Tuesday: Transformers**

- Tokenization: WordPiece, Byte Pair Encoding (BPE)
- Encoder vs decoder architectures
- Masking strategies
- Positional encoding methods
- Scaling laws

**Thursday: Attention Mechanisms**

- Attention for machine translation, Q&A, speech recognition
- Self-attention (Query, Key, Value)
- Multi-head attention
- How to handle long context windows?

**Readings:**

- Vaswani et al., "Attention Is All You Need" (2017)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) (Beltagy et al., 2020)
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023)
- Yun et al., "Are Transformers Universal Approximators of Sequence-to-Sequence Functions?" (2020)
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)

**Key Concepts:**

- Encoder-decoder architectures
- Transformers, Self-attention, Multi-head attention
- Context windows and embedding dimensions
- Scaling laws for neural networks

---

### Week 2: Computer Vision and Segmentation

**Tuesday: Object Detection**

- YOLO (You Only Look Once) - architecture and evolution (v1 to v11)
- Loss functions: focal loss, anchor boxes
- U-Net architecture for medical image segmentation
- Skip connections, transposed convolutions
- Data augmentation strategies

**Thursday: Segmentation**

- Segment Anything Model (SAM, SAM2)
- Memory attention, prompt encoders, mask decoders
- Video Object Segmentation (VOS)
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

### Week 4: Transfer Learning and Fine-Tuning

**Tuesday: Efficient Fine-Tuning**

- Transfer learning
- Feature transferability across layers
- LoRA (Low-Rank Adaptation), QLoRA (quantized LoRA)
- Adapter layers and Prefix tuning

**Thursday: Research Projects**

- How to write effective research papers
- Context-Content-Conclusion (C-C-C) structure
- Project proposals and timeline
- Using AI tools for research

**Readings:**

- Yosinski et al., [How Transferable Are Features in Deep Neural Networks?](https://proceedings.neurips.cc/paper_files/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf) (2014)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- [How to Write a Great Research Paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) (Simon Peyton Jones)
- [Ten Simple Rules for Structuring Papers](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619)

**Key Concepts:**

- Parameter-efficient fine-tuning
- LORA, Low-rank matrix decomposition
- Context-Content-Conclusion

---

### Week 5: RAG, MCP, Skills, and Agentic AI

**Tuesday: RAG (Retrieval-Augmented Generation)**

- Architecture: retriever + generator
- Vector databases and document chunking
- Limitations and alternatives to RAG
- Memory networks

**Thursday: MCP, Tool Use and Agents**

- MCP learning to use APIs (Toolformer)
- Self-supervised training for tool use
- Kani, langchain, hooks, skills and agent frameworks
- Design choices: knowledge location, control flow

**Readings:**

- Schick et al., [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) (2023); Explained: [How does AI learn to use tools? - Toolformer explained](https://www.youtube.com/watch?v=hI2BY7yl_Ac)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, 2020)
- Supplemental: [LangChain Explained](https://www.ibm.com/topics/langchain)

**Key Concepts:**

- API integration with LLMs via MCP
- Agentic AI systems
- Controlling what goes into the context

---

### Week 6: Efficient Architectures

**Tuesday: Knowledge Distillation**

- Distillation: teacher-student models
- Soft softmax with temperature
- Distilling Step-by-Step
- Reduced precision models (Deepseek example)

**Thursday: Mixture of Experts**

- Mixture of Experts (MoE) architectures
- Expert routing and load balancing
- Llama-3, 4 architecture: MoE, Grouped-query attention, RoPE
- GLaM, Switch Transformers

**Readings:**

- Hinton et al., [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015); Explained: [Distilling the Knowledge in a Neural Network](https://www.youtube.com/watch?v=EK61htlw8hY)
- Fedus et al., [A Review of Sparse Expert Models in Deep Learning](https://arxiv.org/pdf/2209.01667.pdf) (2022); Explained: [Sparse Expert Models (Switch Transformers, GLaM, and more)](https://www.youtube.com/watch?v=U5mhpKkOzKs)

**Key Concepts:**

- Distillation
- Reduced precision models
- Mixture of Experts (MoE)/Sparse models

---

### Week 7: In context learning; Mechanistic interpretability

**Tuesday: In-Context Learning and Neural Tangent Kernels**

- In-context (few-shot) learning
- Meta-gradients and linear attention
- ICL vs fine-tuning comparison
- Neural Tangent Kernel (NTK) theory

**Thursday: Mechanistic interpretability and steering**

- Linear probes
- Sparse probes (golden gate bridge)
- Steering by manipulating activation vs by prompting

**Readings:**

- Dai et al., [Why Can GPT Learn In-Context?](https://arxiv.org/abs/2212.10559) (2023); Explained: [In-Context Learning: A Case Study of Simple Function Classes](https://arxiv.org/abs/2208.01066)
- [Neural Tangent Kernels](https://rajatvd.github.io/NTK/) - Video: [Deep Networks Are Kernel Machines (Paper Explained)](https://www.youtube.com/watch?v=ahRPdiCop3E)
- Supplemental: [What learning algorithm is in-context learning?](https://openreview.net/forum?id=0g0X4H8yN4I)
- "Linear Personality Probing and Steering in LLMs: A Big Five Study," 2025
- "Activation Steering in LLMs" (Emergent Mind topic page, 2025).

**Key Concepts:**

- Kernel methods and deep learning
- Probes and steering

---

### Week 8: Reinforcement Learning - Policy Optimization

**Tuesday: RLHF and Policy Gradients**

- Reinforcement learning fundamentals (V, Q, policy)
  - On-policy vs off-policy learning
  - Model-based vs model-free RL
- RLHF (Reinforcement Learning from Human Feedback)
- PPO (Proximal Policy Optimization)
- Actor-critic methods
- Generalized Advantage Estimation (GAE)
- REINFORCE for LLMs

**Thursday: Direct Preference Optimization**

- DPO (Direct Preference Optimization); GRPO
- Imitation learning and behavioral cloning
- Inverse Reinforcement Learning (IRL)
- Behavioral Cloning from Observation (BCO)

**Readings:**

- Review: [Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (PPO, 2017) - Explained: [Preference Tuning LLMs with DPO](https://huggingface.co/blog/pref-tuning)
- Rafailov et al., [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (DPO, 2023)

**Key Concepts:**

- Policy gradients, advantage functions
- Trust regions, clipping, Alignment tax
- DPO, PPO, GRPO

---

### Week 9: Reinforcement Learning - Deep Q-Networks and Games

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
- Supplemental: Silver et al., "Mastering the Game of Go" (AlphaGo, 2016)

**Key Concepts:**

- Q-learning with neural networks
- Self-play training
- Combining language models with strategic reasoning

---

### Week 10: Speech and Audio Processing

**Tuesday: Speech Models**

- Speech recognition fundamentals: Mel-spectrograms and MFCCs
- Whisper architecture
- Conformers: combining convolutions and attention
- Wavenet and dilated convolutions
- Full-duplex Speech-to-speech systems

**Thursday: Break**

**Readings:**

- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022)
- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)

**Key Concepts:**

- Conformer architecture
- Full-duplex audio systems

---

### Week 11: Video Understanding and Generation

**Tuesday: Video understanding**

- Large Vision Models (LVM)
- Video tokenization with VQGAN
- Stable Video Diffusion
- Data curation for video (cuts, fades, optical flow)
- Frame interpolation
- Latent diffusion for video

**Thursday: Video Generation (Sora)**

- Diffusion Transformers (DiT)
- Latent Diffusion Models for video
- Spacetime patches
- CoCa (Contrastive Captioners)
- Recaptioning and language upsampling
- Character consistency in video generation

**Readings:**

- Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (2023)
- Bai et al., "Sequential Modeling Enables Scalable Learning for Large Vision Models" (LVM, 2024)
- OpenAI, [Sora: Video Generation Models as World Simulators](https://openai.com/research/video-generation-models-as-world-simulators) (2024)

**Key Concepts:**

- Sequential modeling for vision
- Vector quantization
- Diffusion transformers for video
- Multi-stage training pipelines

---

### Week 12: LLMs for Research; Course Wrap-up

**Tuesday: AI for Scientific Research**

- AI for reviewing papers
- AI-generated scientific papers
- Course themes recap

**Readings (to be replaced by more recent papers):**

- Lu et al., "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (2024)
- Weng et al., "CycleResearcher: Improving Automated Research via Automated Review" (2024)
- Kusumegi et al., "Scientific production in the era of large language models" (Science, 2025)

---

### Final Classes: Project Presentations

---

## Key Takeaways

- **Build in inductive bias**: invariances, architectural choices (U-Nets; early/late fusion; causal models)
- **Loss function and regularization matter**: SGD as regularization; "keep policy close" penalties in RL
- **Training data quality is crucial**: Use LLMs to curate or generate better data
- **Invert processes to learn**: e.g., denoising for generation
- **Pretrain; then tune**: RLHF, SFT/LORA, ...
