
# CIS6200: Deep Learning at Scale


**Instructor:** Professor Lyle Ungar

## Course Overview

This course covers the core architectures and training methods used in contemporary deep learning. Topics include transformers, vision, diffusion, multimodal models (VLM), deep reinforcement learning, mechanistic interpretability and cutting-edge research in areas such as speech and video generation, RAG, MCP and agents, and scientific paper writing. Students will read and critically discuss the seminal papers across deep learning and contribute to that literature.

### Prerequisites

- Graduate machine learning: CIS5190, CIS5200 or equivalent

---

## Course Structure

- Two lectures per week (mandatory attendance), weekly homework (pytorch and paper critiques), two midterms, and a group final project

### Grading
| Component | Weight |
|-----------|--------|
| Attendance | 10% |
| Homework | 20% |
| Paper Critiques | 20% |
| Final Project | 30% |
| Midterms | 20% |

## Major Course Themes

- **Attention and Transformers**: The backbone of modern deep learning
- **Vision and Segmentation**: YOLO, U-Net, SAM, diffusion models
- **Multimodal Learning**: CLIP, ViLBERT, cross-modal representations
- **Fine-tuning and Transfer**: LoRA, adapters, efficient adaptation
- **RAG, MCP, and Agentic AI**: Tool use, external memory, Toolformer
- **Reinforcement Learning**: Policy optimization (DPO, PPO), Q-learning
- **Game-Playing AI**: AlphaGo/Zero, decision transformers
- **MoE and Distillation**: Sparse models, knowledge transfer
- **Audio/Video**: Conformers, Sora, speech models
- **Interpretability and Steering**: Gradient methods, probes, steering

Recurring questions will include: "Why this architecture and loss function?", "How was the training data collected, cleaned and augemented?", "What are the contributions of pretraining and fine tuning/post training?", "What is internal to the network; what external; why?", "How does this scale?"


---

## Schedule

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
- Cntroling what goes into the context

---
### Week 6: Efficient Architectures
**Tuesday:  Knowledge Distillation**

- Distillation: teacher-student models
- Soft softmax with temperature
- Distilling Step-by-Step
- Reduced precision models (Deepseek example)

**Thursday: Mixture of Experts**

-  Mixture of Experts (MoE) architectures
-  Expert routing and load balancing
- Llama-3, 4 architecture:  MoE, Grouped-query attention, RoPE
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

- linear probes
- sparse probes (golden gate bridge)
- steering by manipulating activation vs by prompting

**Readings:**

- Dai et al., [Why Can GPT Learn In-Context?](https://arxiv.org/abs/2212.10559) (2023); Explained: [In-Context Learning: A Case Study of Simple Function Classes](https://arxiv.org/abs/2208.01066)
- [Neural Tangent Kernels](https://rajatvd.github.io/NTK/) - Video: [Deep Networks Are Kernel Machines (Paper Explained)](https://www.youtube.com/watch?v=ahRPdiCop3E)
- Supplemental: [What learning algorithm is in-context learning?](https://openreview.net/forum?id=0g0X4H8yN4I)
- “Linear Personality Probing and Steering in LLMs: A Big Five Study,” 2025
- “Activation Steering in LLMs” (Emergent Mind topic page, 2025).

**Key Concepts:**

- Kernel methods and deep learning
- Probes and steering

---

### Week 8: Reinforcement Learning - Policy Optimization
**Tuesday: RLHF and Policy Gradients**

- Reinforcement learning fundamentals (V,Q, policy)
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
**Tuesday: Q-learinging; Decision Transformers**

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

**Readings (to be replaced by more recent papers:**

- Lu et al., "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (2024)
- Weng et al., "CycleResearcher: Improving Automated Research via Automated Review" (2024)
- Kusumegi et al., "Scientific production in the era of large language models" (Science, 2025)



### Final Classes: Project Presentations


---

## Key Takeaways

- **Build in inductive bias**: invariances, architectural choices (U-Nets; early/late fusion; causal models)
- **Loss function and regularization matter**: SGD as regularization; "keep policy close" penalties in RL
- **Training data quality is crucial**: Use LLMs to curate or generate better data
- **Invert processes to learn**: e.g., denoising for generation
- **Pretrain; then tune**: RLHF, SFT/LORA, ...

---

## Final Projects

### Timeline
| Date | Milestone |
|------|-----------|
| Week 4 | Team formation |
| Week 6 | Project proposal |
| Week 8 | Baseline results |
| Week 10 | Project checkpoint |
| Week 13 | Class presentations |
| End of Semester | Final report |

### Possible Project Areas
- NLP, vision, speech, multimodal, MoE
- RAG, MCP, Agents, neurosymbolic reasoning
- Attention mechanisms, stable diffusion, state space models
- RL: Policy methods (DPO, PPO), decision transformers
- Explainability and interpretability
- Distillation and model compression
- Graph neural networks
- Scientific applications (biology, chemistry)

---

## Resources

### Infrastructure
- **Canvas**: Links, grades
- **Google Drive**: Homework, readings
- **Ed Discussion**: Q&A and discussions
- **PollEverywhere**: Class participation


### Weekly Cadence
- **Mondays:** Homework and post-quiz due at midnight
- **Tuesdays:** Reading questions and pre-quiz due before class; lecture
- **Thursdays:** Reading questions due before class; lecture; next week's readings released


### Reading Papers

- What is the broader field?
- What gap does the paper fill?
- Why does it matter?
- What is the novel method/approach?
- What are the results?
- What is the broader significance?

---

*This syllabus is subject to change based on class progress and emerging developments in the field.*

---

## Additional Papers

### Foundational Architectures
- Vaswani et al., "Attention Is All You Need" (2017) - *Cited by 149,753*
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019) - *Cited by 124,626*
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018) - *Cited by 12,412*
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2019)
- He et al., "Deep Residual Learning for Image Recognition" (ResNet, 2016)
- Hochreiter & Schmidhuber, "Long Short-Term Memory" (LSTM, 1997)

### Scaling Laws
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- Smith et al., "ConvNets Match Vision Transformers at Scale" (2023)

### Vision Models
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (YOLO, 2015) - *Cited by 57,000*
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015) - *Cited by 100,000*
- Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2018)
- Kirillov et al., "Segment Anything" (SAM, 2023)
- Ravi et al., "SAM 2: Segment Anything in Images and Videos" (2024)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT, 2020)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2022)

### Multimodal Learning
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
- Lu et al., "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations" (2019)
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (2023)
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (2022)
- Yu et al., "CoCa: Contrastive Captioners are Image-Text Foundation Models" (2022)
- Hessel et al., "Grounded Language Acquisition Through the Eyes and Ears of a Single Child" (2024)

### Generative Models
- Kingma & Welling, "Auto-Encoding Variational Bayes" (VAE, 2013)
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM, 2020)
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion, 2022)
- Ramesh et al., "Hierarchical Text-Conditional Image Generation with CLIP Latents" (DALL-E 2, 2022)
- Peebles & Xie, "Scalable Diffusion Models with Transformers" (DiT, 2023)
- Esser et al., "Taming Transformers for High-Resolution Image Synthesis" (VQGAN, 2021)
- Van den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017)

### Video Generation
- Bai et al., "Sequential Modeling Enables Scalable Learning for Large Vision Models" (LVM, 2024)
- Blattmann et al., "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" (2023)
- OpenAI, "Sora: Video Generation Models as World Simulators" (2024)

### Speech and Audio
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper, 2022)
- Gulati et al., "Conformer: Convolution-augmented Transformer for Speech Recognition" (2020)
- Van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016)
- Tang et al., "SALMONN: Towards Generic Hearing Abilities for Large Language Models" (2024)

### Transfer Learning & Fine-tuning
- Yosinski et al., "How Transferable Are Features in Deep Neural Networks?" (2014)
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (Adapters, 2019)
- Li & Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation" (2021)

### RAG, MCP & Agents
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, 2020)
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- Graves et al., "Neural Turing Machines" (2014)
- Weston et al., "Memory Networks" (2014)

### Reinforcement Learning
- Christiano et al., "Deep Reinforcement Learning from Human Preferences" (RLHF, 2017)
- Schulman et al., "Proximal Policy Optimization Algorithms" (PPO, 2017)
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE, 2018)
- Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (DPO, 2023)
- Ahmadian et al., "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (2024)
- Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019)
- Torabi et al., "Behavioral Cloning from Observation" (BCO, 2018)
- Levine et al., "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems" (2020)
- Ross et al., "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger, 2011)

### Deep Q-Learning & Game Playing
- Mnih et al., "Playing Atari with Deep Reinforcement Learning" (DQN, 2013)
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (Double DQN, 2016)
- Silver et al., "Mastering the Game of Go with Deep Neural Networks and Tree Search" (AlphaGo, 2016)
- Silver et al., "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero, 2017)
- Schrittwieser et al., "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero, 2020)
- Meta AI, "Human-level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning" (Cicero, 2022)
- Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
- Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning" (SAC, 2018)
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (CQL, 2020)

### Distillation & Mixture of Experts
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes" (2023)
- Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" (2023)
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
- Du et al., "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts" (2021)
- Jiang et al., "Mixtral of Experts" (Mixtral 8x7B, 2024)


### In-Context Learning & Theory
- Jacot et al., "Neural Tangent Kernel: Convergence and Generalization in Neural Networks" (NTK, 2018)
- Garg et al., "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (2022)
- Hegselmann et al., "TabLLM: Few-shot Classification of Tabular Data with Large Language Models" (2023)
- Von Oswald et al., "Transformers Learn In-Context by Gradient Descent" (2023)
- Choromanski et al., "Rethinking Attention with Performers" (2020) - *Cited by 1,746*


### Large Language Models
- Dubey et al., "The Llama 3 Herd of Models" (2024)
- Groeneveld et al., "OLMo: Accelerating the Science of Language Models" (2024)
- Jiang et al., "Mistral 7B" (2023)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)



### Interpretability, Probing, and Steering
- Tenney et al., "BERT Rediscovers the Classical NLP Pipeline" (2019)
- Chen et al., "Probing BERT in Hyperbolic Spaces" (2020)
- Kim et al., "Probing What Different NLP Tasks Teach Machines about Function Word Comprehension" (2018)
- Turner et al., “Interpretable Steering of Large Language Models via Activation Engineering”
- Yan Leng, "A Unified First‑Order Framework for Activation" (2025)


### Scientific paper writing

- Lu et al., "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (2024)
- Weng et al., "CycleResearcher: Improving Automated Research via Automated Review" (2024)
- Kusumegi et al., "Scientific production in the era of large language models" (Science, 2025)


Probably not covered this year:

### State Space Models
- Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
- Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (Mamba-2, 2024)
- Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces" (S4, 2022)
- Ma et al., "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation" (2024)

### Scientific Applications
- Jumper et al., "Highly Accurate Protein Structure Prediction with AlphaFold" (2021)
- Lin et al., "Evolutionary-scale Prediction of Atomic-level Protein Structure with a Language Model" (ESMFold, 2023)
- Baek et al., "Accurate Prediction of Protein Structures and Interactions Using a Three-Track Neural Network" (RoseTTAFold, 2021)
- Watson et al., "De novo Design of Protein Structure and Function with RFdiffusion" (2023)
- Abramson et al., "Accurate Structure Prediction of Biomolecular Interactions with AlphaFold 3" (2024)

