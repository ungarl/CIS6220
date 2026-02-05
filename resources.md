---
layout: default
title: Resources
---

## Course Infrastructure

| Platform | Purpose | 
|----------|---------|
| Canvas | Links, grades |
| Google Drive | Homework, readings | 
| Ed Discussion | Q&A and discussions |
| PollEverywhere | Class participation |

All to linked from canvas.

## Getting Help

- **Ed Discussion**: Best for questions that might help others
- **Office Hours**:  TBD
- **Email**: Only for things not related to the course; use Ed for technical questions; do private posts on Ed for personal problems, grading issues, etc.

## Weekly Cadence

- **Mondays:** Homework and post-quiz due at midnight
- **Tuesdays:** Reading questions and pre-quiz due before class; lecture
- **Thursdays:** Reading questions due before class; lecture; next week's readings released

## How to Read Papers

Reading research papers is a skill. Here's an approach that works well:

### First Pass (15-20 minutes)
1. Read the title, abstract, and introduction
2. Look at all figures, tables, and their captions
3. Read the conclusion
4. Skim section headings to understand structure

**Goal**: Understand what problem they're solving and their main contribution

### Second Pass (30 minutes)
1. Read the paper more carefully, but skip any dense math
2. Make note of terms or concepts you don't understand; ask an LLM to explain them.
3. Pay attention to experimental setup and results
4. Look up 2-3 key references (if needed)

**Goal**: Understand their approach and be able to summarize it

### Third Pass (optional, for deep understanding)
1. Work through the math and proofs
2. Think about what assumptions they make
3. Consider what's missing or what you'd do differently
4. Try to mentally "re-implement" their approach

**Goal**: Deep understanding, ability to extend or critique

### Questions to Consider
- What is the broader field?
- What gap does the paper fill?
- Why does it matter?
- What is the novel method/approach/problem?
- What are the results? How strong are they?
- What is the broader significance?

## Additional Resources

### Textbooks and Courses
- [Dive into Deep Learning](https://d2l.ai/) — Interactive deep learning textbook
- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/) — Stanford course on LLM training
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) — Production ML course

### Technical References
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

### Blogs and Articles
- [Lilian Weng's Blog](https://lilianweng.github.io/) — Excellent ML research summaries
- [The Gradient](https://thegradient.pub/) — ML research perspectives
- [Chip Huyen's Blog](https://huyenchip.com/blog/) — ML systems and engineering

### Tools We'll Use
- **PyTorch** — Primary deep learning framework
- **Hugging Face** — Models, datasets, and training utilities

## Paper Reading List by Topic

### Foundational Architectures
- Vaswani et al., "Attention Is All You Need" (2017)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT, 2018)
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
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (YOLO, 2015)
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
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

### Distillation & Mixture of Experts
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes" (2023)
- Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" (2023)
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
- Du et al., "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts" (2021)
- Jiang et al., "Mixtral of Experts" (Mixtral 8x7B, 2024)

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

### In-Context Learning & Theory
- Jacot et al., "Neural Tangent Kernel: Convergence and Generalization in Neural Networks" (NTK, 2018)
- Garg et al., "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (2022)
- Hegselmann et al., "TabLLM: Few-shot Classification of Tabular Data with Large Language Models" (2023)
- Von Oswald et al., "Transformers Learn In-Context by Gradient Descent" (2023)
- Choromanski et al., "Rethinking Attention with Performers" (2020)

### Large Language Models
- Dubey et al., "The Llama 3 Herd of Models" (2024)
- Groeneveld et al., "OLMo: Accelerating the Science of Language Models" (2024)
- Jiang et al., "Mistral 7B" (2023)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)

### Interpretability, Probing, and Steering
- Tenney et al., "BERT Rediscovers the Classical NLP Pipeline" (2019)
- Chen et al., "Probing BERT in Hyperbolic Spaces" (2020)
- Kim et al., "Probing What Different NLP Tasks Teach Machines about Function Word Comprehension" (2018)
- Turner et al., "Interpretable Steering of Large Language Models via Activation Engineering"
- Yan Leng, "A Unified First-Order Framework for Activation" (2025)

### Scientific Paper Writing
- Lu et al., "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (2024)
- Weng et al., "CycleResearcher: Improving Automated Research via Automated Review" (2024)
- Kusumegi et al., "Scientific production in the era of large language models" (Science, 2025)

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

