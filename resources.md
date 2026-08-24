---
layout: default
title: Resources
---

## Course Infrastructure

| Platform | Purpose | 
|----------|---------|
| [Canvas](https://canvas.upenn.edu/courses/1946669) | Assignment submission, grades, links |
| Google Drive | Homework and reading materials (distributed here) | 
| Ed Discussion | Q&A and discussions |
| A+ | Attendance |

Homework and readings go out via Google Drive; assignments are handed in on Canvas. All linked
from Canvas.

## Getting Help

- **Ed Discussion**: Best for questions that might help others
- **Office Hours**:  TBD
- **Email**: Only for things not related to the course; use Ed for technical questions; do private posts on Ed for personal problems, grading issues, etc.

## Weekly Cadence

- **Mondays:** Homework and post-quiz due at midnight
- **Tuesdays:** Reading questions and pre-quiz due before class; lecture
- **Thursdays:** Reading questions due before class; lecture; next week's homework, post-quiz, and readings released

See [Homework](homework.html) for homework guidelines and expectations.

## How to Read Papers

A structured approach to reading papers:

### First Pass (15-20 minutes)
1. Read the title, abstract, and introduction
2. Look at all figures, tables, and their captions
3. Read the conclusion
4. Skim section headings to understand structure

**Goal**: Identify the problem and main contribution

### Second Pass (30 minutes)
1. Read the paper more carefully, but skip any dense math
2. Make note of terms or concepts you don't understand; ask an LLM to explain them.
3. Pay attention to experimental setup and results
4. Look up 2-3 key references (if needed)

**Goal**: Understand the approach well enough to summarize it

### Third Pass (optional, for deep understanding)
1. Work through the math and proofs
2. Think about what assumptions they make
3. Consider what's missing or what you'd do differently
4. Try to mentally "re-implement" their approach

**Goal**: Ability to extend or critique the work

### Questions to Consider
- What is the broader field?
- What gap does the paper fill?
- Why does it matter?
- What is the novel method/approach/problem?
- What are the results? How strong are they?
- What is the broader significance?

## Additional Resources

### Textbooks and Courses
- [Dive into Deep Learning](https://d2l.ai/): interactive textbook
- [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2024/): Stanford, LLM training
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/): production ML

### Technical References
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

### Blogs and Articles
- [Lilian Weng's Blog](https://lilianweng.github.io/): ML research summaries
- [The Gradient](https://thegradient.pub/): ML research articles
- [Chip Huyen's Blog](https://huyenchip.com/blog/): ML systems

### Tools We'll Use
- **PyTorch**: primary framework
- **Hugging Face**: models, datasets, training utilities

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
- Beltagy et al., "Longformer: The Long-Document Transformer" (2020)
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts" (2023)
- "LongRoPE2: Near-Lossless LLM Context Window Scaling" (2025)

### Scaling Laws
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)

### Vision Models
- Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (YOLO, 2015)
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- Kirillov et al., "Segment Anything" (SAM, 2023)
- Ravi et al., "SAM 2: Segment Anything in Images and Videos" (2024)
- "SAM 3: Segment Anything with Concepts" (2025)
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT, 2020)
- He et al., "Masked Autoencoders Are Scalable Vision Learners" (MAE, 2022)

### Multimodal Learning
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
- Lu et al., "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations" (2019)
- Li et al., "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (2023)
- Alayrac et al., "Flamingo: a Visual Language Model for Few-Shot Learning" (2022)
- Yu et al., "CoCa: Contrastive Captioners are Image-Text Foundation Models" (2022)

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

### Fine-Tuning
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

### RAG, MCP & Agents
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (RAG, 2020)
- Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023)
- Anthropic, "Model Context Protocol specification" (2024)
- Graves et al., "Neural Turing Machines" (2014)
- Weston et al., "Memory Networks" (2014)
- "Multi-Agent Collaboration Mechanisms: A Survey of LLMs" (2025)
- See also: Evaluation section below (GAIA, SWE-bench, τ-bench — agent-specific benchmarks)

### Distillation, Mixture of Experts & Quantization
- Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes" (2023)
- Mukherjee et al., "Orca: Progressive Learning from Complex Explanation Traces of GPT-4" (2023)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025) — see also its distilled models (Section 2.3/4)
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017)
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding" (2020)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2021)
- Du et al., "GLaM: Efficient Scaling of Language Models with Mixture-of-Experts" (2021)
- Jiang et al., "Mixtral of Experts" (Mixtral 8x7B, 2024)
- DeepSeek-AI, "DeepSeek-V3 Technical Report" (2024)
- Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (2022/ICLR 2023)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale" (2022)
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

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

### In-Context Learning
- Dai et al., "Why Can GPT Learn In-Context?" (2023)
- Garg et al., "What Can Transformers Learn In-Context? A Case Study of Simple Function Classes" (2022)
- Von Oswald et al., "Transformers Learn In-Context by Gradient Descent" (2023)

### Large Language Models
- Dubey et al., "The Llama 3 Herd of Models" (2024)
- Groeneveld et al., "OLMo: Accelerating the Science of Language Models" (2024)
- Jiang et al., "Mistral 7B" (2023)
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
- "A Survey of Test-Time Compute: From Intuitive Inference to Deliberate Reasoning" (2025)

### Evaluation
- Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
- Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (2024)
- Mialon et al., "GAIA: a benchmark for General AI Assistants" (2023)
- Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" (2023/ICLR 2024)
- Yao et al., "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (2024)

### Interpretability, Probing, and Steering
- Alain & Bengio, "Understanding Intermediate Layers Using Linear Classifier Probes" (2016)
- Tenney et al., "BERT Rediscovers the Classical NLP Pipeline" (2019)
- Anthropic, "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning" (2023)
- Anthropic, "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (2024)
- Turner et al., "Activation Addition: Steering Language Models Without Optimization" (2023)
- Anthropic, "A Mathematical Framework for Transformer Circuits" (2021)
- "AxBench" (2025) — benchmark for comparing steering by prompting, fine-tuning, and activation adjustment
- Supplemental: "Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models" (2025)

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

