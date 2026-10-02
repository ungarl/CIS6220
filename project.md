---
layout: default
title: Final Project
---

## Overview

The final project is an opportunity to explore a topic in deep learning at scale in depth. 

Projects will be teams of 3 students.

## Timeline

Each project hand-in replaces that week's homework and is due the following **Monday at midnight**.

| Week | Milestone | Deliverable | Due |
|------|-----------|-------------|-----|
| Week 4 | Team Formation | Submit team members and initial interests | (done) |
| Week 6 | Scoping Meeting | 15-minute meeting with your team's TA before writing the proposal | during Week 6 |
| Week 7 | Proposal | 1-page project proposal | Mon Oct 19 |
| Week 9 | Midpoint Check-in | meet with TA and/or professor | during Week 9 |
| Week 12 | Draft Report |  | Mon Nov 23 |
| Week 14 | Presentations | TBD | in class, Dec 1 and 3 |
| After Week 14 | Final Report | | Mon Dec 7 |

## Project TAs

Each team has a TA who will meet with you at the Week 6 scoping meeting and the Week 9 midpoint check-in. Your TA will reach out to schedule.

| TA | Teams |
|----|-------|
| Sashank Desu | 3Heads1GPU, Art of the Art, GUI Force One, Representation AutoEncoder, Zeroed ReLUs |
| Ruichi Zhang | Backpropagandists, CLIP that chat, Gradient Descenters, StipendNeeded, Team Anima |
| Miranda Miao | Cross Talk, Cross-Attendees, Need More Compute, Vision Suspects |

## Team Formation

Submit:

1. **Team name**
2. **Team members**
3. A paragraph or two on what you might do

## Project Proposal

Your proposal should include:

1. **Problem Statement**: What question are you trying to answer?
2. **Motivation**: Why is this important or interesting?
3. **Related Work**: What prior work exists? How does your project differ?
4. **Approach**: What methods will you use?
5. **Evaluation**: How will you measure success?
6. **Resources**: What compute/data do you need?

## Final Report

Your final report should follow the NeurIPS format (8 pages + unlimited references):

1. **Abstract**: Brief summary of your work
2. **Introduction**: Problem motivation and contributions
3. **Related Work**: Position your work in the literature
4. **Methods**: Technical approach in detail
5. **Experiments**: Setup, results, and analysis
6. **Discussion**: Limitations, future work
7. **Conclusion**: Summary of findings

## Presentation

- Two sessions, both in Week 14.
- Per-team time slot and format: TBD, depending on final team count.

## Project Areas

Here are some suggested project directions. You're encouraged to propose your own ideas.

### Training Efficiency
- Implement and compare different parallelism strategies
- Optimize training throughput for a specific architecture
- Investigate learning rate schedules for large-batch training
- Implement and benchmark activation checkpointing strategies

### Memory Optimization
- Implement FSDP from scratch and compare with PyTorch
- Explore gradient compression techniques
- Build an automatic memory optimizer for PyTorch models
- Investigate offloading strategies (CPU/NVMe)

### Mixture of Experts
- Implement efficient expert routing algorithms
- Study load balancing in MoE models
- Compare dense vs sparse models at different scales
- Investigate expert specialization patterns

### Efficient Architectures
- Implement FlashAttention variants
- Explore efficient attention mechanisms
- Study architectural modifications for inference speed
- Implement and compare linear attention methods

### Quantization
- Compare training-aware vs post-training quantization
- Implement mixed-precision inference
- Study quantization effects across model scales
- Build a quantization toolkit for specific use cases

### Inference Systems
- Build a simple serving system with continuous batching
- Implement speculative decoding
- Study KV-cache optimization strategies
- Benchmark inference frameworks

### Training Dynamics
- Study loss spikes and instabilities in large models
- Investigate the effect of data ordering
- Analyze gradient statistics during training
- Study feature learning dynamics

### Data and Pre-training
- Build a data curation pipeline
- Study the effect of data quality on model performance
- Implement and evaluate data deduplication methods
- Analyze data mixing strategies

### Post-Training
- Implement RLHF/DPO training pipeline
- Study the effect of instruction tuning data
- Compare alignment techniques
- Build evaluation frameworks for aligned models

### Model Behavior and Blind Spots
- Benchmark LLM/agent estimation of elapsed time and task duration, and test whether explicit timestamps or fine-tuning improve calibration
- Study multi-party dialogue disentanglement: strip speaker tags from scripts/transcripts and measure how well a model reconstructs who said what
- Audit what latent assumptions (e.g., speaker age, gender) a model makes from unlabeled text, and whether/how that biases its responses
- Probe for spatial reasoning failures analogous to temporal ones, and compare text-only vs. vision-language models on the same tasks

## Compute Resources

We'll try to figure this out, but **everything** in deep learning is compute constrained. 

- **Hugging Face**: Free tier access for model hosting

For projects requiring significant compute, discuss with the instructor early.

## Evaluation Criteria


## Past Project Examples

Projects from previous offerings of this course (as CIS 6200):

| Team | Project |
|---|---|
| Automatic AGI | GRPO fine-tuning of DeepSeek-R1-Distilled-Qwen-7B for strategic decision-making |
| GLY | Concept-bottleneck models for disease prediction / clinical trustworthiness |
| (H&E)llo World | Pathology whole-slide image classification via self-distillation (DINOv2) and multi-instance learning (ABMIL), few-shot |
| Multimodal Safety | RL-trained multimodal safety judge/detector |
| Team Sigmoid | LLM political bias (liberal vs. conservative axis) |
| The Adversaries | Adversarial robustness benchmarking (GCG attacks, CoT vs. no-CoT) |
| SciVLMapper | Vision-language model decompiling scientific document images back to LaTeX |
| Sparkling Fish | NuClass — pathology cell segmentation/classification via text-image embeddings |
| STSG | LASER — neuro-symbolic spatio-temporal scene graph generation from video |
| Style Transfer | Text style transfer / steering vectors (formal vs. informal) |
| Audiojack | Audio/sound analysis with deep learning |
| GPT is All You Need | Multi-agent RL (HAPPO-BNN) vs. GAT/RAG comparison |
| Power Puff | Code-switching behavior in multilingual LLMs (Qwen), compared to human bilingual code-switching |
| Team Shadowcase | Automated prompt engineering for image generation; latent-space direction analysis (time-of-day/season) |
| Team Transformers | MoE routing — entropy-based adaptive gating vs. Top-p routing |

Strong projects typically:
- Start with a clear, focused question
- Build incrementally on existing work
- Include thorough experiments with proper baselines
- Discuss limitations clearly
- Provide reproducible code

## FAQ

**Can I use my research project?**
Yes, but it must be clearly related to course topics and you must disclose any overlap with other work or courses.

**What if I don't have access to large compute?**
Many interesting projects can be done at small to medium scale. Focus on careful experiments and analysis rather than raw scale.

**Can I work on a project related to my company/internship?**
Yes, but ensure you have appropriate permissions and the project can be shared publicly.

