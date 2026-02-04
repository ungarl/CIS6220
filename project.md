---
layout: default
title: Final Project
---

## Overview

The final project is an opportunity to explore a topic in deep learning at scale in depth. 

Projects will be teams of 3 students.

## Timeline

| Date | Milestone | Deliverable |
|------|-----------|-------------|
| Week 4 | Team Formation | Submit team members and initial interests |
| Week 6 | Proposal | 1-page project proposal |
| Week 9 | Midpoint Check-in | meet with TA and/or professor |
| Week 12 | Draft Report |  |
| Week 14-15 | Presentations | TBD |
| Last day of class | Final Report | |

## Project Proposal (Week 6)

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

- TBD

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

## Compute Resources

We'll try to figure this out, but **everything** in deep learning is compute constrained. 

- **Hugging Face**: Free tier access for model hosting

For projects requiring significant compute, discuss with the instructor early.

## Evaluation Criteria


## Past Project Examples

*(This section will be updated with examples from previous offerings)*

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

