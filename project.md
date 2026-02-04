---
layout: default
title: Final Project
---

## Overview

The final project is an opportunity to explore a topic in deep learning at scale in depth. Projects can be:

- **Research projects**: Investigate a novel research question
- **Engineering projects**: Build or optimize a system
- **Reproduction studies**: Reproduce and extend a published result
- **Survey projects**: In-depth analysis of a research area (requires instructor approval)

Projects can be done individually or in teams of 2-3 students.

## Timeline

| Date | Milestone | Deliverable |
|------|-----------|-------------|
| Week 4 | Team Formation | Submit team members and initial interests on Canvas |
| Week 6 | Proposal | 2-page project proposal |
| Week 9 | Midpoint Check-in | 5-minute progress presentation |
| Week 12 | Draft Report | Optional: submit for early feedback |
| Week 14-15 | Presentations | 15-minute presentation + Q&A |
| Finals Week | Final Report | 8-page paper (NeurIPS format) |

## Project Proposal (Week 6)

Your 2-page proposal should include:

1. **Problem Statement**: What question are you trying to answer?
2. **Motivation**: Why is this important or interesting?
3. **Related Work**: What prior work exists? How does your project differ?
4. **Approach**: What methods will you use?
5. **Evaluation**: How will you measure success?
6. **Timeline**: What will you complete each week?
7. **Resources**: What compute/data do you need?

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

- 15 minutes presentation + 5 minutes Q&A
- Cover motivation, approach, key results, and lessons learned
- All team members should present

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

Students will have access to:

- **Penn HPC Cluster**: Available to all students
- **Cloud Credits**: Limited GCP/AWS credits available (request in proposal)
- **Hugging Face**: Free tier access for model hosting

For projects requiring significant compute, discuss with the instructor early.

## Evaluation Criteria

Projects will be evaluated on:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Technical Depth | 30% | Sophistication of methods and implementation |
| Novelty | 20% | Originality of approach or findings |
| Execution | 20% | Quality of experiments and analysis |
| Presentation | 15% | Clarity of writing and presentation |
| Relevtic Scope | 15% | Appropriate scope for team size and time |

## Past Project Examples

*(This section will be updated with examples from previous offerings)*

Strong projects typically:
- Start with a clear, focused question
- Build incrementally on existing work
- Include thorough experiments with proper baselines
- Discuss limitations honestly
- Provide reproducible code

## FAQ

**Can I use my research project?**
Yes, but it must be clearly related to course topics and you must disclose any overlap with other work or courses.

**What if I don't have access to large compute?**
Many interesting projects can be done at small to medium scale. Focus on careful experiments and analysis rather than raw scale.

**Can I work on a project related to my company/internship?**
Yes, but ensure you have appropriate permissions and the project can be shared publicly.

**When should I meet with the instructor?**
Come to office hours to discuss project ideas anytime, but especially before the proposal deadline.
