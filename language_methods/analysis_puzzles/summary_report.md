# Evaluation Results Summary Report

**Task Type:** visual_puzzles
**Number of Result Files:** 6

## Basic Accuracy Metrics

                  model          strategy  amount  accuracy  count
          gemma-3-4b-it  chain_of_thought results  0.235000    200
          gemma-3-4b-it            direct results  0.245000    200
gpt-4.1-mini-2025-04-14           initial results  0.305000    200
gpt-4.1-mini-2025-04-14 self_verification results  0.075000    200
gpt-4.1-mini-2025-04-14            direct results  0.285000    200
gpt-4.1-mini-2025-04-14  chain_of_thought results  0.320000    200
gpt-4.1-mini-2025-04-14            direct results  0.325000    200
          gemma-3-4b-it           initial results  0.265000    200
          gemma-3-4b-it self_verification results  0.200000    200
          gemma-3-4b-it            direct results  0.265000    200
        llava-1.5-7b-hf           initial results  0.211340    194
        llava-1.5-7b-hf self_verification results  0.164948    194
        llava-1.5-7b-hf            direct results  0.134021    194
        llava-1.5-7b-hf  chain_of_thought results  0.210000    200
        llava-1.5-7b-hf            direct results  0.185000    200

## Strategy Comparison

                  model  amount  chain_of_thought  direct  initial  self_verification  self_verification_improvement  chain_of_thought_improvement
          gemma-3-4b-it results             0.235 0.25500  0.26500           0.200000                      -0.055000                      -0.02000
gpt-4.1-mini-2025-04-14 results             0.320 0.30500  0.30500           0.075000                      -0.230000                       0.01500
        llava-1.5-7b-hf results             0.210 0.15951  0.21134           0.164948                       0.005438                       0.05049

## Statistical Significance Tests

**gemma-3-4b-it_cot_vs_direct:**
- p-value: 0.802587
- Significant: False

**gpt-4.1-mini-2025-04-14_sv_vs_direct:**
- p-value: 0.000000
- Significant: True

**gpt-4.1-mini-2025-04-14_cot_vs_direct:**
- p-value: 0.908073
- Significant: False

**gemma-3-4b-it_sv_vs_direct:**
- p-value: 0.090559
- Significant: False

**llava-1.5-7b-hf_sv_vs_direct:**
- p-value: 0.386476
- Significant: False

**llava-1.5-7b-hf_cot_vs_direct:**
- p-value: 0.483840
- Significant: False

## Hallucination Metrics

                  model          strategy  amount  accuracy  hallucination_rate  miss_rate  precision   recall  f1_score  count
          gemma-3-4b-it  chain_of_thought results  0.235000            0.765000   0.765000   0.235000 0.235000  0.235000    200
gpt-4.1-mini-2025-04-14 self_verification results  0.075000            0.925000   0.925000   0.075000 0.075000  0.075000    200
gpt-4.1-mini-2025-04-14  chain_of_thought results  0.320000            0.680000   0.680000   0.320000 0.320000  0.320000    200
          gemma-3-4b-it self_verification results  0.200000            0.800000   0.800000   0.200000 0.200000  0.200000    200
        llava-1.5-7b-hf self_verification results  0.164948            0.835052   0.835052   0.164948 0.164948  0.164948    194
        llava-1.5-7b-hf  chain_of_thought results  0.210000            0.790000   0.790000   0.210000 0.210000  0.210000    200

## Key Findings

### Best Performing Strategies:

- **gemma-3-4b-it**: initial (0.265 accuracy)
- **gpt-4.1-mini-2025-04-14**: direct (0.325 accuracy)
- **llava-1.5-7b-hf**: initial (0.211 accuracy)

### Overall Best Performance:
**gpt-4.1-mini-2025-04-14** with **direct** strategy: 0.325 accuracy

### Strategy Improvements over Direct Approach:

- **self_verification**: -0.093 average improvement
- **chain_of_thought**: +0.015 average improvement

