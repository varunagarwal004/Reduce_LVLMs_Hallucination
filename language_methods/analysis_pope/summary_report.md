# Evaluation Results Summary Report

**Task Type:** object_detection
**Number of Result Files:** 6

## Basic Accuracy Metrics

                  model          strategy  amount  accuracy  count
           DINO-X_LLaVA          combined     all     0.786   1200
           DINO-X_LLaVA            direct     all     0.776   1200
           YOLOv8_LLaVA          combined     all     0.885   2999
           YOLOv8_LLaVA            direct     all     0.776   2999
          gemma-3-4b-it  chain_of_thought results     0.825    200
          gemma-3-4b-it            direct results     0.825    200
gpt-4.1-mini-2025-04-14           initial results     0.820    200
gpt-4.1-mini-2025-04-14 self_verification results     0.780    200
gpt-4.1-mini-2025-04-14            direct results     0.570    200
gpt-4.1-mini-2025-04-14  chain_of_thought results     0.825    200
gpt-4.1-mini-2025-04-14            direct results     0.550    200
          gemma-3-4b-it           initial results     0.835    200
          gemma-3-4b-it self_verification results     0.835    200
          gemma-3-4b-it            direct results     0.825    200
        llava-1.5-7b-hf           initial results     0.820    200
        llava-1.5-7b-hf self_verification results     0.690    200
        llava-1.5-7b-hf            direct results     0.835    200
        llava-1.5-7b-hf  chain_of_thought results     0.760    200
        llava-1.5-7b-hf            direct results     0.835    200

## Strategy Comparison

                  model  amount  chain_of_thought  combined  direct  initial  self_verification  self_verification_improvement  chain_of_thought_improvement  combined_improvement
           DINO-X_LLaVA     all               NaN     0.786   0.776      NaN                NaN                            NaN                           NaN                 0.010
           YOLOv8_LLaVA     all               NaN     0.885   0.776      NaN                NaN                            NaN                           NaN                 0.109
          gemma-3-4b-it results             0.825       NaN   0.825    0.835              0.835                          0.010                         0.000                   NaN
gpt-4.1-mini-2025-04-14 results             0.825       NaN   0.560    0.820              0.780                          0.220                         0.265                   NaN
        llava-1.5-7b-hf results             0.760       NaN   0.835    0.820              0.690                         -0.145                        -0.075                   NaN

## Statistical Significance Tests

**gemma-3-4b-it_cot_vs_direct:**
- p-value: 1.000000
- Significant: False

**gpt-4.1-mini-2025-04-14_sv_vs_direct:**
- p-value: 0.000000
- Significant: True

**gpt-4.1-mini-2025-04-14_cot_vs_direct:**
- p-value: 0.000000
- Significant: True

**gemma-3-4b-it_sv_vs_direct:**
- p-value: 0.527089
- Significant: False

**llava-1.5-7b-hf_sv_vs_direct:**
- p-value: 0.000006
- Significant: True

**llava-1.5-7b-hf_cot_vs_direct:**
- p-value: 0.013664
- Significant: True

## Hallucination Metrics

                  model          strategy  amount  accuracy  true_positives  false_positives  true_negatives  false_negatives  hallucination_rate  miss_rate  precision  recall  f1_score  count
           DINO-X_LLaVA          combined     all     0.786           554.0            211.0           389.0             46.0            0.351667   0.076667      0.724   0.923  0.812000   1200
           YOLOv8_LLaVA          combined     all     0.885          1950.0            195.0           524.0            330.0            0.271210   0.144737      0.909   0.856  0.882000   2999
          gemma-3-4b-it  chain_of_thought results     0.825             NaN              NaN             NaN              NaN            0.055000   0.120000      0.945   0.880  0.911342    200
gpt-4.1-mini-2025-04-14 self_verification results     0.780             NaN              NaN             NaN              NaN            0.020000   0.100000      0.980   0.900  0.938298    200
gpt-4.1-mini-2025-04-14  chain_of_thought results     0.825             NaN              NaN             NaN              NaN            0.020000   0.155000      0.980   0.845  0.907507    200
          gemma-3-4b-it self_verification results     0.835             NaN              NaN             NaN              NaN            0.045000   0.105000      0.955   0.895  0.924027    200
        llava-1.5-7b-hf self_verification results     0.690             NaN              NaN             NaN              NaN            0.010000   0.115000      0.990   0.885  0.934560    200
        llava-1.5-7b-hf  chain_of_thought results     0.760             NaN              NaN             NaN              NaN            0.035000   0.205000      0.965   0.795  0.871790    200

## Key Findings

### Best Performing Strategies:

- **DINO-X_LLaVA**: combined (0.786 accuracy)
- **YOLOv8_LLaVA**: combined (0.885 accuracy)
- **gemma-3-4b-it**: initial (0.835 accuracy)
- **gpt-4.1-mini-2025-04-14**: chain_of_thought (0.825 accuracy)
- **llava-1.5-7b-hf**: direct (0.835 accuracy)

### Overall Best Performance:
**YOLOv8_LLaVA** with **combined** strategy: 0.885 accuracy

### Strategy Improvements over Direct Approach:

- **self_verification**: +0.028 average improvement
- **chain_of_thought**: +0.063 average improvement

