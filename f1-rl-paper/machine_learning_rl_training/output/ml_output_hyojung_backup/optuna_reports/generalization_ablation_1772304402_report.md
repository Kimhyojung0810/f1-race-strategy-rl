# Generalization reward ablation (논문용)

**설계:** 동일 스텝/시드/트랙. baseline vs +deg vs +stint vs +both.

| setting | val_pre | val_post | 개선폭 | test avg_lap | test position | test pit | test return |
|---------|---------|----------|--------|--------------|----------------|----------|-------------|
| baseline | 105.68 | 93.24 | 12.44 | 92.71 | 11.36 | 1.49 | -0.70 |
| plus_deg | 105.68 | 93.51 | 12.17 | 92.77 | 13.25 | 2.02 | -6.12 |
| plus_stint | 105.68 | 93.32 | 12.37 | 92.60 | 12.02 | 1.20 | -0.30 |
| plus_both | 105.68 | 93.57 | 12.11 | 92.49 | 11.75 | 1.37 | -4.62 |

---

*Generated from generalization_ablation_1772304402.csv*