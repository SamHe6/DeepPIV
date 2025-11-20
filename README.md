# Welcome to DeepPIV: An integrated deep learning framework for predicting dipeptidyl peptidase-IV inhibitory peptides

Dipeptidyl peptidase-IV inhibitory peptides (DPP-IV-IPs) are promising candidates for the treatment of type 2 diabetes,
yet their identification within large sequence spaces remains a major challenge. Here, we present DeepPIV, a deep learning
framework that integrates multiple sources of peptide information for accurate prediction. The model incorporates a
sequence branch that captures motif-level and contextual patterns through convolutional and Transformer layers, an ESM
branch that transfers knowledge from pretrained protein language models, and a fingerprint branch that encodes chemical
and structural properties from SMILES-derived Morgan and MACCS fingerprints. These heterogeneous representations
are adaptively fused by a gated mechanism to form a unified embedding. Benchmark comparisons show that DeepPIV
outperforms conventional machine learning and recent deep learning baselines across multiple metrics, while ablation
experiments confirm the complementary contributions of individual branches and the importance of gated fusion. Motif
analysis further demonstrates that DeepPIV captures both conserved cores and extended sequence patterns associated
with inhibitory activity, establishing it as an effective and biologically meaningful framework for DPP-IV-IPs prediction
and discovery.

![The workflow of this study](https://github.com/SamHe6/DeepPIV/blob/main/workflow.png)

# Code
We provide the source code and you can find them [Code](https://github.com/SamHe6/DeepPIV/tree/main/Code)

# ESM-2 Protein Language Model
```bash
https://huggingface.co/docs/transformers/en/model_doc/esm
```
