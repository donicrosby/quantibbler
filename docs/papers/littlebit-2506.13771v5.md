---
source: https://arxiv.org/html/2506.13771v5
fetched: 2026-10-17
source_sha256: c1d2b5b781e5d3423c6636cd5194a91ed9628fcbddd0e606ec47718606960502
converter: tools/arxiv2md.py (stdlib; math kept as LaTeX alttext)
note: Reference copy for spec citation. Links to code hosts are removed; copyright stays with the authors (arXiv distribution terms).
---

##### Report GitHub Issue

 ×



 Title:




Content selection saved. Describe the issue below:

 Description:




 Submit without GitHub
 Submit in GitHub




 arXiv is now an independent nonprofit!
 Learn more
 ×




 Back to arXiv




 Why HTML?


 Report Issue



 Back to Abstract




 Download PDF






















- Abstract

- 1 Introduction

- 2 Related Works

- 2.1 Binarization and Quantization

- 2.2 Low-Rank Approximation in Quantization

- 3 Methodology

- 3.1 LittleBit Architecture

- Proposition 1

- 3.2 Dual-SVID Initialization

- 3.3 Residual Compensation

- 4 Experiments

- 4.1 Settings

- Evaluation Setup

- Training Details

- Baselines

- 4.2 Main Results

- Superior Perplexity in Sub-1-bit Regime

- Extreme Low-BPW Stability

- Model Generalization & Zero-Shot

- 4.3 Reasoning Performance of Extremely Compressed LLMs

- 5 Analysis

- Memory Footprint Reduction

- KV Cache Compression

- Latency Considerations

- 6 Conclusion

- References

- A Mathematical Analysis and Proofs

- B Ablation Study

- B.1 Residual Compensation

- B.2 SmoothSign versus Straight-Through Estimator

- B.3 Grouped Query Attention

- C Perplexity on C4 and PTB

- D Average Bits Per Weight Calculation

- E Pruning versus SVD

- F Analysis of Generated Samples


 License: CC BY-NC-ND 4.0


arXiv:2506.13771v5 [cs.LG] 05 Feb 2026

**footnotetext: Equal contribution††footnotetext: Corresponding author

# LittleBit: Ultra Low-Bit Quantization

via Latent Factorization

Banseok Lee*Dongkyu Kim*Youngcheon You

  
Youngmin KimSamsung Research

Email: {bs93.lee, dongkyu.k, y01000.you, ym1012.kim}@samsung.com

###### Abstract



The deployment of large language models (LLMs) is frequently hindered by prohibitive memory and computational requirements. While quantization mitigates these bottlenecks, maintaining model fidelity in the sub-1-bit regime remains a persistent challenge. In this paper, we introduce LittleBit, a novel framework for extreme LLM compression. We target quantization rates as low as $0.1$ bits per weight (BPW), achieving a memory reduction of approximately $31\times$ , which effectively compresses Llama2-13B to under $0.9$ GB. We represent weights via low-rank latent matrix factorization and subsequently binarize the resulting factors. To counteract the information loss inherent to such drastic precision reduction, we integrate a multi-scale compensation mechanism that learns importance parameters across row, column, and latent dimensions. Two primary contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for quantization-aware training (QAT) initialization, and Residual Compensation to minimize approximation errors. Extensive experiments confirm the superiority of LittleBit in the sub-1-bit domain; for instance, our method at $0.1$ BPW surpasses the performance of leading techniques operating at $0.7$ BPW on Llama2-7B. We establish a new size-performance trade-off—unlocking a potential $11.6\times$ inference speedup relative to FP16—and render powerful LLMs practical for resource-constrained environments. Our code is available at [redacted code link]


## 1 Introduction

Large language models (LLMs) based on the Transformer architecture [1] have fundamentally transformed natural language processing. However, the scale of these models, often reaching hundreds of billions or trillions of parameters [2, 3], results in prohibitive computational and memory costs. In particular, the high demand for GPU VRAM hinders widespread deployment on consumer or edge devices [4, 5, 6].

Model quantization, which reduces numerical precision, serves as a primary technique to address these bottlenecks [7]. While post-training quantization (PTQ) methods such as GPTQ [8] and AWQ [9] effectively compress models to approximately 4 bits, achieving further compression (e.g., to 1-bit) typically requires quantization-aware training (QAT) [10, 11] to maintain performance. QAT has demonstrated the capability to enable 1-bit compression while preserving model fidelity [12, 13, 14].

Nevertheless, even 1-bit models (e.g., approximately 15.4 GB for 70B parameters [7]) may exceed the capacity of highly resource-constrained devices [15]. This limitation motivates the exploration of sub-1-bit quantization. Although prior work [16] has investigated this area, maintaining performance at extremely low effective bits (e.g., 0.55 bits per weight (BPW)) while ensuring computational efficiency without restrictive hardware dependencies remains challenging [16]. Consequently, it is necessary to develop new techniques that achieve such extreme model compression, aiming for bit rates around 0.1 BPW, while preserving model performance.

*Figure: Figure 1:
Low-bit quantization perplexity on Llama2-13B (WikiText-2). LittleBit surpasses the state-of-the-art sub-1-bit quantization technique. Below 0.5 BPW, where the leading prior method degrades sharply, ours remains robust down to 0.1 BPW.
*

We derive our approach from two observations. First, LLM weight matrices often exhibit pronounced low-rank structure [17, 18]. This observation suggests that factorization methods such as Singular Value Decomposition (SVD) [19] may offer a more stable compression pathway than pruning [20], particularly under extreme compression regimes. For instance, at compression ratios exceeding 50%, SVD-based approaches [21] consistently outperform pruning methods such as Wanda [22] (see Appendix E for details). Second, binarization inherently causes information loss [23]. Recent high-performing 1-bit methods [13, 24, 25] demonstrate that comprehensive scaling over multiple dimensions (e.g., row and column) is crucial for mitigating this loss and stabilizing performance [25].

Building on these insights, we introduce LittleBit, a novel method for extreme sub-1-bit quantization (e.g., 0.1 BPW). In LittleBit, we first represent weights via low-rank factorization ( $\mathbf{W}\approx\mathbf{U}\mathbf{V}^{\top}$ ) and subsequently binarize these factors. To counteract errors from binarization, we leverage a multi-scale compensation mechanism that applies learnable scales across rows, columns, and an additional latent dimension. We complement this architecture with Residual Compensation.

Extensive experiments demonstrate the superiority of LittleBit over STBLLM [16], the leading sub-1-bit technique, on LLMs ranging from 1.3B to 32B parameters. Notably, LittleBit achieves competitive performance on Llama2-13B at 0.1 BPW (Fig. 1), surpassing STBLLM at 0.55 BPW. Furthermore, the 32B LittleBit model at just 0.3 BPW maintains strong performance, substantially outperforming STBLLM and establishing a new state-of-the-art for sub-1-bit quantization.

In summary, our contributions are as follows:

- •

We propose LittleBit, a novel framework unifying latent matrix factorization with multi-scale compensation to enable extreme sub-1-bit quantization. This method achieves effective bits down to 0.1 BPW while preserving model performance, establishing a viable size-performance trade-off for deploying LLMs in resource-constrained environments.

- •

We introduce Dual-SVID for the effective initialization of factorized structures and integrate Residual Compensation to mitigate approximation errors. These techniques collectively stabilize quantization-aware training (QAT) and counteract the information loss inherent in drastic precision reduction, ensuring robust learning dynamics.

- •

We provide extensive empirical validation demonstrating that LittleBit consistently outperforms STBLLM [16], the prior leading sub-1-bit technique, across various model scales. Our results confirm that LittleBit maintains superior fidelity in the deep compression regime, where baseline methods typically suffer severe degradation.

## 2 Related Works

### 2.1 Binarization and Quantization

Network binarization aims for extreme compression and acceleration by constraining weights or activations to $\pm 1$ . This enables efficient bitwise operations [26, 27]. While early studies such as BinaryConnect [28] and BNNs [26] demonstrated feasibility, they encountered substantial accuracy degradation [29]. Researchers partially mitigated this degradation by introducing scaling factors [30] and employing the Straight-Through Estimator (STE) [31] to handle the non-differentiable quantization function during training [32].

The direct application of these early techniques to large language models (LLMs) typically results in severe performance loss [33]. Consequently, LLM-specific quantization strategies have emerged. These are primarily categorized into post-training quantization (PTQ) and quantization-aware training (QAT) [34]. PTQ adapts pre-trained models and achieves competitive results at approximately 4-bit precision with methods such as GPTQ [8] and AWQ [9]. However, these methods generally struggle to maintain performance below 2-bit precision [33, 7]. The sub-1-bit regime presents particular challenges for PTQ methods due to severe information loss [35]. Recent work such as STBLLM [16] demonstrates that incorporating structured binarized compression can extend PTQ into the sub-1-bit regime, but a non-negligible accuracy gap persists.

In contrast, QAT integrates the quantization process directly into the training loop. This enables the model to learn and adapt to the low-precision format [36]. This approach generally yields superior performance compared to PTQ at very low bit-widths. Successful QAT approaches for low-bit scenarios often employ sophisticated scaling mechanisms alongside adaptive training, as evidenced by recent methods [13, 24]. Despite these advancements, consistently achieving high model fidelity when quantizing to less than $1.0$ BPW remains a considerable challenge. This persisting difficulty underscores the need for novel architectures and training strategies specifically designed for such extreme compression regimes.

### 2.2 Low-Rank Approximation in Quantization

Beyond quantization, the inherent parameter redundancy within deep neural networks [35] can be effectively exploited using low-rank approximation methods. This redundancy is particularly evident in LLMs [37, 38]. Techniques such as Singular Value Decomposition (SVD) [19] allow large weight matrices to be represented as products of smaller factors. This offers an alternative compression strategy that may be more resilient than pruning, especially under high compression regimes [20]. Recognizing the potential for complementary benefits, researchers have explored combining low-rank factorization to reduce parameter count with quantization to lower parameter precision [39]. Initial studies applied SVD in various roles within the LLM context. Some approaches used activation statistics to guide decomposition [21], while others integrated SVD principles into initialization or parameter-efficient updates [40, 41, 13]. Further works applied transformations prior to factorization [42] or employed low-rank structures to model quantization errors [43].

*Figure: Figure 2: Comparison of a standard Transformer layer (left) and the LittleBit architecture (right). LittleBit performs linear transformation using parallel Primary and Residual pathways. The Primary path employs binarized factors ( $\mathbf{U}_{\mathrm{sign}},\mathbf{V}_{\mathrm{sign}}$ ) and FP16 scales ( $\mathbf{h},\mathbf{g},\bm{\ell}$ ) on input $\mathbf{X}$ , initialized from $\mathbf{W}$ via Dual-SVID. Simultaneously, the Residual path computes a correction with its own parameters ( $\mathbf{U}_{\mathrm{res,sign}},\mathbf{V}_{\mathrm{res,sign}},\mathbf{h}_{\mathrm{res}},\mathbf{g}_{\mathrm{res}},\bm{\ell}_{\mathrm{res}}$ ) from the approximation residual. Their outputs sum to form $\mathbf{Y}$ , eliminating storage of the effective weight matrices $\widehat{\mathbf{W}}_{\mathrm{pri}}$ and $\widehat{\mathbf{W}}_{\mathrm{res}}$ .*

A more integrated approach involves incorporating low-rank concepts directly within QAT methods. For example, DL-QAT [44] jointly learns a representation amenable to both low-rank approximation and low-precision quantization. However, this method is currently limited to 3-bit quantization. Optimizing structural properties such as rank and numerical precision during training represents a promising path toward effective LLM compression. There is potential for further exploration of even lower precision quantization.

## 3 Methodology

This section details LittleBit. It describes the factorized architecture with multi-scale compensation (Section 3.1), the Dual-SVID initialization for stable QAT (Section 3.2), and Residual Compensation for enhanced fidelity (Section 3.3). These components are optimized via QAT, as detailed in Section 4.

### 3.1 LittleBit Architecture

The LittleBit architecture, depicted in Fig. 2, redesigns linear layers for extreme compression (e.g., 0.1 BPW) by synergistically combining low-rank factorization and binarization. The method leverages the observed low-rank structure in LLM weights [45] to approximate the weight matrix $\mathbf{W}\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ as:

|
 |
 | $\mathbf{W}\approx\mathbf{U}\mathbf{V}^{\top}\quad(\mathbf{U}\in\mathbb{R}^{d_{\mathrm{out}}\times r},\mathbf{V}\in\mathbb{R}^{d_{\mathrm{in}}\times r},r\ll\min(d_{\mathrm{in}},d_{\mathrm{out}}))$
 |
 | (1)

We prioritize this factorization-based approach over other structural compression methods, such as pruning, due to its superior robustness in extreme compression regimes. As the comparative analysis against methods like Wanda demonstrates (Appendix E), SVD-based compression maintains model integrity far more effectively at high compression ratios. This provides a more stable foundation for ultra low-bit quantization.
The resulting factors $\mathbf{U}$ and $\mathbf{V}$ are then binarized:

|
 |
 | $\mathbf{U}_{\mathrm{sign}}=\mathrm{sign}(\mathbf{U}),\quad\mathbf{V}_{\mathrm{sign}}=\mathrm{sign}(\mathbf{V})\in\{-1,+1\}$
 |
 | (2)

To compensate for the significant information loss incurred by binarization [23], LittleBit incorporates learnable FP16 scales. Beyond standard row ( $\mathbf{h}$ ) and column ( $\mathbf{g}$ ) scaling [33, 13, 25], the architecture introduces an additional latent scale ( $\bm{\ell}\in\mathbb{R}^{r}$ ). This latent scale addresses the factorization structure by learning the relative importance of each of the $r$ latent dimensions, which correspond to the columns of the binarized factors $\mathbf{U}_{\mathrm{sign}}$ and $\mathbf{V}_{\mathrm{sign}}$ :

|
 |
 | $\mathbf{h}\in\mathbb{R}^{d_{\mathrm{out}}},\quad\mathbf{g}\in\mathbb{R}^{d_{\mathrm{in}}},\quad\bm{\ell}\in\mathbb{R}^{r}$
 |
 | (3)

The primary effective weight, $\widehat{\mathbf{W}}_{\mathrm{pri}}$ , is implicitly constructed from these components:

|
 |
 | $\widehat{\mathbf{W}}_{\mathrm{pri}}=\mathrm{diag}(\mathbf{h})\mathbf{U}_{\mathrm{sign}}\mathrm{diag}(\bm{\ell})\mathbf{V}_{\mathrm{sign}}^{\top}\mathrm{diag}(\mathbf{g})$
 |
 | (4)

Instead of storing or optimizing $\widehat{\mathbf{W}}_{\mathrm{pri}}$ directly, the model learns latent full-precision factors $\mathbf{U}$ and $\mathbf{V}$ . These are binarized during the forward pass using Eq. 2, along with the FP16 scales $\mathbf{h},\mathbf{g},\bm{\ell}$ . This factorized representation enables efficient computation during the forward pass.

#### Proposition 1

Let $\mathbf{X}\in\mathbb{R}^{\mathrm{seq}\times d_{\mathrm{in}}}$ be the input matrix. The forward computation $\mathbf{Y}=\mathbf{X}{\widehat{\mathbf{W}}_{\mathrm{pri}}}^{\top}$ using the primary approximation (Eq. 4) is efficiently computed as:

|
 |
 | $\mathbf{Y}=((((\mathbf{X}\odot\mathbf{g})\mathbf{V}_{\mathrm{sign}})\odot\bm{\ell})\mathbf{U}_{\mathrm{sign}}^{\top})\odot\mathbf{h}$
 |
 | (5)

where $\odot$ denotes element-wise multiplication with broadcasting.

This decomposition (Eq. 5) replaces a large high-precision General Matrix Multiply (GEMM) operation with two smaller binary matrix multiplications and element-wise scaling operations. This offers computational advantages and reduces memory requirements. The effective bits per weight (BPW) is determined by the storage cost of these learnable parameters relative to the original matrix size (see Appendix D for details).

### 3.2 Dual-SVID Initialization

Naively initializing the highly constrained LittleBit structure can lead to unstable QAT. To circumvent this, we propose Dual-SVID, an SVD-based initialization method designed to provide a starting point where the initial effective weight $\widehat{\mathbf{W}}_{\mathrm{pri},0}$ closely approximates $\mathbf{W}$ . Dual-SVID aims to preserve essential sign and magnitude information from the optimal low-rank SVD factors (obtained from $\mathbf{W}\approx\mathbf{U}^{\prime}{\mathbf{V}^{\prime}}^{\top}$ , then truncated) and map this information to the learnable LittleBit parameters ( $\mathbf{U}_{\mathrm{sign}},\mathbf{V}_{\mathrm{sign}},\mathbf{h},\mathbf{g},\bm{\ell}$ ). The process involves three steps: (1) Obtaining $\mathbf{U}^{\prime},\mathbf{V}^{\prime}$ via a truncated SVD. (2) Setting initial binary factors based on signs (Eq. 6). (3) Decomposing the magnitudes $|\mathbf{U}^{\prime}|\in\mathbb{R}^{d_{\mathrm{out}}\times r}$ and $|\mathbf{V}^{\prime}|\in\mathbb{R}^{d_{\mathrm{in}}\times r}$ . To initialize the scales, a rank-1 SVD (or equivalent rank-1 approximation) is performed on each of these magnitude matrices separately. For $|\mathbf{U}^{\prime}|$ , its best rank-1 approximation can be written as $\mathbf{s}_{U,0}(\bm{\ell}_{u,0})^{\top}$ , where $\mathbf{s}_{U,0}\in\mathbb{R}^{d_{\mathrm{out}}}$ becomes the initial row scale $\mathbf{h}_{0}$ , and $\bm{\ell}_{u,0}\in\mathbb{R}^{r}$ is a component contributing to the latent scale. Similarly, for $|\mathbf{V}^{\prime}|$ , its rank-1 approximation $\mathbf{s}_{V,0}(\bm{\ell}_{v,0})^{\top}$ provides the initial column scale $\mathbf{g}_{0}=\mathbf{s}_{V,0}\in\mathbb{R}^{d_{\mathrm{in}}}$ and another latent scale component $\bm{\ell}_{v,0}\in\mathbb{R}^{r}$ . These steps effectively disentangle multi-dimensional magnitude variations and allow for the estimation of initial row ( $\mathbf{h}_{0}$ ), column ( $\mathbf{g}_{0}$ ), and the final latent scale ( $\bm{\ell}_{0}=\bm{\ell}_{u,0}\odot\bm{\ell}_{v,0}$ by element-wise product), as shown in Eqs. 7 and 8:

|
 |
 | $\displaystyle\mathbf{U}_{\mathrm{sign},0}=\mathrm{sign}(\mathbf{U}^{\prime}),\quad\mathbf{V}_{\mathrm{sign},0}=\mathrm{sign}(\mathbf{V}^{\prime})$
 |
 | (6)

|
 |
 | $\displaystyle|\mathbf{U}^{\prime}|\approx\mathbf{h}_{0}(\bm{\ell}_{u,0})^{\top},\quad|\mathbf{V}^{\prime}|\approx\mathbf{g}_{0}(\bm{\ell}_{v,0})^{\top}$
 |
 | (7)

|
 |
 | $\displaystyle\bm{\ell}_{0}=\bm{\ell}_{u,0}\odot\bm{\ell}_{v,0}$
 |
 | (8)

The resulting initial effective weight $\widehat{\mathbf{W}}_{\mathrm{pri},0}=\mathrm{diag}(\mathbf{h}_{0})\mathbf{U}_{\mathrm{sign},0}\mathrm{diag}(\bm{\ell}_{0})\mathbf{V}_{\mathrm{sign},0}^{\top}\mathrm{diag}(\mathbf{g}_{0})$ preserves key structural information. The learnable parameters are initialized as follows: the latent factors $\mathbf{U}$ and $\mathbf{V}$ are initialized with the factors $\mathbf{U}^{\prime}$ and $\mathbf{V}^{\prime}$ obtained from SVD, respectively, and the scales $\mathbf{h},\mathbf{g},\bm{\ell}$ are initialized to $\mathbf{h}_{0},\mathbf{g}_{0},\bm{\ell}_{0}$ .

### 3.3 Residual Compensation

To further improve fidelity without increasing the primary path’s rank, we introduce Residual Compensation. This technique is motivated by the theoretical insight that a two-stage error correction can be more effective than a single, larger approximation (Appendix A). Crucially, this method does not increase the model’s total parameter budget; instead, the fixed bit budget of a single, higher-rank approximation is strategically reallocated into two lower-rank paths: a primary and a residual path. The term ‘residual’ primarily describes its role during initialization, where the auxiliary path is configured to model the error of the primary approximation. During QAT, both paths are optimized jointly to collectively represent the original weight. The advantage of this dual-path approach over a single path is empirically validated in our ablation studies (Section B.1).

*Figure: Figure 3:
Visualization of Dual-SVID initialized weight components for a selected layer in Llama2-7B (Query Weight, Layer 0). Columns, from left to right, represent effective bits of 0.1, 0.3, 0.55, 0.7, 0.8, and 1.0 BPW. Rows display the primary approximation ( $\widehat{\mathbf{W}}_{\mathrm{pri},0}$ ), the residual approximation ( $\widehat{\mathbf{W}}_{\mathrm{res},0}$ ), and their sum ( $\widehat{\mathbf{W}}_{0}=\widehat{\mathbf{W}}_{\mathrm{pri},0}+\widehat{\mathbf{W}}_{\mathrm{res},0}$ ). The rightmost image shows the corresponding crop of the original weight matrix ( $\mathbf{W}$ ) for reference.
*

To implement this, Residual Compensation employs a parallel auxiliary path that learns an approximation $\widehat{\mathbf{W}}_{\mathrm{res}}$ of the residual error. This auxiliary path mirrors the structure of the LittleBit path:

|
 |
 | $\widehat{\mathbf{W}}_{\mathrm{res}}=\mathrm{diag}(\mathbf{h}_{\mathrm{res}})\mathbf{U}_{\mathrm{res,sign}}\mathrm{diag}(\bm{\ell}_{\mathrm{res}})\mathbf{V}_{\mathrm{res,sign}}^{\top}\mathrm{diag}(\mathbf{g}_{\mathrm{res}})$
 |
 | (9)

The parameters of this auxiliary path ( $\mathbf{U}_{\mathrm{res,sign}},\mathbf{V}_{\mathrm{res,sign}},\mathbf{h}_{\mathrm{res}},\mathbf{g}_{\mathrm{res}},\bm{\ell}_{\mathrm{res}}$ ) are also learnable during QAT. They are initialized using the Dual-SVID strategy applied to the initial residual error $\mathbf{W}_{\mathrm{res},0}=\mathbf{W}-\widehat{\mathbf{W}}_{\mathrm{pri},0}$ . Employing a separate path allows the model to specifically learn and compensate for the potentially distinct characteristics of this residual error. The final effective weight is the sum of both approximations:

|
 |
 | $\widehat{\mathbf{W}}=\widehat{\mathbf{W}}_{\mathrm{pri}}+\widehat{\mathbf{W}}_{\mathrm{res}}$
 |
 | (10)

This targeted error correction is often advantageous for maintaining performance at extreme compression levels, as demonstrated by the ablation studies presented in Section B.1. It is important to note that all learnable parameters from both the primary and residual paths are included when calculating the final BPW (see Appendix D for calculation details).

Figure 3 visualizes the initial state of weight components ( $\widehat{\mathbf{W}}_{\mathrm{pri},0}$ , $\widehat{\mathbf{W}}_{\mathrm{res},0}$ , and their sum $\widehat{\mathbf{W}}_{0}$ ) for a Llama2-7B query weight, derived via Dual-SVID across effective bits from 0.1 to 1.0, compared against the original weight $\mathbf{W}$ (far right). Even at a low 0.1 BPW, the primary approximation $\widehat{\mathbf{W}}_{\mathrm{pri},0}$ (top row, leftmost) captures the dominant low-rank structure of $\mathbf{W}$ , albeit with considerable simplification. As effective bits increase, $\widehat{\mathbf{W}}_{\mathrm{pri},0}$ progressively refines details and more effectively suppresses disruptive approximation noise, allowing underlying weight patterns to emerge with greater clarity.
A key observation is that $\widehat{\mathbf{W}}_{0}$ (bottom row) at a modest 0.3 BPW—benefiting from the initialized residual component $\widehat{\mathbf{W}}_{\mathrm{res},0}$ (middle row)—often presents a more faithful initial representation of $\mathbf{W}$ than $\widehat{\mathbf{W}}_{\mathrm{pri},0}$ alone even at a higher 1.0 BPW.
This underscores that Residual Compensation, even at the initialization stage, is vital for capturing complexities missed by the primary low-rank approximation, thereby providing a more faithful starting point for QAT.

## 4 Experiments

### 4.1 Settings

#### Evaluation Setup

We evaluated LittleBit across diverse LLM families, including Llama [46], Llama2 [47], Llama3 [48], OPT [49], Phi-4 [50], and QwQ [51]. These models span parameter scales from 1.3B to 32B. Perplexity (PPL) on the WikiText-2 [52] validation dataset served as the primary performance metric. Appendix C provides additional results on the C4 [53] and PTB [54] datasets. Furthermore, we assessed zero-shot accuracy on common reasoning benchmarks: WinoGrande [55], OpenBookQA (OBQA) [56], HellaSwag [57], BoolQ [58], ARC-Easy (ARC-e), ARC-Challenge (ARC-c) [59], and PIQA [60].

#### Training Details

We optimized the LittleBit model parameters, initialized via the Dual-SVID method (Section 3.2), using QAT with knowledge distillation (KD) [61, 36, 62]. The original pre-trained full-precision model functioned as the teacher ( $\mathcal{T}$ ) for the LittleBit student model ( $\mathcal{S}$ ). The QAT objective combines the standard output Kullback-Leibler (KL) divergence loss, $\mathcal{L}_{\mathrm{out}}$ , and an intermediate layer mean squared error (MSE) loss, $\mathcal{L}_{\mathrm{inter}}$ , to match hidden representations. We weighted these terms using an empirically determined coefficient $\lambda=10$ :

|
 |
 | $\mathcal{L}_{\mathrm{QAT}}=\mathcal{L}_{\mathrm{out}}+\lambda\mathcal{L}_{\mathrm{inter}}.$
 |
 | (11)

Adhering to the protocol in [24], the training data combined WikiText-2 with selected partitions from C4. The configuration included a sequence length of 2048 tokens, 5 epochs, the Adam optimizer ( $\beta_{1}=0.9,\beta_{2}=0.999$ ) [63], and a cosine learning rate decay with 2% warm-up (see for details). For models employing Grouped-Query Attention (GQA) [64], such as Llama3, Phi-4, and QwQ, we specifically adjusted the latent ranks $r$ for key (K) and value (V) projection layers (see Section B.3) to maintain performance under extreme quantization. We employed SmoothSign (forward: $\mathrm{sign}(x)$ ; backward gradient: $\tanh(kx)$ , $k=100$ ) for backpropagation, as it demonstrated superior performance compared to the Straight-Through Estimator (STE) [31] (see Section B.2). Although proxy gradients are established in the literature, this work validates the specific application of the $\tanh(100x)$ derivative as a stable gradient for QAT in the ultra-low-bit regime. Our ablation study confirms the effectiveness of this approach (Section B.2).

#### Baselines

We benchmarked LittleBit across effective bit rates ranging from approximately 1.0 down to 0.1 bits per weight (BPW). In the challenging sub-1-bit regime, we compared our method against STBLLM [16], a post-training quantization (PTQ) method that employs N:M sparsity. For settings approximating 1.0 BPW, we compared against the 1-bit PTQ method BiLLM [7] and quantization-aware training (QAT) approaches such as OneBit [13] and BinaryMoS [24]. We included results from standard low-bit methods, such as GPTQ [8] and OmniQuant [65], to provide broader context.

### 4.2 Main Results

#### Superior Perplexity in Sub-1-bit Regime

Table 1 presents the perplexity (PPL) results of LittleBit compared to baseline methods. The proposed method demonstrates robust performance, particularly within the sub-1-bit regime. Relative to STBLLM [16], LittleBit achieves markedly superior PPL scores at $0.8$ , $0.7$ , and $0.55$ BPW. For instance, on Llama2-7B, LittleBit at $0.55$ BPW achieves a PPL of $10.47$ , which represents a substantial improvement over the $30.67$ score reported for STBLLM. We observe comparable gains across other models and bit-widths. For Llama2-13B at $0.8$ BPW, LittleBit records a PPL of $8.52$ , whereas STBLLM records $11.90$ .

*Figure: Table 1:
Perplexity (PPL) comparison on WikiText-2 across various LLMs and quantization methods. Lower PPL indicates better performance. BPW denotes effective bits per weight. Methods marked with † use quantization-aware training (QAT). STBLLM [16] utilizes N:M sparsity (ratio in parentheses). LittleBit demonstrates strong performance, particularly in the sub-1-bit regime.
*

|
 | Settings
 | OPT
 | Llama
 | Llama2
 | Llama3
 | Phi-4
 | QwQ

|
 | Method
 | Block Size
 | BPW
 | 1.3B
 | 7B
 | 13B
 | 7B
 | 13B
 | 8B
 | 14.7B
 | 32B

|
 | FullPrecision
 | -
 | 16
 | 14.62
 | 5.67
 | 5.09
 | 5.47
 | 4.88
 | 6.10
 | 6.67
 | 6.34

|
 | OmniQuant
 | -
 | 2
 | 28.82
 | 9.75
 | 7.83
 | 11.20
 | 8.25
 | 349.27
 | 12.09
 | 10.19

|
 | GPTQ
 | 128
 | 2
 | 119.98
 | 39.96
 | 15.08
 | 52.22
 | 23.63
 | 1.5e3
 | 24.96
 | 67.16

|
 | BiLLM
 | 128
 | 1.1
 | 74.07
 | 41.95
 | 13.95
 | 29.00
 | 23.71
 | 54.29
 | 16.95
 | 15.4

|
 | OneBit†
 | -
 | 1
 | 20.27
 | 8.21
 | 7.37
 | 8.36
 | 7.41
 | 13.09
 | 9.92
 | 9.86

|
 | BinaryMoS†
 | -
 | 1
 | 18.09
 | 7.84
 | 7.05
 | 7.74
 | 6.95
 | 10.83
 | 9.51
 | 8.99

|
 | LittleBit †
 | -
 | 1
 | 20.33
 | 9.03
 | 8.17
 | 9.08
 | 8.18
 | 16.30
 | 11.28
 | 12.08

|
 | STBLLM
 | 128
 | 0.80 (6:8)
 | 52.13
 | 15.19
 | 9.43
 | 13.81
 | 11.90
 | 30.90
 | 12.12
 | 12.19

|
 | STBLLM
 | 128
 | 0.70 (5:8)
 | 73.42
 | 19.52
 | 11.47
 | 19.17
 | 14.00
 | 59.83
 | 14.57
 | 13.78

|
 | STBLLM
 | 128
 | 0.55 (4:8)
 | 123.03
 | 38.73
 | 16.88
 | 30.67
 | 27.05
 | 241.95
 | 21.99
 | 18.32

|
 | STBLLM
 | 128
 | 0.30 (2:8)
 | 2.3e3
 | 1.6e3
 | 592.06
 | 1.8e3
 | 893.82
 | 1.7e5
 | 761.05
 | 512.01

|
 | LittleBit †
 | -
 | 0.80
 | 21.32
 | 9.44
 | 8.50
 | 9.53
 | 8.52
 | 17.28
 | 11.70
 | 12.46

|
 | LittleBit †
 | -
 | 0.70
 | 21.99
 | 9.66
 | 8.71
 | 9.85
 | 8.76
 | 18.01
 | 12.05
 | 13.22

|
 | LittleBit †
 | -
 | 0.55
 | 23.35
 | 10.09
 | 9.16
 | 10.47
 | 9.24
 | 18.47
 | 12.80
 | 13.57

|
 | LittleBit †
 | -
 | 0.30
 | 28.30
 | 11.50
 | 10.33
 | 12.00
 | 10.48
 | 20.34
 | 14.71
 | 16.48

|
 | LittleBit †
 | -
 | 0.10
 | 53.76
 | 15.58
 | 13.71
 | 15.92
 | 15.09
 | 26.11
 | 19.73
 | 35.26

*Figure: Table 2:
Zero-shot accuracy (%) comparison on common sense reasoning benchmarks. Compares Full Precision (FP16) models against STBLLM [16] and LittleBit at 0.55 and 0.3 BPW.*

|
 | Models
 | Method
 | WinoGrande
 | OBQA
 | HellaSwag
 | BoolQ
 | ARC-e
 | ARC-c
 | PIQA
 | Average

|
 | Llama2-7B
 | FullPrecision
 | 67.16
 | 40.80
 | 72.95
 | 71.37
 | 69.44
 | 40.95
 | 78.12
 | 62.97

|
 | STBLLM (0.55)
 | 52.80
 | 33.00
 | 36.94
 | 62.17
 | 37.33
 | 25.34
 | 62.46
 | 44.29

|
 | STBLLM (0.3)
 | 50.43
 | 31.80
 | 26.20
 | 37.83
 | 25.88
 | 25.09
 | 53.26
 | 35.78

|
 | LittleBit (0.55)
 | 51.62
 | 34.00
 | 44.57
 | 61.38
 | 44.15
 | 26.96
 | 68.12
 | 47.26

|
 | LittleBit (0.3)
 | 51.30
 | 34.80
 | 37.61
 | 61.80
 | 39.98
 | 25.09
 | 65.83
 | 45.20

|
 | Llama2-13B
 | FullPrecision
 | 69.45
 | 41.80
 | 76.58
 | 69.29
 | 73.19
 | 44.53
 | 78.61
 | 64.78

|
 | STBLLM (0.55)
 | 55.25
 | 31.20
 | 35.10
 | 62.23
 | 42.51
 | 27.22
 | 61.75
 | 45.04

|
 | STBLLM (0.3)
 | 51.54
 | 28.80
 | 26.01
 | 55.63
 | 27.53
 | 24.74
 | 53.32
 | 38.22

|
 | LittleBit (0.55)
 | 53.04
 | 35.60
 | 51.06
 | 50.58
 | 46.09
 | 29.10
 | 70.78
 | 48.03

|
 | LittleBit (0.3)
 | 51.30
 | 34.00
 | 43.81
 | 55.96
 | 42.59
 | 25.17
 | 68.77
 | 45.94

|
 | Llama3-8B
 | FullPrecision
 | 72.92
 | 45.00
 | 79.18
 | 81.25
 | 80.21
 | 52.98
 | 79.54
 | 70.15

|
 | STBLLM (0.55)
 | 52.17
 | 25.80
 | 30.61
 | 57.16
 | 30.35
 | 23.72
 | 57.56
 | 39.62

|
 | STBLLM (0.3)
 | 49.57
 | 26.60
 | 26.36
 | 51.62
 | 26.05
 | 24.40
 | 51.74
 | 36.62

|
 | LittleBit (0.55)
 | 50.12
 | 30.20
 | 36.78
 | 57.55
 | 46.80
 | 22.95
 | 66.27
 | 44.38

|
 | LittleBit (0.3)
 | 51.93
 | 28.20
 | 33.91
 | 57.98
 | 43.48
 | 24.32
 | 64.64
 | 43.49

*Figure: (a) Phi-4 14B*

*Figure: (b) QwQ-32B*

*Figure: Figure 4: Zero-shot accuracy (%) on 7 common sense reasoning tasks. The comparison highlights the robustness of LittleBit (solid lines) against STBLLM (dashed lines) in the sub-0.5 BPW regime. While STBLLM degrades sharply, LittleBit preserves viable reasoning capabilities for both (a) Phi-4 and (b) QwQ models.*

#### Extreme Low-BPW Stability

A primary advantage of LittleBit lies in its exceptional stability and performance in the extreme sub-0.5 BPW range, specifically at $0.3$ BPW and $0.1$ BPW. In this regime, methods such as STBLLM exhibit severe performance degradation. For example, Llama2-7B yields a PPL of $1.8\times 10^{3}$ at $0.3$ BPW. Conversely, LittleBit maintains strong PPL scores, such as $12.00$ for Llama2-7B at $0.3$ BPW and $15.92$ at an unprecedented rate of $0.1$ BPW. When configured for effective bits approximating $1.0$ BPW by adjusting the latent rank $r$ , LittleBit yields perplexity scores comparable to specialized 1-bit QAT methods. For example, on Llama2-7B, LittleBit achieves $9.08$ compared to $8.36$ for OneBit [13]. It also remains competitive with PTQ methods such as BiLLM [7]. Although BinaryMoS [24] may exhibit slightly lower PPL in certain 1-bit scenarios (e.g., Llama2-7B: BinaryMoS $7.74$ ), this performance difference is attributable to its use of more complex mechanisms, such as dynamic scaling, which represent a distinct set of architectural and computational trade-offs. Our analysis of the trade-off involved in adjusting the latent rank to tune effective bits from $1.0$ down to $0.1$ BPW indicates that performance degradation is minimal down to approximately $0.55$ BPW. However, a quantization cliff appears between $0.3$ and $0.1$ BPW. This observation positions the $0.3$ – $0.55$ BPW range as an optimal sweet spot for balancing compression and accuracy.

#### Model Generalization & Zero-Shot

We consistently observe the advantages of LittleBit across a diverse set of model architectures and sizes. This includes the Llama3 family, which is typically considered challenging for quantization, as well as other contemporary models such as Phi-4. For example, Llama3-8B quantized by LittleBit to $0.55$ BPW yields a PPL of $18.47$ . This result substantially outperforms the STBLLM score of $241.95$ at $0.55$ BPW with 4:8 sparsity. The zero-shot evaluation results presented in Table 2 corroborate these perplexity-based findings. For instance, Llama2-7B compressed by LittleBit to $0.55$ BPW achieves a mean accuracy of $47.26\%$ , and the model maintains $45.20\%$ mean accuracy even at an extreme $0.3$ BPW. These results compare favorably with those of STBLLM at similar or even higher effective bits. STBLLM on Llama2-7B at $0.55$ BPW scores $44.29\%$ . This outcome suggests superior preservation of the intrinsic reasoning capabilities of the models even under extreme compression. It highlights the efficacy of LittleBit in creating highly compact yet powerful LLMs.

### 4.3 Reasoning Performance of Extremely Compressed LLMs

The capability of LittleBit to achieve sub-1-bit compression while maintaining robust model fidelity prompts an evaluation of its reasoning abilities, particularly for large-scale models under extreme compression. We investigated the Phi-4 14B model and extended our analysis to the QwQ-32B model, comparing their performance against STBLLM across 7 zero-shot reasoning tasks.

When compressed with LittleBit to $0.55$ BPW, Phi-4 achieved an average accuracy of approximately $52.3\%$ . Although this performance is marginally lower than STBLLM at a similar compression level ( $54.2\%$ at $0.55$ BPW), LittleBit demonstrates substantially more graceful degradation. At the more aggressive $0.3$ BPW setting, the LittleBit-compressed Phi-4 maintained an average accuracy of $48.7\%$ , which markedly outperforms STBLLM at $0.3$ BPW ( $40.3\%$ ). Even at an extreme $0.1$ BPW, Phi-4 retained a notable accuracy of $43.6\%$ . We observed a consistent trend with the QwQ-32B model, as illustrated in Figure 4. While STBLLM exhibited severe performance collapse in the sub-0.5 BPW regime, LittleBit maintained a stable trajectory down to 0.1 BPW. These results highlight the ability of LittleBit to preserve robust reasoning capabilities across varying model scales as compression becomes increasingly extreme.

*Figure: Table 3: Memory footprint comparison (GB) for Llama2 [47] models under different quantization methods. LittleBit is evaluated at various BPWs, achieved by adjusting the latent rank $r$ . Compression factors relative to FP16 are shown in parentheses.The gap between the effective BPW and the compression factor arises because quantization is applied only to the Transformer blocks, while components like the embedding layer and the lm_head remain in FP16.*

|
 | Model
 | FP16
 | BiLLM [7]
 | OneBit [13]
 | LittleBit (Ours)

|
 | BPW
 | 16
 | 1.1
 | 1
 | 0.8
 | 0.55
 | 0.3
 | 0.1

|
 | Llama2-7B
 | 13.49 GB
 | 1.60 GB (8.43 $\times$ )
 | 1.36 GB (9.92 $\times$ )
 | 1.19 GB (11.34 $\times$ )
 | 0.98 GB (13.77 $\times$ )
 | 0.79 GB (17.08 $\times$ )
 | 0.63 GB (21.41 $\times$ )

|
 | Llama2-13B
 | 26.06 GB
 | 2.80 GB (9.31 $\times$ )
 | 2.28 GB (11.43 $\times$ )
 | 1.95 GB (13.36 $\times$ )
 | 1.51 GB (17.26 $\times$ )
 | 1.15 GB (22.66 $\times$ )
 | 0.84 GB (31.02 $\times$ )

|
 | Llama2-70B
 | 138.04 GB
 | 15.40 GB (8.96 $\times$ )
 | 9.72 GB (14.20 $\times$ )
 | 7.97 GB (17.31 $\times$ )
 | 5.83 GB (23.68 $\times$ )
 | 3.70 GB (37.31 $\times$ )
 | 1.98 GB (69.72 $\times$ )

*Figure: Figure 5: Conceptual view of KV Cache storage: the standard method (left) stores the full hidden dimension ( $d_{\mathrm{model}}$ ), whereas LittleBit (right) caches a reduced latent dimension ( $r$ )*

|
 | BPW
 | KV Latent Rank
 | KV Cache

|
 |
 | ( $r$ )
 | Reduction Factor

|
 | 0.80
 | 1,624
 | $\approx$ 2.5 $\times$

|
 | 0.55
 | 1,112
 | $\approx$ 3.7 $\times$

|
 | 0.30
 | 600
 | $\approx$ 6.8 $\times$

|
 | 0.10
 | 192
 | $\approx$ 21.3 $\times$

*Figure: Table 4: Estimated KV cache memory reduction for Llama2-7B using LittleBit, with a reduction factor of approximately $d_{\mathrm{model}}/r$ .*

## 5 Analysis

#### Memory Footprint Reduction

LittleBit is designed for resource-constrained environments, such as on-device deployment, and achieves substantial reductions in model memory footprint. As summarized in Table 3, quantizing a Llama2-7B model (originally $13.49$ GB in FP16) to an effective bit rate of $0.3$ BPW using LittleBit reduces its required storage to $0.79$ GB. This corresponds to a compression factor exceeding $17\times$ . At the extreme setting of $0.1$ BPW, the footprint is further reduced to $0.63$ GB (over $21\times$ compression). We observe comparable reductions for larger models, such as Llama2-70B, where the memory footprint decreases from $138.04$ GB to under $2$ GB at $0.1$ BPW, achieving nearly $70\times$ compression. These substantial memory reductions significantly expand the range of devices capable of hosting large-scale language models, thereby enabling complex language tasks on hardware with limited VRAM or storage capacity.

#### KV Cache Compression

The proposed factorization (Eq. 4) inherently compresses the KV cache when applied to key and value projection matrices. Because the forward computation (Eq. 5) operates on latent states, intermediate rank $r$ vectors are cached rather than full ${d}_{\mathrm{model}}$ -dimensional vectors, as illustrated in Fig. 5. Consequently, KV cache memory requirements are reduced by a factor of approximately ${d}_{\mathrm{model}}/{r}$ . For example, a reduction of up to 21.3 $\times$ is observed for Llama2-7B at 0.1 BPW (Section 4.3). While this effect aligns with explicit compression methods such as MLA [66] and ASVD [42], the approach described herein achieves this via unified factorization within attention layers, thereby reducing both weight and activation memory simultaneously.

*Figure: Figure 6: Kernel-level latency (bars, left axis) and speedup relative to FP16 (dashed line, right axis) on an NVIDIA A100 GPU for a Llama2-70B MLP layer ( $8,192\times 28,672$ ). The benchmark compares the FP16 GEMM baseline with OneBit and the proposed LittleBit (LB) kernel across various BPW configurations ranging from 1.0 to 0.1.*

#### Latency Considerations

In addition to memory optimization, inference latency presents a critical performance metric. The factorized architecture (Eq. 5) facilitates computational acceleration through low-rank binary operations, distinct from methods solely prioritizing memory or requiring specific hardware. To evaluate this capability, a custom 1-bit GEMV CUDA kernel was developed to accelerate the primary computational path. As illustrated in Fig. 6 for a Llama2-70B MLP layer, the kernel demonstrates speedups that inversely correlate with bit-width, peaking at 11.6 $\times$ over an optimized FP16 baseline at 0.1 BPW.

These findings suggest the proposed method enables efficient deployment with reduced latency. The discrepancy between theoretical computational gains and practical latency is likely attributable to memory access dominance during small-batch inference. The factorized approach involves multiple memory-bound operations, and the custom kernel has not yet reached the optimization level of industry-standard libraries. provides a comprehensive analysis covering theoretical FLOPs, practical latency, and end-to-end decoding throughput.

## 6 Conclusion

This study proposed LittleBit, a method designed to advance LLM compression into the sub-0.5 BPW regime while maintaining viable performance at 0.1 BPW. Performance improvements were realized through SVD-inspired latent matrix factorization, factor binarization, and a multi-scale compensation mechanism encompassing row, column, and latent dimensions. Furthermore, Dual-SVID initialization was introduced to stabilize quantization-aware training (QAT), alongside Residual Compensation for error mitigation. Empirical results demonstrate that the proposed method markedly outperforms prior sub-1-bit techniques and preserves fidelity across LLMs scaling up to 32B parameters. Consequently, LittleBit suggests a favorable size–performance trade-off, thereby facilitating LLM deployment in resource-constrained environments. Future research may address practical deployment through hardware co-design for edge devices, such as neural processing units (NPUs). Additional improvements in model fidelity could be examined by integrating advanced techniques such as non-uniform bit allocation, applied either across model layers or within single layers via dynamic bit budget distribution between primary and residual paths.

## Acknowledgements

We are grateful to Dr. Heonjae Ha, Dr. SangJeong Lee, Minseop Choi, Changdong Kim, and Hyochan Chong for their insightful discussions and generous support in this work.

## References

- [1]
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017)

Attention is All You Need.

Advances in Neural Information Processing Systems 30.

Cited by: §1.

- [2]
D. Narayanan, M. Shoeybi, J. Casper, P. LeGresley, M. Patwary, V. Korthikanti, D. Vainbrand, P. Kashinkunti, J. Bernauer, B. Catanzaro, et al. (2021)

Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.

In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis,

pp. 1–15.

Cited by: §1.

- [3]
M. Isaev, N. McDonald, and R. Vuduc (2023)

Scaling Infrastructure to Support Multi-Trillion Parameter LLM Training.

In Architecture and System Support for Transformer Models (ASSYST@ ISCA 2023),

Cited by: §1.

- [4]
Y. Zheng, Y. Chen, B. Qian, X. Shi, Y. Shu, and J. Chen (2025)

A Review on Edge Large Language Models: Design, Execution, and Applications.

ACM Computing Surveys 57 (8), pp. 1–35.

Cited by: §1.

- [5]
X. Pan, E. Li, Q. Li, S. Liang, Y. Shan, K. Zhou, Y. Luo, X. Wang, and J. Zhang (2024)

InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference.

arXiv preprint arXiv:2409.04992.

Cited by: §1.

- [6]
L. Zhang, Z. Zhang, R. Li, Z. Tian, S. Mei, D. Li, et al. (2025)

Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference.

In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing,

pp. 17393–17406.

Cited by: §1.

- [7]
W. Huang, Y. Liu, H. Qin, Y. Li, S. Zhang, X. Liu, M. Magno, and X. Qi (2024)

BiLLM: Pushing the Limit of Post-Training Quantization for LLMs.

arXiv preprint arXiv:2402.04291.

Cited by: §1,
§1,
§2.1,
§4.1,
§4.2,
Table 3.

- [8]
E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh (2022)

GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers.

arXiv preprint arXiv:2210.17323.

Cited by: §1,
§2.1,
§4.1.

- [9]
J. Lin, J. Tang, H. Tang, S. Yang, W. Chen, W. Wang, G. Xiao, X. Dang, C. Gan, and S. Han (2024)

AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration.

Proceedings of Machine Learning and Systems 6, pp. 87–100.

Cited by: §1,
§2.1.

- [10]
B. Jacob, S. Kligys, B. Chen, M. Zhu, M. Tang, A. Howard, H. Adam, and D. Kalenichenko (2018)

Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.

In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),

pp. 2704–2713.

Cited by: §1.

- [11]
J. Kim, Y. Bhalgat, J. Lee, C. Patel, and N. Kwak (2019)

QKD: Quantization-aware Knowledge Distillation.

arXiv preprint arXiv:1911.12491.

Cited by: §1.

- [12]
H. Wang, S. Ma, L. Dong, S. Huang, H. Wang, L. Ma, F. Yang, R. Wang, Y. Wu, and F. Wei (2023)

BitNet: Scaling 1-bit Transformers for Large Language Models.

arXiv preprint arXiv:2310.11453.

Cited by: §1.

- [13]
Y. Xu, X. Han, Z. Yang, S. Wang, Q. Zhu, Z. Liu, W. Liu, and W. Che (2024)

OneBit: Towards Extremely Low-bit Large Language Models.

Advances in Neural Information Processing Systems 37, pp. 66357–66382.

Cited by: §1,
§1,
§2.1,
§2.2,
§3.1,
§4.1,
§4.2,
Table 3.

- [14]
L. Ma, M. Sun, and Z. Shen (2024)

FBI-LLM: Scaling up Fully Binarized LLMs from Scratch via Autoregressive Distillation.

arXiv preprint arXiv:2407.07093.

Cited by: §1.

- [15]
Y. Zheng, Y. Chen, B. Qian, X. Shi, Y. Shu, and J. Chen (2025)

A Review on Edge Large Language Models: Design, Execution, and Applications.

ACM Computing Surveys 57 (8), pp. 1–35.

Cited by: §1.

- [16]
P. Dong, L. Li, Y. Zhong, D. Du, R. Fan, Y. Chen, Z. Tang, Q. Wang, W. Xue, Y. Guo, et al. (2024)

STBLLM: Breaking the 1-bit Barrier with Structured Binary LLMs.

arXiv preprint arXiv:2408.01803.

Cited by: Table 8,
3rd item,
§1,
§1,
§2.1,
§4.1,
§4.2,
Table 1,
Table 2,
Table 2.

- [17]
A. Jaiswal, L. Yin, Z. Zhang, S. Liu, J. Zhao, Y. Tian, and Z. Wang (2024)

From Low Rank Gradient Subspace Stabilization to Low-Rank Weights: Observations, Theories, and Applications.

arXiv preprint arXiv:2407.11239.

Cited by: §1.

- [18]
Q. Wang, J. Ke, M. Tomizuka, Y. Chen, K. Keutzer, and C. Xu (2025)

Dobi-SVD: Differentiable SVD for LLM Compression and Some New Perspectives.

arXiv preprint arXiv:2502.02723.

Cited by: §1.

- [19]
G. H. Golub and C. Reinsch (1971)

Singular Value Decomposition and Least Squares Solutions.

In Linear algebra,

pp. 134–151.

Cited by: §1,
§2.2.

- [20]
S. Han, H. Mao, and W. J. Dally (2015)

Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding.

arXiv preprint arXiv:1510.00149.

Cited by: §1,
§2.2.

- [21]
X. Wang, S. Alam, Z. Wan, H. Shen, and M. Zhang (2025)

SVD-LLM V2: Optimizing Singular Value Truncation for Large Language Model Compression.

arXiv preprint arXiv:2503.12340.

Cited by: Table 9,
Appendix E,
§1,
§2.2.

- [22]
M. Sun, Z. Liu, A. Bair, and J. Z. Kolter (2023)

A Simple and Effective Pruning Approach for Large Language Models.

arXiv preprint arXiv:2306.11695.

Cited by: Table 9,
Appendix E,
§1.

- [23]
R. Gong, Y. Ding, Z. Wang, C. Lv, X. Zheng, J. Du, Y. Yong, S. Gu, H. Qin, J. Guo, et al. (2025)

A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms.

Neural Networks, pp. 107856.

Cited by: §1,
§3.1.

- [24]
D. Jo, T. Kim, Y. Kim, and J. Kim (2024)

Mixture of Scales: Memory-Efficient Token-Adaptive Binarization for Large Language Models.

Advances in Neural Information Processing System 37, pp. 137474–137494.

Cited by: §1,
§2.1,
§4.1,
§4.1,
§4.2.

- [25]
Z. Li, X. Yan, T. Zhang, H. Qin, D. Xie, J. Tian, L. Kong, Y. Zhang, X. Yang, et al. (2024)

ARB-LLM: Alternating Refined Binarizations for Large Language Models.

arXiv preprint arXiv:2410.03129.

Cited by: §1,
§3.1.

- [26]
I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio (2016)

Binarized Neural Networks.

Advances in Neural Information Processing Systems 29.

Cited by: §2.1.

- [27]
C. Fu, S. Zhu, H. Su, C. Lee, and J. Zhao (2018)

Towards Fast and Energy-Efficient Binarized Neural Network Inference on FPGA.

arXiv preprint arXiv:1810.02068.

Cited by: §2.1.

- [28]
M. Courbariaux, Y. Bengio, and J. David (2015)

BinaryConnect: Training Deep Neural Networks with Binary Weights during Propagations.

Vol. 28.

Cited by: §2.1.

- [29]
H. Qin, R. Gong, X. Liu, X. Bai, J. Song, and N. Sebe (2020)

Binary Neural Networks: A Survey.

Pattern Recognition 105, pp. 107281.

Cited by: §2.1.

- [30]
M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi (2016)

XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.

In European Conference on Computer Vision,

pp. 525–542.

Cited by: §2.1.

- [31]
Y. Bengio, N. Léonard, and A. Courville (2013)

Estimating or Propagating Gradients through Stochastic Neurons for Conditional Computation.

arXiv preprint arXiv:1308.3432.

Cited by: §B.2,
§2.1,
§4.1.

- [32]
M. Schoenbauer, D. Moro, L. Lew, and A. Howard (2024)

Custom Gradient Estimators are Straight-Through Estimators in Disguise.

arXiv preprint arXiv:2405.05171.

Cited by: §2.1.

- [33]
Y. Shang, Z. Yuan, Q. Wu, and Z. Dong (2023)

PB-LLM: Partially Binarized Large Language Models.

arXiv preprint arXiv:2310.00034.

Cited by: §2.1,
§3.1.

- [34]
A. Gholami, S. Kim, Z. Dong, Z. Yao, M. W. Mahoney, and K. Keutzer (2022)

A Survey of Quantization Methods for Efficient Neural Network Inference.

In Low-power computer vision,

pp. 291–326.

Cited by: §2.1.

- [35]
M. Denil, B. Shakibi, L. Dinh, M. Ranzato, and N. De Freitas (2013)

Predicting Parameters in Deep Learning.

Advances in Neural Information Processing Systems 26.

Cited by: §2.1,
§2.2.

- [36]
Z. Liu, B. Oguz, C. Zhao, E. Chang, P. Stock, Y. Mehdad, Y. Shi, R. Krishnamoorthi, and V. Chandra (2024)

LLM-QAT: Data-Free Quantization Aware Training for Large Language Models.

In Findings of the Association for Computational Linguistics: ACL 2024,

pp. 467–484.

Cited by: §2.1,
§4.1.

- [37]
X. Men, M. Xu, Q. Zhang, Q. Yuan, B. Wang, H. Lin, Y. Lu, X. Han, and W. Chen (2025)

ShortGPT: Layers in Large Language Models are More Redundant Than You Expect.

In Findings of the Association for Computational Linguistics: ACL 2025,

pp. 20192–20204.

Cited by: §2.2.

- [38]
S. He, G. Sun, Z. Shen, and A. Li (2024)

What Matters in Transformers? Not All Attention is Needed.

arXiv preprint arXiv:2406.15786.

Cited by: §2.2.

- [39]
R. Saha, N. Sagan, V. Srivastava, A. Goldsmith, and M. Pilanci (2024)

Compressing Large Language Models using Low Rank and Low Precision Decomposition.

Advances in Neural Information Processing Systems 37, pp. 88981–89018.

Cited by: §2.2.

- [40]
Q. Zhang, M. Chen, A. Bukharin, N. Karampatziakis, P. He, Y. Cheng, W. Chen, and T. Zhao (2023)

AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning.

arXiv preprint arXiv:2303.10512.

Cited by: §2.2.

- [41]
F. Meng, Z. Wang, and M. Zhang (2024)

PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.

Advances in Neural Information Processing Systems 37, pp. 121038–121072.

Cited by: §2.2.

- [42]
Z. Yuan, Y. Shang, Y. Song, D. Yang, Q. Wu, Y. Yan, and G. Sun (2023)

ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models.

arXiv preprint arXiv:2312.05821.

Cited by: §2.2,
§5.

- [43]
C. Zhang, J. Cheng, G. A. Constantinides, and Y. Zhao (2024)

LQER: Low-Rank Quantization Error Reconstruction for LLMs.

arXiv preprint arXiv:2402.02446.

Cited by: §2.2.

- [44]
W. Ke, Z. Li, D. Li, L. Tian, and E. Barsoum (2024)

DL-QAT: Weight-Decomposed Low-Rank Quantization-Aware Training for Large Language Models.

In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track,

pp. 113–119.

Cited by: §2.2.

- [45]
X. Wang, Y. Zheng, Z. Wan, and M. Zhang (2024)

SVD-LLM: Truncation-aware Singular Value Decomposition for Large Language Model Compression.

arXiv preprint arXiv:2403.07378.

Cited by: §3.1.

- [46]
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, et al. (2023)

LLaMA: Open and efficient foundation language models.

arXiv preprint arXiv:2302.13971.

Cited by: §4.1.

- [47]
H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. (2023)

Llama 2: Open Foundation and Fine-tuned Chat Models.

arXiv preprint arXiv:2307.09288.

Cited by: §4.1,
Table 3.

- [48]
A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, et al. (2024)

The Llama 3 Herd of Models.

arXiv preprint arXiv:2407.21783.

Cited by: §4.1.

- [49]
S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al. (2022)

OPT: Open Pre-trained Transformer Language Models.

arXiv preprint arXiv:2205.01068.

Cited by: §4.1.

- [50]
M. Abdin, J. Aneja, H. Behl, S. Bubeck, R. Eldan, S. Gunasekar, M. Harrison, R. J. Hewett, M. Javaheripi, P. Kauffmann, et al. (2024)

Phi-4 Technical Report.

arXiv preprint arXiv:2412.08905.

Cited by: §4.1.

- [51]
Q. Team (2025)

QwQ-32B: Embracing the Power of Reinforcement Learning.

External Links: Link

Cited by: §4.1.

- [52]
S. Merity, C. Xiong, J. Bradbury, and R. Socher (2016)

Pointer Sentinel Mixture Models.

arXiv preprint arXiv:1609.07843.

Cited by: §4.1.

- [53]
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2020)

Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.

Journal of Machine Learning Research 21 (140), pp. 1–67.

Cited by: §4.1.

- [54]
M. Marcus, G. Kim, M. A. Marcinkiewicz, R. MacIntyre, A. Bies, M. Ferguson, K. Katz, and B. Schasberger (1994)

The Penn Treebank: Annotating Predicate Argument Structure.

In Proceedings of the Workshop on Human Language Technology,

HLT ’94, pp. 114–119.

Cited by: §4.1.

- [55]
K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi (2021)

WinoGrande: An Adversarial Winograd Schema Challenge at Scale.

Communications of the ACM 64 (9), pp. 99–106.

Cited by: §4.1.

- [56]
T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal (2018)

Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering.

arXiv preprint arXiv:1809.02789.

Cited by: §4.1.

- [57]
R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi (2019)

HellaSwag: Can a Machine Really Finish Your Sentence?.

arXiv preprint arXiv:1905.07830.

Cited by: §4.1.

- [58]
C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova (2019)

BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions.

arXiv preprint arXiv:1905.10044.

Cited by: §4.1.

- [59]
P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)

Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.

arXiv preprint arXiv:1803.05457.

Cited by: §4.1.

- [60]
Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. (2020)

PIQA: Reasoning about physical commonsense in natural language.

In Proceedings of the AAAI Conference on Artificial Intelligence,

Vol. 34, pp. 7432–7439.

Cited by: §4.1.

- [61]
G. Hinton, O. Vinyals, and J. Dean (2015)

Distilling the Knowledge in a Neural Network.

arXiv preprint arXiv:1503.02531.

Cited by: §4.1.

- [62]
W. Zhang, L. Hou, Y. Yin, L. Shang, X. Chen, X. Jiang, and Q. Liu (2020)

TernaryBERT: Distillation-aware Ultra-low Bit BERT.

arXiv preprint arXiv:2009.12812.

Cited by: §4.1.

- [63]
D. P. Kingma and J. Ba (2015)

Adam: A Method for Stochastic Optimization.

In Proceedings of the Third International Conference on Learning Representations (ICLR),

Cited by: §4.1.

- [64]
J. Ainslie, J. Lee-Thorp, M. De Jong, Y. Zemlyanskiy, F. Lebrón, and S. Sanghai (2023)

GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.

arXiv preprint arXiv:2305.13245.

Cited by: §B.3,
§4.1.

- [65]
W. Shao, M. Chen, Z. Zhang, P. Xu, L. Zhao, Z. Li, K. Zhang, P. Gao, Y. Qiao, and P. Luo (2023)

OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models.

arXiv preprint arXiv:2308.13137.

Cited by: §4.1.

- [66]
DeepSeek-AI (2024)

DeepSeek-V2: A Strong Mixture-of-Experts Language Model with Expert Attention.

arXiv preprint arXiv:2405.04434.

Cited by: §5.

- [67]
N. Shazeer (2019)

Fast Transformer Decoding: One Write-Head is All You Need.

arXiv preprint arXiv:1911.02150.

Cited by: §B.3.

## Appendix A Mathematical Analysis and Proofs

This section provides mathematical details for the LittleBit computation and explores theoretical aspects related to low-rank quantization errors, offering motivation for certain design choices.

###### Proposition 1 (LittleBit Forward Pass Computation (Proof Detail)).

As stated in Section 3.1 (Proposition 1), for input $\mathbf{X}\in\mathbb{R}^{\mathrm{seq}\times d_{\mathrm{in}}}$ , the primary quantized weight matrix is $\widehat{\mathbf{W}}_{\mathrm{pri}}=\mathrm{diag}(\mathbf{h})\mathbf{U}_{\mathrm{sign}}\mathrm{diag}(\bm{\ell})\mathbf{V}_{\mathrm{sign}}^{\top}\mathrm{diag}(\mathbf{g})$ . The forward pass $\mathbf{Y}=\mathbf{X}\widehat{\mathbf{W}}_{\mathrm{pri}}^{\top}$ can be computed as shown in Eq. 12.

|
 |
 | $\mathbf{Y}=((((\mathbf{X}\odot\mathbf{g})\mathbf{V}_{\mathrm{sign}})\odot\bm{\ell})\mathbf{U}_{\mathrm{sign}}^{\top})\odot\mathbf{h}.$
 |
 | (12)

Terms are defined in Section 3.1.

###### Proof.

Starting with $\mathbf{Y}=\mathbf{X}\widehat{\mathbf{W}}_{\mathrm{pri}}^{\top}$ :

|
 |
 | $\displaystyle\mathbf{Y}$
 | $\displaystyle=\mathbf{X}\left(\mathrm{diag}(\mathbf{h})\mathbf{U}_{\mathrm{sign}}\mathrm{diag}(\bm{\ell})\mathbf{V}_{\mathrm{sign}}^{\top}\mathrm{diag}(\mathbf{g})\right)^{\top}$
 |

|
 |
 |
 | $\displaystyle=\mathbf{X}\,\mathrm{diag}(\mathbf{g})\mathbf{V}_{\mathrm{sign}}\mathrm{diag}(\bm{\ell})\mathbf{U}_{\mathrm{sign}}^{\top}\mathrm{diag}(\mathbf{h})$
 |

|
 |
 |
 | $\displaystyle=((((\mathbf{X}\odot\mathbf{g})\mathbf{V}_{\mathrm{sign}})\odot\bm{\ell})\mathbf{U}_{\mathrm{sign}}^{\top})\odot\mathbf{h}.$
 |
 | (13)

This yields Eq. 12, where $\odot$ implies element-wise multiplication with broadcasting.
∎

###### Claim 1 (Quantization Error vs. Factor Rank).

Consider a simplified model where a rank- $r$ matrix $\mathbf{W}^{(r)}=\mathbf{U}^{(r)}(\mathbf{V}^{(r)})^{\top}$ is approximated. Let $\mathbf{U}_{\mathrm{sign}}^{(r)}=\mathrm{sign}(\mathbf{U}^{(r)})$ , $\mathbf{V}_{\mathrm{sign}}^{(r)}=\mathrm{sign}(\mathbf{V}^{(r)})$ . If magnitudes are crudely approximated using fixed rank-1 SVDs of $|\mathbf{U}^{(r)}|$ and $|\mathbf{V}^{(r)}|$ to form scales $\mathbf{s}_{U}^{(r)},\mathbf{s}_{V}^{(r)}$ :

|
 |
 | $\widehat{\mathbf{W}}_{r}:=\left(\mathbf{U}_{\mathrm{sign}}^{(r)}\odot(\mathbf{s}_{U}^{(r)}\mathbf{1}^{\top}_{r})\right)\left(\mathbf{V}_{\mathrm{sign}}^{(r)}\odot(\mathbf{s}_{V}^{(r)}\mathbf{1}^{\top}_{r})\right)^{\top}.$
 |
 | (14)

The error $\mathcal{E}(r):=\left\|\mathbf{W}^{(r)}-\widehat{\mathbf{W}}_{r}\right\|_{F}$ might be non-decreasing with $r$ .

###### Proof Sketch / Heuristic Argument.

As rank $r$ (and thus complexity of $\mathbf{W}^{(r)}$ , $\mathbf{U}^{(r)}$ , $\mathbf{V}^{(r)}$ ) increases, the fixed-structure rank-1 magnitude approximation (via $\mathbf{s}_{U}^{(r)},\mathbf{s}_{V}^{(r)}$ ) may become a relatively poorer fit for the increasingly complex actual magnitudes of $\mathbf{U}^{(r)},\mathbf{V}^{(r)}$ . This degradation in scaling accuracy could lead to $\mathcal{E}(r)$ not decreasing, or even increasing, motivating more sophisticated scaling as in LittleBit.
∎

###### Claim 2 (Quantization Bias vs. SVD Component Structure).

Let $\mathbf{W}$ ’s rank- $r$ SVD approximation be $\widetilde{\mathbf{W}}_{r}=\widetilde{\mathbf{W}}_{r_{1}}+\widetilde{\mathbf{R}}_{r_{2}}$ (primary $\widetilde{\mathbf{W}}_{r_{1}}$ , residual $\widetilde{\mathbf{R}}_{r_{2}}$ ). Consider a quantization operator $\mathcal{Q}$ . The relative quantization error $\|\mathcal{Q}(\mathbf{A})-\mathbf{A}\|_{F}/\|\mathbf{A}\|_{F}$ might be larger for matrices $\mathbf{A}$ with "flatter" singular value spectra or less energy concentration (e.g., $\widetilde{\mathbf{R}}_{r_{2}}$ vs. $\widetilde{\mathbf{W}}_{r_{1}}$ ). This suggests that $\mathcal{Q}$ applied to $\widetilde{\mathbf{R}}_{r_{2}}$ might be less effective (i.e., higher relative quantization error) than when applied to $\widetilde{\mathbf{W}}_{r_{1}}$ .

###### Proof Sketch.

 $\widetilde{\mathbf{W}}_{r_{1}}$ has dominant singular values, while $\widetilde{\mathbf{R}}_{r_{2}}$ has smaller ones, often resulting in lower energy concentration for $\widetilde{\mathbf{R}}_{r_{2}}$ . If $\mathcal{Q}$ ’s relative error is sensitive to this (e.g., higher for lower concentration), quantizing $\widetilde{\mathbf{W}}_{r_{1}}$ and $\widetilde{\mathbf{R}}_{r_{2}}$ separately may be better than quantizing $\widetilde{\mathbf{W}}_{r}$ jointly, as it allows adapting $\mathcal{Q}$ to their differing structures.
∎

###### Proposition 2 (Potential Advantage of Two-Stage Quantization of SVD Components).

Using the setup of Claim 2, let $\widehat{\mathbf{W}}_{r}^{\mathrm{single}}=\mathcal{Q}(\widetilde{\mathbf{W}}_{r})$ and $\widehat{\mathbf{W}}^{\mathrm{two-stage}}=\mathcal{Q}(\widetilde{\mathbf{W}}_{r_{1}})+\mathcal{Q}(\widetilde{\mathbf{R}}_{r_{2}})$ . The two-stage error $\mathcal{E}_{\mathrm{two-stage}}=\left\|\mathbf{W}-\widehat{\mathbf{W}}^{\mathrm{two-stage}}\right\|_{F}$ can be less than the single-stage error $\mathcal{E}_{\mathrm{single-stage}}=\left\|\mathbf{W}-\widehat{\mathbf{W}}_{r}^{\mathrm{single}}\right\|_{F}$ . This occurs if the quantization error of the sum is greater than the sum of (vectorial) quantization errors of parts:

|
 |
 | $\|(\mathcal{Q}(\widetilde{\mathbf{W}}_{r_{1}})-\widetilde{\mathbf{W}}_{r_{1}})+(\mathcal{Q}(\widetilde{\mathbf{R}}_{r_{2}})-\widetilde{\mathbf{R}}_{r_{2}})\|_{F}<\|\mathcal{Q}(\widetilde{\mathbf{W}}_{r})-\widetilde{\mathbf{W}}_{r}\|_{F}.$
 |
 | (15)

###### Proof Outline and Discussion.

Let $\mathbf{E}_{\mathrm{trunc}}=\mathbf{W}-\widetilde{\mathbf{W}}_{r}$ . Errors are $\Delta_{1}=\mathcal{Q}(\widetilde{\mathbf{W}}_{r_{1}})-\widetilde{\mathbf{W}}_{r_{1}}$ , $\Delta_{2}=\mathcal{Q}(\widetilde{\mathbf{R}}_{r_{2}})-\widetilde{\mathbf{R}}_{r_{2}}$ , $\Delta_{\mathrm{sum}}=\mathcal{Q}(\widetilde{\mathbf{W}}_{r})-\widetilde{\mathbf{W}}_{r}$ .
Then $\mathcal{E}_{\mathrm{two-stage}}=\|\mathbf{E}_{\mathrm{trunc}}-(\Delta_{1}+\Delta_{2})\|_{F}$ and $\mathcal{E}_{\mathrm{single-stage}}=\|\mathbf{E}_{\mathrm{trunc}}-\Delta_{\mathrm{sum}}\|_{F}$ .
The condition Eq. 15 makes $\mathcal{E}_{\mathrm{two-stage}}<\mathcal{E}_{\mathrm{single-stage}}$ likely. Non-linear $\mathcal{Q}$ (e.g., with adaptive scaling) can satisfy this by better handling the distinct characteristics of $\widetilde{\mathbf{W}}_{r_{1}}$ and $\widetilde{\mathbf{R}}_{r_{2}}$ . This motivates LittleBit’s separate residual handling, though its residual is learned differently.
∎

## Appendix B Ablation Study

### B.1 Residual Compensation

To provide further intuition behind the benefits of Residual Compensation, we visualize the output activations of selected Transformer layers under different quantization schemes. Specifically, we extract the final outputs from Transformer layers 0, 5, 10, and 15 of an OPT-1.3B model, evaluated on two randomly sampled input sequences. For each layer and input, we display a set of three activation maps corresponding to the full-precision baseline, as well as two quantized variants—with and without residual compensation. These groupings allow us to directly observe the structural differences induced by the presence or absence of the residual compensation.

As shown in  Fig. 7, residual compensation preserves a high-fidelity resemblance to the full-precision baseline across all inspected layers. In the no-residual setting, the output feature distributions exhibit noticeable deformation relative to the full-precision baseline. In contrast, applying residual compensation consistently restores structural similarity to the original distribution, preserving both directional patterns and magnitude diversity.

Table 5 presents the PPL results for the OPT-1.3B model, comparing configurations with and without residual compensation. As indicated in the table, at the extremely low bit of 0.1 BPW, the version with residual compensation (PPL 60.011) performed worse than the version without it (PPL 48.512). This suggests that for relatively small models (1.3B), when pushing effective bits to such extreme lows, the additional complexity introduced by the residual compensation mechanism might outweigh its benefits and instead hinder performance. However, for the same OPT-1.3B model at other effective bits evaluated (ranging from 0.3 BPW to 1.0 BPW), the inclusion of residual compensation consistently resulted in better (lower) PPL. Moreover, experiments conducted with other, larger models generally showed that residual compensation tended to provide superior performance. In light of these overall advantages observed across various settings and particularly for larger models, residual compensation was adopted as a standard component in all experiments presented in this study.

|
 |
 |

Input 1
  
 |

Input 2

|
 |

layer-0 output

 |

  
 |

|
 |

layer-5 output

 |

  
 |

|
 |

layer-10 output

 |

  
 |

|
 |

layer-15 output

 |

  
 |

*Figure: Figure 7: Visualization of Transformer layer outputs for layers 0, 5, 10, and 15 under the LittleBit quantization scheme on OPT-1.3B. For each of two randomly sampled input sequences (annotated at the top), we show horizontally grouped triplets of activation maps per layer: full-precision baseline (left), quantized with residual compensation (center), and quantized without residual compensation (right).*

*Figure: Table 5: Perplexity (WikiText-2) comparison between Residual and Non-Residual for OPT-1.3B at various BPWs. Learning rate was 8e-5. Lower PPL is better.*

|
 | BPW
 | Residual PPL
 | Non-Residual PPL
 | Difference (Residual - Non-Residual)

|
 | 1.00
 | 20.329
 | 21.691
 | -1.362

|
 | 0.80
 | 21.338
 | 22.688
 | -1.350

|
 | 0.70
 | 22.001
 | 23.313
 | -1.312

|
 | 0.55
 | 23.457
 | 24.788
 | -1.331

|
 | 0.30
 | 28.984
 | 29.724
 | -0.740

|
 | 0.10
 | 60.011
 | 48.512
 | 11.499

### B.2 SmoothSign versus Straight-Through Estimator

For backpropagating through the non-differentiable $\mathrm{sign}(x)$ function, we compared the Straight-Through Estimator (STE) [31] with our proposed SmoothSign technique. SmoothSign employs the $\mathrm{sign}(x)$ function for the forward pass. For the backward pass, it utilizes a smooth proxy gradient, specifically the derivative of $\tanh(kx)$ where $k=100$ (illustrated in Fig. 8).

Table 6 (OPT-1.3B results) shows SmoothSign yielding better PPL as effective bits decrease, with a notable advantage at 0.1 BPW. Due to its superior stability and performance in the ultra-low bit regime, SmoothSign was adopted for all QAT experiments (consistent with Section 4.1).

(a) Forward pass: $\mathrm{sign}(x)$ and $\tanh(100x)$

(b) Backward pass: gradient $\frac{d}{dx}\tanh(100x)$

*Figure: Figure 8: The SmoothSign activation technique. (a) Functions relevant to the forward pass: SmoothSign uses the $\mathrm{sign}(x)$ function (actual forward pass). The plot also shows $\tanh(100x)$ , which serves as the basis for deriving the smooth proxy gradient shown in (b). (b) The smooth proxy gradient, $\frac{d}{dx}\tanh(100x)$ , utilized for the backward pass.*

*Figure: Table 6: Perplexity (WikiText-2) comparison between STE and SmoothSign for OPT-1.3B at various effective bits (BPW). Learning rate was 8e-5. Lower PPL is better.*

|
 | BPW
 | STE PPL
 | SmoothSign PPL
 | Difference (SmoothSign - STE)

|
 | 1.00
 | 20.322
 | 20.329
 | +0.007

|
 | 0.80
 | 21.344
 | 21.338
 | -0.006

|
 | 0.70
 | 22.100
 | 22.001
 | -0.099

|
 | 0.55
 | 23.528
 | 23.457
 | -0.071

|
 | 0.30
 | 29.058
 | 28.984
 | -0.074

|
 | 0.10
 | 60.401
 | 60.011
 | -0.390

### B.3 Grouped Query Attention

*Figure: Table 7: Ablation study on adjusting the latent dimension factor ( $r$ ) for key (K) and value (V) projection layers in Llama3-8B with GQA, targeting an overall effective bit of  0.1 BPW. Performance measured by perplexity (PPL, lower is better) on WikiText-2 and C4 validation sets, and average zero-shot accuracy (%) on common sense reasoning tasks (higher is better). The ’KV Factor’ indicates the multiplier applied to the K/V layers’ latent rank $r$ compared to the standard calculation for 0.1 BPW.*

|
 | KV Factor
 | Approx. BPW
 | WikiText-2 PPL
 | C4 PPL
 | Avg. Zero-shot Acc (%)

|
 | 1 $\times$ (Baseline)
 | $\sim$ 0.098
 | 25.80
 | 34.04
 | 44.85

|
 | 2 $\times$
 | $\sim$ 0.101
 | 25.22
 | 33.20
 | 44.92

|
 | 4 $\times$
 | $\sim$ 0.107
 | 24.93
 | 32.69
 | 44.70

|
 | 8 $\times$
 | $\sim$ 0.119
 | 25.31
 | 33.21
 | 44.87

Modern LLMs often use Grouped Query Attention (GQA) [64] or Multi-Query Attention (MQA) [67], where key (K) and value (V) projection layers have smaller output dimensions than query (Q) projections. Applying LittleBit at ultra-low bits (e.g., 0.1 BPW) to these GQA/MQA models can result in an extremely small latent dimension $r$ for K/V layers (e.g., $r\approx 20$ for Llama3-8B K/V layers at 0.1 BPW overall), potentially creating an information bottleneck and hindering QAT or performance.

To investigate, we performed an ablation on Llama3-8B ( $\sim$ 0.1 BPW target), increasing the latent rank $r$ for K/V layers by 2 $\times$ , 4 $\times$ , and 8x over the standard calculation (Appendix D), while other layers remained unchanged.
Table 7 shows that a 4x factor for K/V rank yielded the best PPL on WikiText-2 and C4, with minimal impact on overall BPW (e.g., $\sim$ 0.098 BPW to $\sim$ 0.107 BPW for 4 $\times$ ). Average zero-shot accuracy also remained competitive. An 8x factor showed diminishing returns.

Given the significant PPL improvement with the 4 $\times$ K/V rank factor (nearly 1 point on WikiText-2 vs. 1 $\times$ ) at a negligible BPW cost (<10% relative increase in effective bits for transformer layers, <4% of total parameters), this strategy was adopted for GQA/MQA models (Llama3-8B, Phi-4, QwQ-32B) in our main results. For these models, the latent rank $r$ for K and V projection layers was calculated based on a 4 $\times$ multiplier compared to the standard calculation for the target BPW, while all other layers used the standard rank calculation. This approach helps mitigate potential information bottlenecks in GQA/MQA layers under extreme quantization without significantly altering the target compression level.

## Appendix C Perplexity on C4 and PTB

To further assess LittleBit’s generalization, we evaluated its performance on additional benchmarks: the diverse C4 web text corpus and the distinct PTB news corpus (Table 8). Focusing on Llama2-7B/13B against STBLLM, the results on these datasets confirm the trends observed on WikiText-2. LittleBit consistently achieves lower perplexity than STBLLM at comparable sub-1-bit regimes. This performance advantage becomes more significant below 0.55 BPW, where STBLLM’s performance degrades considerably, particularly on PTB.

Notably, LittleBit maintains robust performance and stability even at the extreme 0.1 BPW level across all tested datasets. This consistent ability to maintain strong language modeling capabilities under severe compression highlights the generalizability of our approach, which integrates latent factorization, multi-scale compensation, and QAT. It reinforces LittleBit’s potential for effectively deploying capable LLMs in resource-limited environments.

*Figure: Table 8:
Perplexity (PPL) comparison on C4 and PTB validation sets for Llama2 models. Lower PPL indicates better performance. STBLLM [16] utilizes N:M sparsity (ratio in parentheses).
*

|
 | Model
 | Method
 | BPW
 | C4
 | PTB

|
 | Llama2-7B
 | FullPrecision
 | 16
 | 6.97
 | 37.91

|
 | STBLLM (6:8)
 | 0.80
 | 15.42
 | 2.4e3

|
 | STBLLM (4:8)
 | 0.55
 | 30.99
 | 527.56

|
 | STBLLM (2:8)
 | 0.30
 | 808.98
 | 2.3e4

|
 | LittleBit
 | 0.80
 | 11.77
 | 40.75

|
 | LittleBit
 | 0.55
 | 12.94
 | 50.53

|
 | LittleBit
 | 0.30
 | 14.78
 | 59.17

|
 | LittleBit
 | 0.10
 | 19.73
 | 83.09

|
 | Llama2-13B
 | FullPrecision
 | 16
 | 6.47
 | 50.94

|
 | STBLLM (6:8)
 | 0.80
 | 12.96
 | 165.51

|
 | STBLLM (4:8)
 | 0.55
 | 27.38
 | 424.82

|
 | STBLLM (2:8)
 | 0.30
 | 442.49
 | 1.7e3

|
 | LittleBit
 | 0.80
 | 10.60
 | 44.87

|
 | LittleBit
 | 0.55
 | 11.53
 | 53.52

|
 | LittleBit
 | 0.30
 | 13.07
 | 51.66

|
 | LittleBit
 | 0.10
 | 18.88
 | 87.59

## Appendix D Average Bits Per Weight Calculation

This section details the calculation of the average effective bits per weight (which we denote as $b$ in the following equations) for a linear layer implemented using the LittleBit architecture, including the parameters from the primary ( $\widehat{\mathbf{W}}_{\mathrm{pri}}$ ) and residual compensation ( $\widehat{\mathbf{W}}_{\mathrm{res}}$ ) pathways, as described in Sections 3.1 and 3.3. The calculation demonstrates how to determine the latent rank $r$ required to achieve a target value for $b$ within the range of approximately 0.1 to 1.0.

A LittleBit linear layer replaces the original weight matrix $\mathbf{W}\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$ with two parallel structures, each comprising:

- •

Two binary sign matrices: $\mathbf{U}_{\mathrm{sign}}\in\{\pm 1\}^{d_{\mathrm{out}}\times r}$ and $\mathbf{V}_{\mathrm{sign}}\in\{\pm 1\}^{d_{\mathrm{in}}\times r}$ . These require $1$ bit per element.

- •

Three FP16 scaling vectors: $\mathbf{h}\in\mathbb{R}^{d_{\mathrm{out}}}$ , $\mathbf{g}\in\mathbb{R}^{d_{\mathrm{in}}}$ , and $\bm{\ell}\in\mathbb{R}^{r}$ . These require $16$ bits per element.

Since both the primary and residual paths have this structure, the total number of bits required to store the parameters for a LittleBit layer is:

|
 |
 | Total Bits
 | $\displaystyle=2\times\left(\text{Bits for }\mathbf{U}_{\mathrm{sign}}+\text{Bits for }\mathbf{V}_{\mathrm{sign}}+\text{Bits for }\mathbf{h}+\text{Bits for }\mathbf{g}+\text{Bits for }\bm{\ell}\right)$
 |

|
 |
 |
 | $\displaystyle=2\times\left((d_{\mathrm{out}}\times r\times 1)+(d_{\mathrm{in}}\times r\times 1)+(d_{\mathrm{out}}\times 16)+(d_{\mathrm{in}}\times 16)+(r\times 16)\right)$
 |

|
 |
 |
 | $\displaystyle=2r(d_{\mathrm{out}}+d_{\mathrm{in}})+32(d_{\mathrm{out}}+d_{\mathrm{in}})+32r$
 |

Note: If Residual Compensation is not used, the initial factor of 2 should be removed from the calculation.

The average bits per weight, denoted as $b$ , is calculated by dividing the total bits by the number of parameters in the original FP16 weight matrix ( $d_{\mathrm{out}}\times d_{\mathrm{in}}$ ):

|
 |
 | $b=\frac{2r(d_{\mathrm{out}}+d_{\mathrm{in}})+32(d_{\mathrm{out}}+d_{\mathrm{in}})+32r}{d_{\mathrm{out}}\times d_{\mathrm{in}}}$
 |
 | (16)

To achieve a target value for $b$ , we rearrange Eq. 16 to solve for the required latent rank $r$ :

|
 |
 | $r=\frac{(b\times d_{\mathrm{out}}\times d_{\mathrm{in}})-32(d_{\mathrm{out}}+d_{\mathrm{in}})}{2(d_{\mathrm{out}}+d_{\mathrm{in}})+32}$
 |
 | (17)

Since $r$ must be an integer, we typically round the calculated value to the nearest suitable integer. This chosen integer $r$ then determines the actual value of $b$ achieved, which will be very close to the target value.

- •

Example 1: Linear Layer ( $$4,096$\times$4,096$$ )
Let $d_{\mathrm{out}}=$4,096$$ and $d_{\mathrm{in}}=$4,096$$ .
The original number of parameters is $$4,096$\times$4,096$=$16,777,216$$ .
 $d_{\mathrm{out}}+d_{\mathrm{in}}=$8,192$$ .

For a target $b\approx 0.55$ :
Using Eq. 17 with $b=0.55$ :

|
 |
 | $r=\frac{(0.55\times$16,777,216$)-32($8,192$)}{2($8,192$)+32}\approx 546$
 |

We choose integer $r=546$ .
The actual value of $b$ for $r=546$ , using Eq. 16, is:

|
 |
 | $b=\frac{2(546)($8,192$)+32($8,192$)+32(546)}{$16,777,216$}\approx 0.5498$
 |

- •

Example 2: Linear Layer ( $$4,096$\times$11,008$$ ) Let $d_{\mathrm{out}}=$4,096$$ and $d_{\mathrm{in}}=$11,008$$ .
The original number of parameters is $$4,096$\times$11,008$=$45,090,816$$ .
 $d_{\mathrm{out}}+d_{\mathrm{in}}=$15,104$$ .

For a target $b\approx 0.1$ :
Using Eq. 17 with $b=0.1$ :

|
 |
 | $r=\frac{(0.1\times$45,090,816$)-32($15,104$)}{2($15,104$)+32}\approx 133$
 |

We choose integer $r=133$ .
The actual value of $b$ for $r=133$ , using Eq. 16, is:

|
 |
 | $b=\frac{2(133)($15,104$)+32($15,104$)+32(133)}{$45,090,816$}\approx 0.0999$
 |

By following this procedure for each linear layer in the model, we can select appropriate ranks ( $r$ ) to achieve a desired overall value for $b$ (average bits per weight), thereby controlling the trade-off between model compression and performance. The specific ranks used to achieve the average bits per weight (BPW) figures reported in the main paper (e.g., in Tables 1 and 3) are determined using this calculation methodology for $b$ for each layer type within the respective models.

## Appendix E Pruning versus SVD

To select a base compression method to achieve sub-0.5 BPW, we compared pruning (Wanda [22]) with SVD-based compression (SVD-LLMv2 [21]) on Llama-7B ( Table 9). SVD showed superior robustness at extreme compression ratios (e.g., 25% parameter retention).

Consequently, low-rank factorization was chosen to design LittleBit due to its performance resilience and the deployment advantages of dense factorized matrices over sparse pruned models (which may require specialized hardware or overhead). Although SVD-LLMv2 itself performs well, we found that initializing LittleBit’s Dual-SVID with vanilla SVD was sufficient. The subsequent QAT process effectively recovered performance, rendering the added complexity of SVD-LLMv2 for initialization unnecessary. Accordingly, we employ the simpler vanilla SVD within our Dual-SVID initialization procedure.

*Figure: Table 9: Perplexity (WikiText-2) comparison between Pruning (Wanda) and SVD-based compression (SVD-LLMv2) on Llama-7B at different parameter retention ratios. Lower perplexity is better. FP16 PPL is 5.68.*

|
 | Remaining Parameter Ratio
 | Pruning [22]
 | SVD [21]

|
 | 50%
 | 8.7
 | 13.6

|
 | 37.5%
 | 46.0
 | 20.1

|
 | 25%
 | 1842.7
 | 52.6

## Appendix F Analysis of Generated Samples

The below generated samples illustrate qualitative differences as the BPW of the LittleBit model decreases. Two main observations are:

- 1.

Impact on Specificity, Factual Detail, and Coherence:
As the effective bits decrease from 0.8 BPW towards 0.1 BPW, the generated text exhibits significant changes in detail, factual accuracy, and coherence. At 0.8 BPW, the Mona Lisa sample, while attempting to provide details (e.g., "complex facial features"), becomes verbose and repetitive, and anachronistically refers to Da Vinci and the Mona Lisa in the context of the "20th century." The Turing sample at this effective bit, however, offers a standard, factually sound definition. As the effective bit drops to 0.55 BPW, the Mona Lisa sample loses coherence significantly, expressing uncertainty about the painting’s location and bizarrely shifting to a help-seeking persona ("Could you help me find the Mona Lisa painting…"). The Turing sample at 0.55 BPW begins to broaden its definition, losing some precision but remaining somewhat related.

- 2.

Degradation in Factual Accuracy and Relevance at Extreme Compression:
Extreme compression, especially at 0.1 BPW, leads to severe degradation in factual accuracy and relevance, as vividly demonstrated by both samples. The Mona Lisa sample at 0.1 BPW starts with a generally acceptable statement (e.g., "iconic masterpiece") but quickly devolves into hallucination, referring to the Mona Lisa as a "statue" and associating it with "Milan’s fashion world." Similarly, the Turing sample at 0.1 BPW deviates drastically from the core definition of computer science, incorrectly defining it as "the study of the theory and methods of how the human mind and the brain work" and "the science of mind." These examples underscore that while LittleBit might maintain superficial grammatical structure at very low bits, the ability to recall and generate accurate, relevant factual information is compromised, often leading to the generation of plausible-sounding but nonsensical statements.

### Sample 1: Mona Lisa

Prompt: Leonardo da Vinci painted the Mona Lisa. The Mona Lisa is

|
 | BPW
 |

Generated Output (Phi-4 Model)

|
 | \endfirsthead      Sample 1: Mona Lisa – Continued

|
 | BPW
 |

Generated Output (Phi-4 Model)

|
 | \endhead        Continued on next page


 Experimental support, please
 view the build logs
 for errors. Generated by


 L
 A
 T
 E

 xml

 .




## Instructions for reporting errors



We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile
 support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the
 methods listed below:


- Click the "Report Issue" () button, located in the page header.



Tip: You can select the relevant text first, to include it in your report.


Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we
 may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability
 should not be a barrier to accessing research. Thank you for your continued support in championing open access for
 all.


Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.





 We gratefully acknowledge support from
 our major funders,
 member institutions, ,
 and all contributors.


 About
 ·
 Help
 ·
 Contact
 ·
 Subscribe
 ·
 Copyright
 ·
 Privacy
 ·
 Accessibility
 ·
 Operational Status (opens in new tab)




 Major funding support from
