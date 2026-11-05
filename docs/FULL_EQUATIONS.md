# ModalStripe: implemented equations

This is the model used in the saved text experiments. Batch indices are omitted. Positions start at zero. Tokens are ASCII characters, not word tokens. There are two blocks, width $d=64$, four attention heads of width $d_h=16$, local window $w=8$, and $R=8$ modal poles. The vocabulary has 128 symbols. All vector products written $\odot$ are channelwise. $\sigma$ is the sigmoid.

## 1. Input and output

For input character $x_t$, let $e_t=E[x_t]\in\mathbb R^{64}$, with $e_{-1}=0$ at a window boundary. The input is

$$
h_t^{(0)}=e_t+W_{\mathrm{prev}}e_{t-1}+\operatorname{PE}(t). \tag{1}
$$

The sinusoidal position encoding is

$$
\operatorname{PE}_{2i}(t)=\sin\!\left(t\,10000^{-2i/64}\right),\qquad
\operatorname{PE}_{2i+1}(t)=\cos\!\left(t\,10000^{-2i/64}\right),\quad i=0,\ldots,31. \tag{2}
$$

After the two blocks,

$$
\ell_t=W_{\mathrm{head}}\operatorname{LN}_{\mathrm{out}}(h_t^{(2)})\in\mathbb R^{128},\qquad
P(x_{t+1}=v\mid x_{0:t})=\operatorname{softmax}(\ell_t)_v. \tag{3}
$$

The output projection is untied from the embedding matrix. Next-character training minimizes

$$
\mathcal L=-\frac1{N_{\mathrm{target}}}\sum_{(x,t)\in\mathcal D_{\mathrm{target}}}
\log P(x_{t+1}\mid x_{0:t}). \tag{4}
$$

Here $\mathcal D_{\mathrm{target}}$ indexes the scored next-character targets in a batch or evaluation split. Evaluation averages the two final-book losses equally before seed aggregation and exponentiation.

## 2. Attention, including the local control

Given normalized inputs $a_t\in\mathbb R^{64}$, each head $h=1,\ldots,4$ computes

$$
q_t^{(h)}=W_Q^{(h)}a_t,\qquad
k_t^{(h)}=W_K^{(h)}a_t,\qquad
v_t^{(h)}=W_V^{(h)}a_t\in\mathbb R^{16}. \tag{5}
$$

With a causal mask $M_{t,j}$,

$$
A_{t,j}^{(h)}=
\frac{\exp\!\left(q_t^{(h)\top}k_j^{(h)}/\sqrt{16}+M_{t,j}\right)}
{\sum_{i=0}^{T-1}\exp\!\left(q_t^{(h)\top}k_i^{(h)}/\sqrt{16}+M_{t,i}\right)}. \tag{6}
$$

For global attention, $M_{t,j}=0$ for $0\le j\le t$ and $-\infty$ otherwise. For window-8 local attention,

$$
M_{t,j}^{\mathrm{local}}=
\begin{cases}
0,&\max(0,t-7)\le j\le t,\\
-\infty,&\text{otherwise}.
\end{cases} \tag{7}
$$

The head outputs and multi-head output are

$$
b_t^{(h)}=\sum_{j=0}^{T-1}A_{t,j}^{(h)}v_j^{(h)},\qquad
\operatorname{MHA}(a)_t=W_A[b_t^{(1)};b_t^{(2)};b_t^{(3)};b_t^{(4)}]. \tag{8}
$$

The implementation uses a single bias-free QKV projection and then reshapes it into heads. The separate matrices above denote its slices. Global decoding retains earlier keys and values; local decoding retains at most eight positions.

## 3. Modal write and read gates

The modal mixer receives the same width-64 normalized representation:

$$
a_t=\operatorname{LN}_1(h_t),\qquad
[\nu_t;\eta_t;\rho_t]=W_P a_t,\quad W_P\in\mathbb R^{192\times64}. \tag{9}
$$

Writing $W_P=[W_v;W_w;W_r]$, its gated write and read are

$$
u_t=\nu_t\odot\sigma(\eta_t)
=(W_v a_t)\odot\sigma(W_w a_t),\qquad
g_t=\sigma(\rho_t)=\sigma(W_r a_t). \tag{10}
$$

These gates depend on the input. The modal kernel below depends on lag and learned parameters.

## 4. Damped modes and the causal kernel

For mode $r=1,\ldots,8$, the parameters are a real log-decay $\theta_r$, a real frequency $\omega_r$, and a complex residue vector $c_r\in\mathbb C^{64}$:

$$
\alpha_r=\exp(\theta_r)>0,\qquad
p_r=\exp(-\alpha_r+\mathrm{i}\omega_r),\qquad |p_r|<1. \tag{11}
$$

Each pole is shared across channels. The channel-specific kernel is

$$
K_{\ell,k}=\operatorname{Re}\sum_{r=1}^{8}c_{r,k}p_r^{\ell},
\qquad \ell\ge0,\quad k=1,\ldots,64. \tag{12}
$$

Its real form, as computed in the implementation, is

$$
K_{\ell,k}=\sum_{r=1}^{8}e^{-\alpha_r\ell}
\left[\operatorname{Re}(c_{r,k})\cos(\omega_r\ell)
-\operatorname{Im}(c_{r,k})\sin(\omega_r\ell)\right]. \tag{13}
$$

The modal convolution and output are

$$
z_{t,k}=\sum_{j=0}^{t}K_{t-j,k}u_{j,k},\qquad
m_t=W_o(z_t\odot g_t). \tag{14}
$$

The residual network's gain is not bounded merely because its individual poles have magnitude below one.

## 5. Full-sequence evaluation by FFT

For sequence length $T$, construct $K_0,\ldots,K_{T-1}$ and choose

$$
N=2^{\lceil\log_2(2T-1)\rceil}. \tag{15}
$$

Pad $U,K\in\mathbb R^{T\times64}$ to length $N$ along the sequence dimension. Then

$$
Z=\left[\operatorname{irFFT}_N\left(
\operatorname{rFFT}_N(U)\odot\operatorname{rFFT}_N(K)
\right)\right]_{0:T}. \tag{16}
$$

The transform product is channelwise. The slice retains the first $T$ positions. Padding to at least $2T-1$ prevents circular wraparound into those positions. This is the full-sequence path used for training and prefill output computation.

## 6. Decoding by recurrence

Starting from $S_{-1}=0\in\mathbb C^{8\times64}$,

$$
s_{t,r,k}=p_r s_{t-1,r,k}+u_{t,k},\qquad
z_{t,k}=\operatorname{Re}\sum_{r=1}^{8}c_{r,k}s_{t,r,k}. \tag{17}
$$

Expanding the recurrence gives

$$
s_{t,r,k}=\sum_{j=0}^{t}p_r^{t-j}u_{j,k},\qquad
z_{t,k}=\sum_{j=0}^{t}\left(\operatorname{Re}\sum_{r=1}^{8}c_{r,k}p_r^{t-j}\right)u_{j,k}. \tag{18}
$$

Equation (18) is Equation (14). This uses real-valued writes $u_j$ and zero initial state. Prefill initializes the decoding state by the weighted sum

$$
s_{T-1,r,k}=\sum_{j=0}^{T-1}p_r^{T-1-j}u_{j,k}. \tag{19}
$$

The implementation builds this state separately from its FFT computation of sequence outputs. It stores complex64 states: $8\times64\times8=4096$ bytes per example. This count covers only the modal state.

## 7. The complete first block

Let $H$ denote the sequence entering the first block. All additions below are residual updates:

$$
\begin{aligned}
H^{m}&=H+\operatorname{Modal}(\operatorname{LN}_1(H)),\\
H^{a}&=H^{m}+\operatorname{LocalMHA}_8(\operatorname{LN}_1(H^{m})),\\
H^{(1)}&=H^{a}+\operatorname{FFN}_1(\operatorname{LN}_2(H^{a})).
\end{aligned} \tag{20}
$$

The two occurrences of $\operatorname{LN}_1$ share parameters, but operate on different inputs. Local attention follows the modal residual update. The FFN uses a separate normalization.

The second block is global causal attention:

$$
\begin{aligned}
G&=H^{(1)}+\operatorname{GlobalMHA}(\operatorname{LN}_3(H^{(1)})),\\
H^{(2)}&=G+\operatorname{FFN}_2(\operatorname{LN}_4(G)).
\end{aligned} \tag{21}
$$

Each FFN is

$$
\operatorname{FFN}_b(x)=W_{2,b}\operatorname{GELU}(W_{1,b}x). \tag{22}
$$

All linear projections are bias-free. LayerNorm has learned scale and offset. For approximately matched parameters, the modal/local block's FFN width is 124 and the global block's width is 261. The fixed-FFN control uses 256 in each block.

## 8. What changes relative to full attention

Full attention uses a global-attention residual update and an FFN in each of two blocks. Local + global changes only the first attention mask. ModalStripe adds the modal residual update before that first local-attention operation, as in Equation (20).

The modal operator has no query–key score matrix. It compresses earlier gated writes into eight complex states per channel. The complete model still has one global-attention block. That block retains a KV cache growing with sequence length and has quadratic full-sequence attention work.

The text experiments did not pass frequency-count features to the model. Two inherited inactive frequency parameters are excluded from these equations because they do not contribute to the computed outputs. No word-frequency routing, mixture of experts, multi-token prediction, or FP8 path was implemented in these runs.

## Dimensions

| Quantity | Shape (one example) |
|---|---|
| Input representations $H$, gated writes $U$, reads $G_{\mathrm{read}}$ | $T\times64$ |
| Modal projection $W_P$ | $192\times64$ |
| $W_v,W_w,W_r,W_o,W_{\mathrm{prev}},W_A$ | $64\times64$ |
| Per-head $W_Q^{(h)},W_K^{(h)},W_V^{(h)}$ | $16\times64$ |
| Modal poles $p$ | $8$ complex entries |
| Modal residues $C$ | $8\times64$ complex entries |
| Modal state $S_t$ | $8\times64$ complex entries |
| Kernel $K$ | $T\times64$ real entries |
| Embedding matrix $E$ | $128\times64$ |
| Output projection $W_{\mathrm{head}}$ | $128\times64$ |

The diagrams use blue for representations, cream for operations and green for cached state. Dashed blue lines mark module boundaries or expansion links. Every component above maps to the saved implementation in `experiments/rigorous/model.py`; the experiment construction is in `study.py`.
