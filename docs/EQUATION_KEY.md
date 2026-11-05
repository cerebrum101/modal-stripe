# Equation key for the redesigned schematics

The diagrams describe the saved implementation. Width $d=64$, modes $R=8$, local window $w=8$. Batch indices are omitted. A row $S_{t,r,:}$ has $d$ complex entries. All weights and poles are fixed during a decoding sequence. Gates depend on the current normalized input.

## Figure 1: intervention and controls

A residual module is

$$
\mathcal R_f(H)=H+f(\operatorname{LN}_f(H)). \tag{1}
$$

The first-block alternatives are

$$
\begin{aligned}
B_1^{\mathrm{full}} &= \mathcal R_F\circ\mathcal R_G,\\
B_1^{\mathrm{local+global}} &= \mathcal R_F\circ\mathcal R_L,\\
B_1^{\mathrm{ModalStripe}} &= \mathcal R_F\circ\mathcal R_L\circ\mathcal R_M.
\end{aligned} \tag{2}
$$

Composition acts right to left; the figure reads top to bottom. The symbols denote module forms, not shared model weights. In ModalStripe, $\operatorname{LN}_M$ and $\operatorname{LN}_L$ reuse one set of parameters and normalize different intermediate representations. The FFN has its own normalization. Hidden FFN widths differ under parameter matching; this diagram does not claim identical parameter allocations.

Every primary architecture retains the second block

$$
B_2=\mathcal R_{F,2}\circ\mathcal R_{G,2}. \tag{3}
$$

$G$ is four-head global causal attention; $L$ is four-head causal attention restricted to $\max(0,t-7)\le j\le t$. The input stem and output head are defined in the full equation sheet. No branch in Figure 1 is executed in parallel with another: the lower columns are alternative model definitions.

## Figure 2: one modal decoding step

For normalized input $a_t\in\mathbb R^d$, define the write and read gates

$$
u_t=\psi(a_t)=(W_v a_t)\odot\sigma(W_w a_t),\qquad
g_t=\rho(a_t)=\sigma(W_r a_t). \tag{4}
$$

The learned transition is diagonal across modes:

$$
P=\operatorname{diag}(p_1,\ldots,p_R),\qquad
p_r=\exp[-\exp(\theta_r)+\mathrm{i}\omega_r]. \tag{5}
$$

The fixed broadcast module is $\mathcal B(u)=\mathbf 1_R u^{\mathsf T}$. It has no learned parameters. The update, readout and output are

$$
\begin{aligned}
S_t &= P S_{t-1}+\mathbf 1_R u_t^{\mathsf T},& S_{-1}&=0,\\
z_{t,k} &= \mathcal C(S_t)_k
=\operatorname{Re}\sum_{r=1}^{R}c_{r,k}S_{t,r,k},\\
m_t &= W_o(z_t\odot g_t).
\end{aligned} \tag{6}
$$

Both summands in the state update have shape $R\times d$. The outer product broadcasts each input channel to every mode; each mode then evolves with a different pole. The readout multiplies by complex residues without conjugation. $P$ is $R\times R$; $C=[c_{r,k}]$ is $R\times d$; $W_v,W_w,W_r,W_o$ are $d\times d$.

The delay in Figure 2 is temporal storage, not a trainable transformation. At batch one, complex64 storage for this state is $8\times64\times8=4096$ bytes. This does not count attention caches, activations, parameters or peak process memory.

## Equivalent sequence evaluation

Expanding the zero-initial-state recurrence gives

$$
z_{t,k}=\sum_{j=0}^{t}K_{t-j,k}u_{j,k},\qquad
K_{\ell,k}=\operatorname{Re}\sum_{r=1}^{R}c_{r,k}p_r^\ell. \tag{7}
$$

For $N=2^{\lceil\log_2(2T-1)\rceil}$,

$$
Z=\left[\operatorname{irFFT}_N\!\left(
\operatorname{rFFT}_N(U)\odot\operatorname{rFFT}_N(K)
\right)\right]_{0:T}. \tag{8}
$$

Zero-padding computes the same causal convolution without circular wraparound. This equivalence concerns the modal operator. It does not equate modal memory to attention or make the full model linear. The full model retains global attention and a growing KV cache.
