# Research and communication review

Scope: the earlier standalone schematics and equation notes, checked against the frozen model. This is an internal review, not external peer review or an endorsement by Yann LeCun. The paper and experiments are unchanged.

## Perspective

LeCun's 2022 proposal separates modules by function and discusses representations that retain information needed for prediction. That provides a useful question here: what information enters, what persists, and what is available at readout? Applying this question to a diagram is our interpretation. His proposal is about autonomous intelligence and world models; ModalStripe's character prediction experiment is neither a world model nor a JEPA.

Primary source: https://ai.meta.com/blog/yann-lecun-advances-in-ai-research/ (23 February 2022).

## Findings before redesign

1. **Wrong levels in one picture.** A model backbone, normalization details, a per-step recurrence and implementation cache accounting appeared at the same visual level. The result was long traces without a clear claim. Separate the intervention from the recurrent operator.
2. **Controls were absent from the explanatory story.** The relevant question is whether adding modal computation helps beyond removing one global-attention block. Draw all three first-block choices. Cache reduction by itself does not identify a benefit from modal memory.
3. **The state transition was not a state-transition diagram.** A box containing a recurrence does not show which quantity persists between steps. Draw a one-step delay and a feedback path, with the state dimensions explicit.
4. **Learned parameters looked like activations.** Pole/residue boxes feeding state boxes suggested time-varying inputs. Put fixed-after-training parameters in operator definitions. Draw runtime tensors as arrows.
5. **Attention and modal memory were treated as interchangeable abstractions.** Attention has query-dependent selection over stored positions. The modal branch accumulates gated writes through a lag-stationary kernel and has an input-dependent output gate. Convolution/recurrence equivalence holds for the modal branch; there is no equivalence to arbitrary attention.
6. **Linear dynamics were easy to confuse with a linear model.** The state dynamics are linear in gated writes. Input gates, normalization, attention and FFNs make the complete network nonlinear.
7. **The global block was visually discounted.** Keep it prominent in the architecture. Only the modal cell has fixed state size; whole-model cache and attention work still grow with context.
8. **Implementation diagrams did not establish a research contribution.** Eight modes and stable poles alone do not establish semantic representation, associative recall or long-context generalization. Figures should explain the tested mechanism, not imply those outcomes.
9. **Visual grammar lacked discipline.** Arrow direction changed, operator labels became paragraphs, dashed boxes lacked one stable meaning, and parameter/data/state edges were mixed. Use left-to-right model flow, top-to-bottom control blocks, explicit residual notation, one state-feedback loop, short labels, Helvetica and standard math.

## Method-level assessment

The implemented modal branch is a gated diagonal complex state-space operator. Its poles are learned, shared across channels and fixed during a sequence; its writes and output gate depend on the input. That is a specific inductive bias, not evidence that the state learns a semantic world representation. Spectral evaluation alone does not establish a new FNO or sequence-model family.

The finite state superposes earlier writes. Query-conditioned selection over stored token positions is absent from this branch. Exact retrieval therefore needs separate evidence; low character perplexity cannot establish it. The retained global block also prevents a bounded-memory claim for the complete network.

The empirical contribution remains a controlled small-model comparison. The equal-character result is more favorable than the timing-calibrated result, and the local-plus-global control accounts for almost all cache reduction. The figures should expose that control, rather than visually implying unique modal efficiency. The reused normalization parameters are an implementation choice worth documenting, not a general principle of representation learning.

Introducing a JEPA or world-model objective would change the research question and require new experiments. A diagram should not borrow those ideas as visual credibility for a next-character model.

## Redesign decisions

Figure 1 answers where the intervention occurs and which architecture controls were tested. Residual modules are written at one consistent level. The whole-model backbone retains the second global block. The control panel shows alternatives, not parallel branches of an ensemble.

Figure 2 answers what the modal state stores and how one decoding step updates and reads it. A delay makes causality explicit. Writes are broadcast across modes before addition, so both summands have shape R × d. Parameters appear in the equation key, not as runtime data sources.

The FFT path is a compact identity on the equation sheet. It computes the same modal operator over a sequence; it is not a separate learned representation or another architecture. The older attention-versus-modal flowchart is dropped because it duplicated the cell without clarifying the tested intervention.

## Remaining scientific limits

This redesign does not resolve limited scale, limited tuning, budget sensitivity, sparse context diagnostics or the lack of GPU evidence. It adds no experiment and makes no claim of endorsement or human authorship. The existing 22 equation groups remain available; a shorter equation key gives the symbols needed for these two figures.
