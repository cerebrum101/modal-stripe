# Exploratory local-only efficiency control

Added after primary test outcomes were inspected. This is explicitly post hoc, not part of the original confirmatory comparison. It addresses a remaining strong-baseline concern: short character prediction may require little global memory, and local-only attention already performed well in the legacy experiment.

Use two window8 local-attention blocks with the same shared stem, approximately matched parameter budget, fixed ASCII128 alphabet and data splits. Select rate from the original three values with a 400-step seed0 development-only sweep. Train seeds0/1/2 for1800 updates at context128/batch16. Evaluate the same final books and context blocks without using final scores for tuning. Keep all pilot results. Report independently and do not use it to relabel the original test as untouched for this adaptive architecture comparison.
