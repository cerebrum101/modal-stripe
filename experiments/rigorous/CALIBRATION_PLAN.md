# Supplemental timing calibration

Specified while the primary study is still training, before final-book outcomes are inspected. Actual accumulated-time runs may be distorted by host-load changes between models. Retain those results.

After measure.py records interleaved training-step costs, compute striped/full step-time ratios for corresponding repeat and seed at context128. Use the median of the 60 ratios, and train full attention for ceil(1800 * median ratio) updates with the same selected learning rate and cosine schedule. Three seeds; training/development sources unchanged. The decision uses timing alone, never final-book quality. This is a calibrated time-equivalent update budget, not an exact FLOP budget. Report it separately from equal tokens and literal equal elapsed time.
