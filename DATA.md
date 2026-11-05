# Corpus sources

The original text snapshots, including source notices, are retained rather than replacing them with freshly downloaded copies. SHA-256 values are recorded in run provenance and the release manifest.

| Work | Project Gutenberg ID | Audited-study use |
|---|---|---|
| Hamlet | [1524](https://www.gutenberg.org/ebooks/1524) | First 80% train, next 10% development |
| Alice's Adventures in Wonderland | [11](https://www.gutenberg.org/ebooks/11) | First 80% train, next 10% development |
| Moby-Dick | [2701](https://www.gutenberg.org/ebooks/2701) | First 80% train, next 10% development |
| Frankenstein | [84](https://www.gutenberg.org/ebooks/84) | Complete final evaluation book |
| The Adventures of Sherlock Holmes | [1661](https://www.gutenberg.org/ebooks/1661) | Complete final evaluation book |

The audited files are in `experiments/rigorous/data/`. Legacy data in `experiments/data/` also includes the character-level tiny Shakespeare snapshot from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`; it was used in earlier exploration, not as an additional audited final book.

Normalization and split rules are in `experiments/rigorous/PROTOCOL.md`. Original source texts had already informed the earlier project, so the development sources are not an independent project-level holdout. The final books were excluded from primary training and model selection. Later local-only/context checks are explicitly exploratory.
