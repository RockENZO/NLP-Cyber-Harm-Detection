# Model artifacts

The checkpoint and tokenizer directories in this folder are historical research
artifacts. Large files use Git LFS; a normal checkout can contain pointer files
instead of usable weights. Follow the repository README to download only the
artifacts required by a particular demo. The current word/character classifier
study creates its own artifacts and does not require these neural checkpoints.

## Removed incomplete FLAN-T5 reference

Earlier revisions tracked `models/flan-t5-base` as a Git submodule entry pointing
to commit `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`, without a corresponding
`.gitmodules` URL. This entry did not supply model files and made recursive
checkouts fail, including GitHub's Pages build. It was removed from the current
tree; the original reference remains in Git history. No weights were deleted
from this entry, and its upstream repository and training provenance have not
been established.

No current executable code loads that local path. The historical T5 training
script, `training/unified_t5_fraud.py`, defaults to the separate model identifier
`google/flan-t5-small`. That upstream base model is not a trained scam classifier
and must not be substituted for a fine-tuned checkpoint when reporting results.
