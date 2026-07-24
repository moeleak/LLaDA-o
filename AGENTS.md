# Repository Workflow

- Work directly on `main`; do not create feature or topic branches.
- Commit every repository modification using the Conventional Commits format.
- Push completed commits directly to `main`.
- Before attributing a quality change to YaRN, run the 100-sample isolation
  benchmark. Both arms must use the same sample IDs, checkpoint-native resized
  image, native multimodal positions, prompt, seed, decoding parameters,
  16K-resident KV capacity, and disabled KV compression; only RoPE scaling and
  its advertised maximum position may differ.
- Do not launch more than 100 benchmark samples unless the user explicitly
  authorizes the larger run.
- A causal true-long YaRN comparison must keep full-page tiles, sequential
  positions above 16K, the 128K model limit, 65,536-token resident KV
  capacity, sample IDs, seed, prompt, and decoding identical. Compare
  unscaled 128K with YaRN 128K; native-resized original 16K versus full-page
  YaRN is a deployment comparison, not a controlled YaRN experiment.
