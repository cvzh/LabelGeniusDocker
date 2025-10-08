# LabelGenius

Zero-setup labeling via CLI + Docker (CLIP zero-shot, fine-tuned CLIP, GPT).

## Quick Start (Docker CPU – works everywhere)
```bash
docker pull ghcr.io/<YOU>/labelgenius:cpu
docker run --rm -v "$PWD:/work" -w /work ghcr.io/<YOU>/labelgenius:cpu \
  clip-zero --text_path Demo_data/D1_1_first20.csv \
            --mode text \
            --labels politics sports business tech \
            --save_csv outputs/results.csv
