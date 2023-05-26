# 1.1) Train Masker
python3 rationalizers train --config configs/masker/imdb_sparsemap_30p.yaml --seed 0
# >>> experiments/masker_imdb_sparsemap_30p/versionjkexuek0/checkpoints/epoch=1.ckpt

# 1.2) Train Editor
python3 rationalizers train --config configs/editor/imdb_sparsemap_30p.yaml --seed 0
# >>> experiments/editor_imdb_sparsemap_30p/version3rqzafjw/checkpoints/epoch=0.ckpt

# 1.3) Extract counterfactuals
python3 scripts/get_edits.py \
  --ckpt-name "sparsemap_30p" \
  --ckpt-path "experiments/editor_imdb_sparsemap_30p/version3rqzafjw/checkpoints/epoch=0.ckpt" \
  --dm-name "revised_imdb" \
  --dm-dataloader "test" \
  --num-beams 15

# 1.4) Extract rationales (only)
python3 scripts/get_rationales.py \
    --ckpt-name "sparsemap_30p" \
    --ckpt-path "experiments/editor_imdb_sparsemap_30p/version3rqzafjw/checkpoints/epoch=0.ckpt" \
    --dm-name "revised_imdb" \
    --dm-dataloader "test"


# 2.0) Before proceeding, we need to extract counterfactuals for all training examples (this may take a while)
python3 scripts/get_edits.py \
  --ckpt-name "sparsemap_30p" \
  --ckpt-path "experiments/editor_imdb_sparsemap_30p/version3rqzafjw/checkpoints/epoch=0.ckpt" \
  --dm-name "imdb" \
  --dm-dataloader "train" \
  --num-beams 15

# 2.1) Train rationalizers with data augmentation
python3 rationalizers train --config configs/data_augmentation/imdb_sparsemap_30p.yaml --seed 1

# 2.2) Train rationalizers with agreement regularization
python3 rationalizers train --config configs/agreement_regularization/imdb_sparsemap_30p.yaml --seed 1


# Optional: Train student for computing forward simulability
python3 scripts/forward_simulation.py \
    --student-type "bow" \
    --train-data "data/edits/revised_imdb_test_beam_15_sparsemap_30p.tsv" \
    --test-data "data/edits/revised_imdb_test_beam_15_sparsemap_30p.tsv" \
    --batch-size 16 \
    --epochs 10 \
    --seed 0
# >>> lightning_logs/version_2/checkpoints/epoch=4-step=140.ckpt


# Optional: Get counterfactuals for another masker (required for counterfactual simulation)
python3 scripts/get_edits.py \
  --ckpt-name "sparsemap_30p" \
  --ckpt-path "experiments/editor_imdb_sparsemap_30p/version3rqzafjw/checkpoints/epoch=0.ckpt" \
  --ckpt-path-factual "experiments/masker_imdb_sparsemap_30p/versionjkexuek0/checkpoints/epoch=1.ckpt" \
  --dm-name "revised_imdb" \
  --dm-dataloader "test" \
  --num-beams 15
