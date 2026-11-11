# Geospatial land-cover capstone plan

## Goal
Classify image tiles into land-cover classes and compare a convolutional baseline with a transformer-based vision model. This aligns with the public capstone description; it does not reproduce the platform's learner-specific assignment or claim submitted results.

## Data contract
Use an openly licensed, documented satellite image dataset. Arrange RGB chips by split under `data/{train,validation,test}/{class_name}/`. Keep neighboring tiles from the same geographic region in one split to reduce spatial leakage. Record sensor, date range, ground resolution, class definitions, licensing, and geographic coverage before training.

## Pipeline
1. Validate class counts and inspect a stratified sample.
2. Apply training-only augmentation (crop/flip/brightness) and fixed normalization.
3. Train a small CNN baseline; then compare a pretrained vision transformer only if its weights and license are documented.
4. Select checkpoints on validation macro F1. Evaluate once on held-out geographic regions.
5. Report confusion matrix, per-class recall, macro F1, calibration, and representative errors with coordinates removed.
6. Document compute, data version, random seed, limitations, and intended non-operational use.

## Report template
- Dataset and license:
- Geographic split method:
- Class distribution:
- Baseline and model configuration:
- Validation selection rule:
- Test metrics (measured only after execution):
- Error patterns and known limits:
- Ethical and environmental risks:
