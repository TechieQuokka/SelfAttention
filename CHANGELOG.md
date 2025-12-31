# Changelog

## [2025-12-31] - Code Quality Improvements

### Fixed
- **Label Smoothing Consistency**: Fixed inconsistency between training and evaluation
  - Training now uses `label_smoothing=0.1` (regularization)
  - Validation/Test use `label_smoothing=0.0` (fair evaluation)
  - This resolves the discrepancy where validation loss (6.56) was artificially higher than test loss (5.30)

### Changed
- `train.py`:
  - Modified `_compute_loss()` to accept `training` parameter
  - Training losses use label smoothing from config
  - Validation losses use no label smoothing for fair comparison

- `evaluate.py`:
  - Explicitly set `label_smoothing=0.0` with documentation

- `config.py`:
  - Added `label_smoothing` parameter to `TrainingConfig`
  - Default value: 0.1 (applied during training only)

### Impact
- **Before**: Validation showed inflated perplexity (708) due to label smoothing
- **After**: Validation and test will show consistent, comparable metrics (~200 perplexity)
- Future training runs will report accurate validation metrics

### Technical Details
```python
# Training (with regularization)
loss = cross_entropy(..., label_smoothing=0.1)

# Validation/Test (fair evaluation)
loss = cross_entropy(..., label_smoothing=0.0)
```

This follows PyTorch best practices:
- Label smoothing is a training-time regularization technique
- Evaluation should measure true model performance without regularization
