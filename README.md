# BERT fine-tuning for Target and Frame Identification in Frame Semantic Parsing

**For usage instructions refer to [`FrameTrigger.ipynb`](FrameTrigger.ipynb)**

Fine-tuned on the samples sentences coming from the FrameNet_v17 dataset.

Best results with `bert-base-cased`:

| Task  | FN 1.7 Dev  | FN 1.7 Test  |
| -------------  | ------------- | ------------- |
| Target Identification  | 0.89         | 0.86         |
| Frame Identification  | 0.86         | 0.83         |