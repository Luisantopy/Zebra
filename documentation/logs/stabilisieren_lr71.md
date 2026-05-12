config = {
            "optimizer": "sgd",
            "lr": 0.071,
            "alpha": 0.60,
            "min_recall": 0.39,
            "momentum": 0.87,
        }

-- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_152349
✅ Best model saved
⏹ Early stopping after epoch 3
📦 Final model saved: trained_models/exp_20260406_152349/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.1529 | Recall: 0.7902 | Precision: 0.0846

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_153123
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260406_153123/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.5821 | Recall: 0.5455 | Precision: 0.6240

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_154212
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_154212/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 30 → Test F1: 0.3846 | Recall: 0.2797 | Precision: 0.6154

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_155117
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 10
📦 Final model saved: trained_models/exp_20260406_155117/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.5556 | Recall: 0.4196 | Precision: 0.8219

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_161046
✅ Best model saved
⏹ Early stopping after epoch 3
📦 Final model saved: trained_models/exp_20260406_161046/final_model.pth
✅ Best model loaded for test evaluation
Seed 50 → Test F1: 0.1338 | Recall: 0.7063 | Precision: 0.0739

📊 Summary over seeds:
Avg Test F1: 0.3618 ± 0.1909
Avg Recall:  0.5483
Avg Precision: 0.4440
Min Test F1: 0.1338
Max Test F1: 0.5821