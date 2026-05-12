config = {
            "optimizer": "sgd",
            "lr": 0.055,
            "alpha": 0.60,
            "min_recall": 0.39,
            "momentum": 0.87,
        }
        
--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_202127
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260406_202127/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 10 → Test F1: 0.0472 | Recall: 0.0747 | Precision: 0.0345

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_203659
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 7
📦 Final model saved: trained_models/exp_20260406_203659/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.5271 | Recall: 0.4232 | Precision: 0.6986

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_205650
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_205650/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 30 → Test F1: 0.1011 | Recall: 0.0581 | Precision: 0.3889

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_210956
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_210956/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 40 → Test F1: 0.3255 | Recall: 0.2863 | Precision: 0.3770

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_212254
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_212254/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.0645 | Recall: 0.1162 | Precision: 0.0447

📊 Summary over seeds:
Avg Test F1: 0.2131 ± 0.1862
Avg Recall:  0.1917
Avg Precision: 0.3087
Min Test F1: 0.0472
Max Test F1: 0.5271