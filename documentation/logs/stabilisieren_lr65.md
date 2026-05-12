config = {
            "optimizer": "sgd",
            "lr": 0.065,
            "alpha": 0.60,
            "min_recall": 0.39,
            "momentum": 0.87,
        }
        
--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_164456
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_164456/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.5235 | Recall: 0.6017 | Precision: 0.4633

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_165826
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 6
📦 Final model saved: trained_models/exp_20260406_165826/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 20 → Test F1: 0.3366 | Recall: 0.2158 | Precision: 0.7647

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_171617
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_171617/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.2877 | Recall: 0.4315 | Precision: 0.2158

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_172949
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260406_172949/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.3967 | Recall: 0.3942 | Precision: 0.3992

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_174521
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_174521/final_model.pth
✅ Best model loaded for test evaluation
Seed 50 → Test F1: 0.5132 | Recall: 0.5228 | Precision: 0.5040

📊 Summary over seeds:
Avg Test F1: 0.4115 ± 0.0939
Avg Recall:  0.4332
Avg Precision: 0.4694
Min Test F1: 0.2877
Max Test F1: 0.5235