config = {
            "optimizer": "sgd",
            "lr": 0.061,
            "alpha": 0.60,
            "min_recall": 0.39,
            "momentum": 0.87,
        }
        
-
--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_155520
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 7
📦 Final model saved: trained_models/exp_20260409_155520/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 10 → Test F1: 0.1288 | Recall: 0.0754 | Precision: 0.4419

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_161437
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 14
📦 Final model saved: trained_models/exp_20260409_161437/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.7911 | Recall: 0.7063 | Precision: 0.8990

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_165011
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 6
📦 Final model saved: trained_models/exp_20260409_165011/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.7059 | Recall: 0.5952 | Precision: 0.8671

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_170801
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260409_170801/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.6485 | Recall: 0.5675 | Precision: 0.7566

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_172336
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_172336/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.2249 | Recall: 0.1468 | Precision: 0.4805

📊 Summary over seeds:
Avg Test F1: 0.4999 ± 0.2693
Avg Recall:  0.4183
Avg Precision: 0.6890
Min Test F1: 0.1288
Max Test F1: 0.7911

-----------------------------------
mit L2 Regularisierung, weight=1e-4

--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_143932
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_143932/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.3786 | Recall: 0.5754 | Precision: 0.2821

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_145233
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_145233/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.4186 | Recall: 0.4286 | Precision: 0.4091

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_150605
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260407_150605/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 30 → Test F1: 0.3130 | Recall: 0.2540 | Precision: 0.4076

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_152149
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_152149/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.3653 | Recall: 0.3849 | Precision: 0.3477

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_153521
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_153521/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.0000 | Recall: 0.0000 | Precision: 0.0000

📊 Summary over seeds:
Avg Test F1: 0.2951 ± 0.1514
Avg Recall:  0.3286
Avg Precision: 0.2893
Min Test F1: 0.0000
Max Test F1: 0.4186

---------------------
mit L2 Regularisierung, weight=5e-4

--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_155040
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_155040/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.4415 | Recall: 0.5992 | Precision: 0.3495

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_160409
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_160409/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.1507 | Recall: 0.5476 | Precision: 0.0873

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_161704
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_161704/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.1854 | Recall: 0.4087 | Precision: 0.1199

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_163008
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_163008/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 40 → Test F1: 0.1495 | Recall: 0.3135 | Precision: 0.0981

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_164306
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_164306/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.0000 | Recall: 0.0000 | Precision: 0.0000

📊 Summary over seeds:
Avg Test F1: 0.1854 ± 0.1432
Avg Recall:  0.3738
Avg Precision: 0.1310
Min Test F1: 0.0000
Max Test F1: 0.4415

------------------------
mit Dropout im Head 0.3
--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_172532
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_172532/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 10 → Test F1: 0.0674 | Recall: 0.0476 | Precision: 0.1154

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_173856
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_173856/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.3003 | Recall: 0.3611 | Precision: 0.2571

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_175224
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 7
📦 Final model saved: trained_models/exp_20260407_175224/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 30 → Test F1: 0.3988 | Recall: 0.2619 | Precision: 0.8354

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_181238
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 7
📦 Final model saved: trained_models/exp_20260407_181238/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.7549 | Recall: 0.7698 | Precision: 0.7405

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_183330
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260407_183330/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.0062 | Recall: 0.0040 | Precision: 0.0137

📊 Summary over seeds:
Avg Test F1: 0.3055 ± 0.2672
Avg Recall:  0.2889
Avg Precision: 0.3924
Min Test F1: 0.0062
Max Test F1: 0.7549

---------------------------------
mit Dropout im Head 0.2

--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_185001
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_185001/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 10 → Test F1: 0.0836 | Recall: 0.0516 | Precision: 0.2203

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_190301
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 10
📦 Final model saved: trained_models/exp_20260407_190301/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 20 → Test F1: 0.3013 | Recall: 0.1865 | Precision: 0.7833

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_193024
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 6
📦 Final model saved: trained_models/exp_20260407_193024/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.7460 | Recall: 0.7341 | Precision: 0.7582

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_194824
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_194824/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.1760 | Recall: 0.4127 | Precision: 0.1118

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_200132
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_200132/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.3370 | Recall: 0.3611 | Precision: 0.3160

📊 Summary over seeds:
Avg Test F1: 0.3288 ± 0.2273
Avg Recall:  0.3492
Avg Precision: 0.4379
Min Test F1: 0.0836
Max Test F1: 0.7460