config = {
            "optimizer": "sgd",
            "lr": 0.060,
            "alpha": 0.60,
            "min_recall": 0.39,
            "momentum": 0.87,
        }
        
--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_181548
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_181548/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.2938 | Recall: 0.4108 | Precision: 0.2286

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_182857
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 7
📦 Final model saved: trained_models/exp_20260406_182857/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.6994 | Recall: 0.7095 | Precision: 0.6895

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_184906
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 13
📦 Final model saved: trained_models/exp_20260406_184906/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 30 → Test F1: 0.3378 | Recall: 0.2075 | Precision: 0.9091

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_192307
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_192307/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.5178 | Recall: 0.6349 | Precision: 0.4371

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260406_193612
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260406_193612/final_model.pth
✅ Best model loaded for test evaluation
Seed 50 → Test F1: 0.3802 | Recall: 0.4149 | Precision: 0.3509

📊 Summary over seeds:
Avg Test F1: 0.4458 ± 0.1473
Avg Recall:  0.4755
Avg Precision: 0.5231
Min Test F1: 0.2938
Max Test F1: 0.6994

--------------------------------

config = {
            "optimizer": "sgd",
            "lr": 0.060,
            "alpha": 0.55,
            "min_recall": 0.39,
            "momentum": 0.87,
        }

--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_092109
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_092109/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 10 → Test F1: 0.0669 | Recall: 0.0913 | Precision: 0.0528

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_093414
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_093414/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 20 → Test F1: 0.0779 | Recall: 0.2656 | Precision: 0.0456

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_094729
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_094729/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.4771 | Recall: 0.6473 | Precision: 0.3777

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_100028
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260407_100028/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.3184 | Recall: 0.3983 | Precision: 0.2652

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260407_101336
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 11
📦 Final model saved: trained_models/exp_20260407_101336/final_model.pth
✅ Best model loaded for test evaluation
Seed 50 → Test F1: 0.6734 | Recall: 0.5519 | Precision: 0.8636

📊 Summary over seeds:
Avg Test F1: 0.3227 ± 0.2333
Avg Recall:  0.3909
Avg Precision: 0.3210
Min Test F1: 0.0669
Max Test F1: 0.6734

-----------------------------
mit L2 Regularisierung weight_decay=1e-4

--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260408_200334
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260408_200334/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.3414 | Recall: 0.5317 | Precision: 0.2514

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260408_201626
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 14
📦 Final model saved: trained_models/exp_20260408_201626/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.8009 | Recall: 0.7183 | Precision: 0.9050

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260408_205320
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260408_205320/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.3319 | Recall: 0.4643 | Precision: 0.2583

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260408_210620
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260408_210620/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.4259 | Recall: 0.4960 | Precision: 0.3731

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260408_211957
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 5
📦 Final model saved: trained_models/exp_20260408_211957/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.0135 | Recall: 0.0079 | Precision: 0.0455

📊 Summary over seeds:
Avg Test F1: 0.3827 ± 0.2519
Avg Recall:  0.4437
Avg Precision: 0.3667
Min Test F1: 0.0135
Max Test F1: 0.8009

-------------------
mit Dropout im Head 0.2, ohne L2


--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_082710
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_082710/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.1293 | Recall: 0.4484 | Precision: 0.0755

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_084036
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_084036/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.3791 | Recall: 0.4603 | Precision: 0.3222

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_085406
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_085406/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.4643 | Recall: 0.5556 | Precision: 0.3989

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_090741
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_090741/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.4942 | Recall: 0.5040 | Precision: 0.4847

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_092115
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_092115/final_model.pth
✅ Best model loaded for test evaluation
⚠️ Kein Threshold erfüllt min_recall → fallback auf best F1
Seed 50 → Test F1: 0.0526 | Recall: 0.0635 | Precision: 0.0449

📊 Summary over seeds:
Avg Test F1: 0.3039 ± 0.1796
Avg Recall:  0.4063
Avg Precision: 0.2653
Min Test F1: 0.0526
Max Test F1: 0.4942

----------------------
L2 Regularisierung mit weight_decay=5e-4


--- Seed 10 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_113024
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_113024/final_model.pth
✅ Best model loaded for test evaluation
Seed 10 → Test F1: 0.4908 | Recall: 0.5278 | Precision: 0.4586

--- Seed 20 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_114247
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_114247/final_model.pth
✅ Best model loaded for test evaluation
Seed 20 → Test F1: 0.1916 | Recall: 0.5437 | Precision: 0.1163

--- Seed 30 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_115512
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_115512/final_model.pth
✅ Best model loaded for test evaluation
Seed 30 → Test F1: 0.3905 | Recall: 0.5556 | Precision: 0.3011

--- Seed 40 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_120756
✅ Best model saved
✅ Best model saved
✅ Best model saved
⏹ Early stopping after epoch 6
📦 Final model saved: trained_models/exp_20260409_120756/final_model.pth
✅ Best model loaded for test evaluation
Seed 40 → Test F1: 0.5269 | Recall: 0.4087 | Precision: 0.7410

--- Seed 50 ---
Device: mps
📁 Experiment directory: trained_models/exp_20260409_122532
✅ Best model saved
⏹ Early stopping after epoch 4
📦 Final model saved: trained_models/exp_20260409_122532/final_model.pth
✅ Best model loaded for test evaluation
Seed 50 → Test F1: 0.1574 | Recall: 0.4722 | Precision: 0.0944

📊 Summary over seeds:
Avg Test F1: 0.3514 ± 0.1516
Avg Recall:  0.5016
Avg Precision: 0.3423
Min Test F1: 0.1574
Max Test F1: 0.5269