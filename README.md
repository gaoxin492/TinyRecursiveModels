# Less is More: Recursive Reasoning with Tiny Networks

This repository is a reproduction and experimental verification of the paper “Less is More: Recursive Reasoning with Tiny Networks” by [Alexia Jolicoeur-Martineau (2025)](https://arxiv.org/abs/2510.04871). This repository is not an official implementation. It is maintained solely for research reproduction. For the official version, please visit [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

## Experiments

### Reproduction Results

| Method | Params | Sudoku | Maze | ARC-1 (@2) | ARC-2 (@2) |
| --- | --- | --- | --- | --- | --- |
| TRM-Att | 7M | 77.71 | 78.70 | 41.00  | 3.33 |
| TRM-MLP | 5M | 84.80 | / | / | / |

### Model Checkpoints on Hugging Face
[TinyRecursiveModel-Maze-Hard](https://huggingface.co/Sanjin2024/TinyRecursiveModel-Maze-Hard)

[TinyRecursiveModels-Sudoku-Extreme-att](https://huggingface.co/Sanjin2024/TinyRecursiveModels-Sudoku-Extreme-att)

[TinyRecursiveModels-Sudoku-Extreme-mlp](https://huggingface.co/Sanjin2024/TinyRecursiveModels-Sudoku-Extreme-mlp)

[TinyRecursiveModels-ARC-AGI-1](https://huggingface.co/Sanjin2024/TinyRecursiveModels-ARC-AGI-1)

[TinyRecursiveModels-ARC-AGI-2](https://huggingface.co/Sanjin2024/TinyRecursiveModels-ARC-AGI-2)

The file `pretrain.py` has been slightly modified to handle missing evaluators gracefully:
```python
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        import traceback
        print("No evaluator found:", repr(e))
        traceback.print_exc()
        evaluators = []
```

In addition to evaluation during training, a standalone evaluation script `run_eval.py` has been added. This script allows loading checkpoints and running evaluation separately. We report exact accuracy for Maze and Sudoku, and pass@k for ARC.
```bash
torchrun --nproc_per_node=8 run_eval.py
# or evaluate all tasks
bash eval_scripts.sh
```
All experiments were conducted on 8 × H GPUs with a global batch size of 4608.

#### ARC-AGI-1
```bash
run_name="pretrain_att_arc1concept_8"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
epochs=200000  \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```
*Runtime:* 37h
<img width="1603" height="788" alt="image" src="https://github.com/user-attachments/assets/cf702613-969a-49a8-ad77-2d30df666485" />
<img width="1597" height="786" alt="image" src="https://github.com/user-attachments/assets/c96a51cb-ab62-4adb-85ac-cf7c746fc8f7" />
<img width="1600" height="1186" alt="image" src="https://github.com/user-attachments/assets/08d44b1a-3c11-4ffe-aa2e-3c996adfd3fa" />


#### ARC-AGI-2
```bash
run_name="pretrain_att_arc2concept_8"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
epochs=200000  \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```
*Runtime:* 49h
<img width="1640" height="791" alt="image" src="https://github.com/user-attachments/assets/d2b7053c-676c-4dd2-a37f-78be331778e4" />
<img width="1649" height="784" alt="image" src="https://github.com/user-attachments/assets/ed95c51f-4912-43d1-a17a-358d5234a78e" />
<img width="1646" height="1187" alt="image" src="https://github.com/user-attachments/assets/c9b769a6-0ef7-49a5-843b-3c9426afb92d" />


#### Sudoku-Extreme:
```bash
run_name="pretrain_mlp_t_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=100000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

run_name="pretrain_att_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=100000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```
*Runtime:* 40min
<img width="1504" height="786" alt="image" src="https://github.com/user-attachments/assets/0f6ba0eb-5bad-491a-8d4b-f3a0e7cb6909" />
<img width="1505" height="1174" alt="image" src="https://github.com/user-attachments/assets/e86a3205-f556-4a86-8e9a-da7e50e2acf9" />

#### Maze-Hard:
```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```
*Runtime:* 2h
<img width="1514" height="792" alt="image" src="https://github.com/user-attachments/assets/14b321ea-48a7-4c13-ac8b-0a7ec8dea600" />
<img width="1515" height="1186" alt="image" src="https://github.com/user-attachments/assets/60f8865e-cede-46c5-8394-c6d68d8d5e34" />


