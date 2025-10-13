# Less is More: Recursive Reasoning with Tiny Networks

This repository is a reproduction and experimental verification of the paper “Less is More: Recursive Reasoning with Tiny Networks” by [Alexia Jolicoeur-Martineau (2025)](https://arxiv.org/abs/2510.04871)

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

This repository is not an official implementation.
It is maintained solely for research reproduction. For the official version, please visit [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)


## Experiments

### Reproduction Results

| Method | Params | Sudoku | Maze | ARC-1 (@2) | ARC-2 (@2) |
| --- | --- | --- | --- | --- | --- |
| TRM-Att | 7M | 77.73 | 78.70 | 38.50  | 3.33 |
| TRM-MLP | 5M | 84.73 | / | / | / |

In the first training run, I didn’t plot the pass@k curves — I’ll include them in the next updated results. For the ARC experiments, I have increased the number of training epochs, and the performance is expected to improve further.

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
torchrun --nproc_per_node=8 run_eval.py > output.txt 2>&1
# or evaluate all tasks
bash eval_scripts.sh
```
All experiments were conducted on 8 × H200 GPUs with a global batch size of 4608.

#### ARC-AGI-1
```bash
run_name="pretrain_att_arc1concept_8"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```
*Runtime:* 15~16h

#### ARC-AGI-2
```bash
run_name="pretrain_att_arc2concept_8"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```
*Runtime:* 23~24h

#### Sudoku-Extreme:
```bash
run_name="pretrain_mlp_t_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
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
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```
*Runtime:* 40min

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
