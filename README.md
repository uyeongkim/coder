# Multi-Agent Coding Agent with PPO & SHPPO

This repository implements a multi-agent system for code generation, built on top of a Qwen LLM and fine-tuned via LoRA. It includes two RL strategies:

- **PPO**: Proximal Policy Optimization  
- **SHPPO**: Shared-parameter Heterogeneous PPO

Each strategy lives in its own subfolder (`ppo/` and `shppo/`), and both expose an `env.py` defining the training environment.

<!-- Repository URL: https://github.com/uyeongkim/coder.git -->

<!-- --- -->

## 📂 Repository Structure

```
.
├── ppo/
│   └── model.py        # PPO-specific model wrappers
│   └── trainer.py      # PPO trainer class (logging, checkpointing)
├── shppo/
│   └── model.py        # SHPPO-specific model wrappers
│   └── trainer.py      # SHPPO trainer class (logging, checkpointing)
├── env.py              # Environment setup for PPO/SHPPO
└── utils.py            # Shared utilities (data loading, metrics, helpers)
```



## ⚙️ Features

- **Base Model**: Qwen2.5-coder LLM  
   - We tested with 1.5B model, but larger variants (7B, 14B) are also supported
- **Fine-Tuning**: LoRA adapters for efficient weight updates  
- **RL Algorithms**:  
  - PPO: single-agent policy optimization  
  - SHPPO: multi-agent, shared-parameter PPO  
- **LoRA Integration**: seamless injection of low-rank adapters into Qwen’s transformer layers  
- **Metrics & Logging**: custom rewards for code correctness, performance logs via W&B or console


## 🛠 Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/uyeongkim/coder.git
   cd coder
   ```
2. Create & activate a Python environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   > **Requirements** include: `torch`, `transformers`, `accelerate`, `wandb`, `rich`, `gym`, etc.


## 🚀 Quickstart

### PPO training
```bash
python -m ppo.env \
  --base_model qwen \
  --lora_rank 8 \
  --learning_rate 1e-5 \
  --num_steps 1e6 \
  --save_dir outputs/ppo
```

### SHPPO (multi-agent) training
```bash
python -m shppo.env \
  --base_model qwen \
  --lora_rank 8 \
  --learning_rate 5e-6 \
  --num_steps 2e6 \
  --num_agents 3 \
  --save_dir outputs/shppo
```

Each `env.py` script handles:
1. Environment setup (agent ↔ code-challenges loop)  
2. LoRA injection into Qwen  
3. PPO/SHPPO optimizer & scheduler  
4. Checkpointing & logging



## 📚 Configuration

If you want to customize training settings, refer to the `PPOConfig` or `SHPPOConfig` dataclasses defined in each `trainer.py` file (under `ppo/` and `shppo/` respectively).


## 🧩 Utilities

- **`model.py`**:  
  - `LoRAQwenModel` wraps Qwen + LoRA adapters  
  - `PolicyHead` & `ValueHead` on top of hidden states

- **`trainer.py`**:  
  - `[METHOD]Trainer` orchestrates rollout collection, optimization, logging, and checkpointing  
  - Pluggable for both PPO and SHPPO loops
  - Change [METHOD] to `PPO` or `SHPPO` as needed

- **`utils.py`**:  
  - Data loading, reward calculators, metric trackers, seed-setting, etc.



## 📖 Acknowledgement

<!-- Please cite this work if you use it: -->

We build on the Qwen 2.5 repository — please cite the original technical report if you use Qwen models. We sincerely thank the Qwen team for their excellent contributions.

We also acknowledge the original SHPPO work. Since the official implementation was not available, we re-implemented SHPPO based on the paper. You can find our version in the `./shppo` directory.

If you find this repository helpful, please consider citing our work.


```bibtex
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
```bibtex
@article{GUO2025130716,
title = {Heterogeneous multi-agent reinforcement learning for zero-shot scalable collaboration},
journal = {Neurocomputing},
year = {2025},
author = {Xudong Guo and Daming Shi and Junjie Yu and Wenhui Fan},
}
```


MIT-licensed – see [LICENSE](LICENSE) for details.
