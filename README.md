[English](./README.md) | [中文简体](./README.zh_CN.md)

<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-green.svg">
</a>
<a href="https://github.com/IAAR-Shanghai/Grimoire/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/Grimoire/issues">
</a>
<a href="https://arxiv.org/abs/2401.03385">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-blue.svg">
</a>

# 📖 Grimoire

Enhance the capabilities of small language models using grimoires.
<p align="center"><img src="./assets/grim_framework.jpg" alt=""></p>

## Contents
- [Project Structure](#project-structure)
- [Get Started](#get-started)
- [Results](#results)
- [Contact Us](#contact-us)
- [Citation](#citation)


## Project Structure
The project is organized into several key directories and modules. Here's an overview of the project structure:
```
.
├── assets        # Store project assets, such as images, diagrams, or any visual elements used to enhance the presentation and understanding of the project.
├── configs       # Store configuration files.
├── core          # Core codebase.
│   ├── data      # Data processing module.
│   ├── evaluator # Evaluator module.
│   └── llm       # Load Large Language Models (LLMs) module.
├── data          # Store datasets and data processing scripts.
├── external      # Store the Grimoire Ranking model based on the classifier approach.
├── outputs       # Store experiment output files.
├── prompts       # Store text files used as prompts when interacting with LLMs.
├── stats         # Store experiment statistical results.
└── tests         # Store test code or unit tests.
```


## Get Started

1. Prepare for the environment.
   * `conda create -n grimoire python=3.8.18`
   * `conda activate grimoire`
   * `pip install -r requirements.txt`
2. Run
   * [data/embed.py](data/embed.py) to embed datasets.
   * [data/compute_similarity.py](data/compute_similarity.py) to compute similarity matrix.
   * These are useful when you run similarity-based experiments.
3. Configure
   * the llms in [configs/llm.yaml](configs/llm.yaml).
   * the experiments in [configs/experiment.yaml](configs/experiment.yaml).
4. Look into [experiments.py](experiments.py) to see how to run experiments.
5. Run [analyst.py](analyst.py) to analyze the results saved in `outputs`.

## Results
<p align="center"><img src="./assets/res_gpt-3.5-turbo.jpg" alt=""></p>
<p align="center"><img src="./assets/acc_diff_grim_to_baseline.jpg" alt=""></p>


## Contact Us

For any questions, feedback, or suggestions, please open a GitHub Issue. You can reach out through [GitHub Issues](https://github.com/IAAR-Shanghai/Grimoire/issues).


## Citation
```
@article{grimoire,
      title={Grimoire is All You Need for Enhancing Large Language Models}, 
      author={Ding Chen and Shichao Song and Qingchen Yu and Zhiyu Li and Wenjin Wang and Feiyu Xiong and Bo Tang},
      year={2024},
      eprint={2401.03385},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```