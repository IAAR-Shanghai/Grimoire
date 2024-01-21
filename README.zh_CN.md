[English](./README.md) | [中文简体](./README.zh_CN.md)

<h1 align="center">
    📖 Grimoire is All You Need for Enhancing LLMs
</h1>
<p align="center">💡Enhance the capabilities of small language models using grimoires.
<p align="center">
<a href="https://opensource.org/license/apache-2-0/">
    <img alt="License: Apache" src="https://img.shields.io/badge/License-Apache2.0-green.svg">
</a>
<a href="https://github.com/IAAR-Shanghai/Grimoire/issues">
    <img alt="GitHub Issues" src="https://img.shields.io/github/issues/IAAR-Shanghai/Grimoire?color=red">
</a>
<a href="https://arxiv.org/abs/2401.03385">
    <img alt="arXiv Paper" src="https://img.shields.io/badge/Paper-arXiv-blue.svg">
</a></p>

## 目录
- [介绍](#介绍)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [联系我们](#联系我们)
- [引用](#引用)

## 介绍

上下文学习（ICL）是增强大型语言模型在特定任务上性能的关键方法之一，具体而言是通过提供一组少量的示例样本辅助大模型。然而，不同类型的模型在ICL能力上表现出显著差异，这是由模型架构、学习数据量和参数大小等因素造成的。通常来说，模型参数越大，学习数据越广泛，ICL能力越强。在本文中，我们提出了一种名为SLEICL（Strong LLM Enhanced ICL）的方法，该方法`通过使用强语言模型从示例中学习，然后进行总结归纳并将这些学到的技能（Grimoire）传递给弱语言模型来辅助其进行推理和应用。`

这种方法确保了ICL的稳定性和有效性。相对于直接让弱语言模型从提示示例中学习，SLEICL降低了这些模型的ICL难度。我们在5个语言模型上进行的实验证明，使用SLEICL方法，弱语言模型的表现相对于Zero-shot或者Few-shot设置时都实现了一致的改进。甚至一些弱语言模型在SLEICL的帮助下超越了GPT4-1106-preview（Zero-shot）的性能。
<p align="center"><img src="./assets/grim_framework.jpg" alt=""></p>

## 项目结构
该项目结构包括几个关键模块。以下是项目结构的概览：
```
.
├── archived      # 存储我们的实验所使用的grimoire和hard样本。
├── assets        # 存储项目素材，例如图像、图表或任何用于增强项目演示和理解的素材。
├── configs       # 存储配置文件。
├── core          # 核心代码库。
│   ├── data      # 数据处理模块。
│   ├── evaluator # 评估模块。
│   └── llm       # 加载大型语言模型 (LLMs) 模块。
├── data          # 存储数据集和数据处理脚本。
├── external      # 存储基于分类器方法的魔法书排名模型。
├── outputs       # 存储实验输出文件。
├── prompts       # 存储与LLMs交互时使用的文本文件。
├── stats         # 存储实验统计结果。
└── tests         # 存储测试代码或单元测试。
```

## 快速开始
1. 准备环境
   * `conda create -n grimoire python=3.8.18`
   * `conda activate grimoire`
   * `pip install -r requirements.txt`
2. 运行
   * [data/embed.py](data/embed.py) 以嵌入数据集。
   * [data/compute_similarity.py](data/compute_similarity.py) 以计算相似性矩阵。
   * 当运行基于相似性的实验时，这些步骤很有用。
3. 配置
   * 在 [configs/llm.yaml](configs/llm.yaml) 中配置 LLMS。
   * 在 [configs/experiment.yaml](configs/experiment.yaml) 中配置实验。
   * 如果需要复现我们的实验，可以将本所使用的grimoire和hard样本通过如下命令加载至当前路径下：`cp -r ./archived/.cache ./`
4. 查看 [experiments.py](experiments.py) 以了解如何运行实验。
5. 运行 [analyst.py](analyst.py) 以分析保存在 `outputs` 中的结果。

## 实验结果
<p align="center"><img src="./assets/res_gpt-3.5-turbo.jpg" alt=""></p>
<p align="center"><img src="./assets/acc_diff_grim_to_baseline.jpg" alt=""></p>


## 联系我们
如有任何问题、反馈或建议，请打开 GitHub Issue。您可以通过 [GitHub Issues](https://github.com/IAAR-Shanghai/Grimoire/issues) 联系我们。

## 待办事项
<details>
<summary>展开所有待办事项</summary>

- [ ] 编写统一的 setup.sh 来实现自动的环境配置和 embed.py 和 compute_similarity.py 的执行；
- [ ] 提供一个部署 Vllm 模型的简易教程；
- [ ] 实现直接从huggingface加载大模型；
- [ ] 增加 experiment.yaml 中的可配置项；
- [ ] 基于 Docker 对实验环境和代码进行打包，便于研究者快速使用部署；

</details>

## 引用
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
