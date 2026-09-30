## ___***TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models***___
<div align="center">
<img src='assets/title_logo.png' style="height:100px"></img>
 
 <a href='https://arxiv.org/pdf/2503.05638'><img src='https://img.shields.io/badge/arXiv-2503.05638-b31b1b.svg'></a> &nbsp;
 <a href='https://trajectorycrafter.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://www.youtube.com/watch?v=dQtHFgyrids'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a>&nbsp;
 <a href='https://huggingface.co/spaces/Doubiiu/TrajectoryCrafter'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;

<strong>International Conference on Computer Vision (ICCV) 2025, Oral</strong>

</div>

🤗 If you find TrajectoryCrafter useful, **please help ⭐ this repo**, which is important to Open-Source projects. Thanks!

## 🔆 Introduction

- __[2025-03-10]__: 🔥🔥 Update the arXiv preprint.
- __[2025-02-23]__: Launch the project page.


TrajectoryCrafter can generate high-fidelity novel views from <strong>casually captured monocular video</strong>, while also supporting highly precise pose control. Below shows some examples:

<table class="center">
    <tr style="font-weight: bolder;">
        <td>Input Video &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; New Camera Trajectory</td>
    </tr>
  <td>
    <img src=assets/a1.gif style="width: 100%; height: auto;">
  </td>
  </tr>
  <tr>
  <td>
    <img src=assets/a5.gif style="width: 100%; height: auto;">
  </td>
  </tr> 
  <tr>
  <td>
    <img src=assets/a2.gif style="width: 100%; height: auto;">
  </td>
  </tr>
    <tr>
  <td>
    <img src=assets/a4.gif style="width: 100%; height: auto;">
  </td>
  </tr>
</table>


## ⚙️ Setup

### 0. GPU memory requirement

We recommend deploying it on a GPU with VRAM ≥ 28GB.


### 1. Clone TrajectoryCrafter
```bash
git clone --recursive https://github.com/TrajectoryCrafter/TrajectoryCrafter.git
cd TrajectoryCrafter
```
### 2. Setup environments
```bash
conda create -n trajcrafter python=3.10
conda activate trajcrafter
pip install -r requirements.txt
```

### 3. Download pretrained models
Ideally, you can load pretrained models directly from HuggingFace. If you encounter issues connecting to HuggingFace, you can download the pretrained models locally instead. To do so, you can:

1. Download the pretrained models using HuggingFace or using git-lfs
```bash
# HuggingFace (recommend)
sh download/download_hf.sh 

# git-lfs (much slower but more stable)
sh download/download_lfs.sh 
```

2. Change default path of the pretrained models to your local path in [inference.py](./inference.py).

## 💫 Inference 
### 1. Command line

Run [inference.py](./inference.py) using the following script. Please refer to the [configuration document](docs/config_help.md) to set up inference parameters and camera trajectory. 
```bash
  sh run.sh
```

### 2. Local gradio demo

```bash
  python gradio_app.py
```

##  📢 Limitations
Our model excels at handling videos with well-defined objects and clear motion, as demonstrated in the demo videos. However, since it is built upon a pretrained video diffusion model, it may struggle with complex cases that go beyond the generation capabilities of the base model.

## 🤗 Related Works
Including but not limited to: [CogVideo-Fun](https://github.com/aigc-apps/CogVideoX-Fun), [ViewCrafter](https://github.com/Drexubery/ViewCrafter), [DepthCrafter](https://github.com/Tencent/DepthCrafter), [GCD](https://gcd.cs.columbia.edu/), [NVS-Solver](https://github.com/ZHU-Zhiyu/NVS_Solver), [DimensionX](https://github.com/wenqsun/DimensionX), [ReCapture](https://generative-video-camera-controls.github.io/), [TrajAttention](https://xizaoqu.github.io/trajattn/), [GS-DiT](https://wkbian.github.io/Projects/GS-DiT/), [DaS](https://igl-hkust.github.io/das/), [RecamMaster](https://github.com/KwaiVGI/ReCamMaster), [GEN3C](https://research.nvidia.com/labs/toronto-ai/GEN3C/), [CAT4D](https://cat-4d.github.io/)...

## 📜 Citation
If you find this work helpful, please consider citing:
```BibTeXw
@InProceedings{Yu_2025_ICCV,
    author    = {Yu, Mark and Hu, Wenbo and Xing, Jinbo and Shan, Ying},
    title     = {TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {100-111}
}
```

