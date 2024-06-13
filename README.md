#  LangNav: Language as a Perceptual Representation for Navigation 

## About LangNav
LangNav is an LLM-based navigation agent which performs multi-step navigation end-to-end via textual descriptions of the scene. The language-based perceptual representation makes LangNav more data efficient compared to VL models. With only a few language-based trajectories from a R2R environment, we use GPT-4 to efficiently generate a huge amount of synthetic training data. A smaller language model (LLaMA2-7B) can then be trained on these synthetic data and do the task. In this repo, we provide the inference code, the model, and the training dataset we used for the paper:

**LangNav: Language as a Perceptual Representation for Navigation**

[Bowen Pan](https://people.csail.mit.edu/bpan/), [Rameswar Panda](https://rpand002.github.io/), [SouYoung Jin](https://souyoungjin.github.io/), [Rogerio Feris](https://www.rogerioferis.org/), [Aude Oliva](http://olivalab.mit.edu/), [Phillip Isola](https://web.mit.edu/phillipi/) [Yoon Kim](https://people.csail.mit.edu/yoonkim/)

*NAACL 2024 (Findings)*

[[Paper](https://arxiv.org/pdf/2310.07889)][[GitHub](https://github.com/pbw-Berwin/LangNav)][[MIT News](https://news.mit.edu/2024/researchers-use-large-language-models-to-help-robots-navigate-0612)]

## Prerequisites

We don't have to install the [Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator) as we have pre-extracted the caption of each viewpoint.

But we still need to prepare the data in directories
- MP3D navigability graphs: `connectivity`
    - Download the [connectivity maps [23.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
- R2R data: `data`
    - Download the [R2R data [5.8MB]](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R/data).
- BLIP caption of the scene: `img_features`
    - Download the [caption data [113MB]](https://drive.google.com/file/d/1X7F48q--15h8cdzA_A5NozHtRJdtJsI8/view?usp=drive_link) (r2r_blip_DETR_vis2text).


Install the [Pytorch-Transformers](https://github.com/huggingface/transformers).

## Multi-step Navigation with Language-based Representation

Evaluate our [`LangNav-Sim2k-Llama2`](https://huggingface.co/bpan/LangNav-Sim2k-Llama2) model on the R2R datasets.
```bash
sh eval_scripts/eval_langnav_2k_synthetic_100_real.sh
```
We will also release the synthetic training dataset and the other models. Stay tuned!

## Citation
If you use or discuss our LangNav, please cite our paper:
```
@article{pan2023langnav,
  title={Langnav: Language as a perceptual representation for navigation},
  author={Pan, Bowen and Panda, Rameswar and Jin, SouYoung and Feris, Rogerio and Oliva, Aude and Isola, Phillip and Kim, Yoon},
  journal={arXiv preprint arXiv:2310.07889},
  year={2023}
}
```