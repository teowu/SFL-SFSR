# Shared Features Learning on Semantic-Favorable Super Resolution
CVDL2020 Project, Contributed by Haoning Wu, Jiaheng Han

Mentor: Yadong Mu

## Background and Motivation

Super Resolution task this year has been progressed by Deep End-to-end Neural Networks. When talking about Single Image Super Resolution (SISR), we are always wondering where the implicit mechanism of the encoder-decoder structure of such end-to-end networks lies. 

Most common view is that the SR generator (encoder-decoder) becomes some feature extractor which learns "add what details on what original textures", and among these years, GAN-based methods are proposed to learn the **latent distribution** of real images and generate more photo-realistic SR result instead of just trying to improve the PSNR/SSIM performance.

So, our baseline methods are set as RCAN in 2018 and ESRGAN pipeline (with RCAN generator) in 2018.

However, GAN surely can make "realistic" textures on SR domain, but it fails to decode the distinct semantics into different textures, which often generates statistically real but not semantically reasonable textures. This thing fails especially when we try to reconstruct large scale (e.g. *14 by 14 -> 224 by 224, 16x*)  ground truth of faces or other fine-grained specific contents: we may SR a smiling face into something very strange, maybe an angry face, or something not a face.

Another problem often confusing researchers is the Low Resolution fine-grained classification, focusing on more real-world fine-grained classification usage where the recognition objective is a small part of the whole input and is detected through some coarse-grained detection algorithm. When often **the higher-scale fine-grained classification** networks are learnt, these networks do not transfer well on small samples. 

On the contrary, recent researches on fine-grained classification has demonstrated that the attention areas (where the pixels contribute the most to the Prediction Labels) are often very local, which suggests that prior knowledge from common fine-grained classification will be focusing more on **local semantic features**.

Now the branches are set together and consider how a human see something far: we're doing the "what this is (recognition)" and "human eye super-resolution" synchronously, where the two task help each other in this mechanism. We 

**We desire to construct deep learning pipelines to modulate human eye super-resolution process, learn Semantic-Favorable Super Resolution, especially in extreme cases.** For this purpose, we have decided to indirectly use our pretrained higher-scale fine-grained classfication models, and use this thing to both help the Semantic-Favorable Super Resolution and Low-Scale Fine-Grained Classification. Our contributions are mainly as follows:

- First, without introducing real labels of the Low Resolution fine-grained images, we designed a specific feature-extractor prior module into the common SR pipeline as a prior and help the Super Resolution network to learn more "realistic" results not only in textures, but in semantics as well.
- Second, we carefully designed the **shared features** learning for *Super Resolution* and *Low Resolution Classfication*, enforcing the encoder to learn both semantic and textural features and helps extract the latent distribution of the image space. This will both help low-res recognition and extreme super resolution, where in this paper we only talk about the latter one.
- Third, we composed the spatial attention mechanism which can make use of class activation maps to localize the attention areas and help rebuild semantic-favorable super resolution results on heat point areas.

## Method Overview

Our method is based on "Learning from the Recognition priors". Because of the characteristics of the priors, we proposed **three ways of Utilizing the Priors**, shown in the following figure.

![Fig_2](figs/fig_2.png)

And the prior is based on the pre-training of the following FCN-VGG Classifier.

![Fig_3](figs/fig_7.png)

You can see the detailed method in how-to-train section.


## How to Train

There are generally four stages of the whole experiment pipeline.

### Stage 1: Pre-train a fine-grained classification

![Fig_4](figs/fig_8.png)

RUN as follows. It will take around 20 minutes for pre-processing.

```shell
python3 pre_train.py
```

Please rename your pre-trained model as *../bird/prt.pth* before processing the following stages.

**If you have only got HR datasets, please run ``python3 downsampler.py`` to generate LR samples.**

### Stage 2: Add Fine-Grained Classifier Loss

![Fig_5](figs/fig_6.png)

The overall loss with CLSLoss added is shown in the figure above.

You may try both ways of importing this loss, 
1. directly learning from pre-trained classification results, 
2. learn from the feature layers in the pre-trained *Classification Module*. The second one is set as default.

RUN as follows.

```shell
python3 train.py Stage_2.yml
```

### Stage 3: Shared-Features Learning (SFL)

![Fig_6](figs/fig_5.png)

RUN as follows.

```shell
python3 train.py Stage_3.yml
```

*REMINDer: The low-res Classification result is not benchmarked in the process above.*

### Stage 4: Spatial Attention Mechanism (SAM)

![Fig_7](figs/fig_4.png)

Before training Stage 4, you should at first generate the class activation map dataset and place it in the directory corresponding to [it](Stage_4.yml). The figure above shows the method used in [cam_generator.py](cam_generator.py)

```shell
python3 cam_generator.py
```

![Fig_8](figs/fig_3.png)

Then RUN as follows.

```shell
python3 train.py Stage_4.yml
```

### Combine the Stages
```shell
python3 train.py train.yml
```

## How to Test

*Not finish building yet. When finished, run the following code.*

```shell
python3 test.py test.yml
```


## base OpenSource Repositories

The proposed method is brand new, but in order to not "build the wheels again", our codes are based on the following OpenSource Repositories on GitHub.

1. **Xintao Wang et. al, BasicSR** (basic SR and SRGAN pipeline, used as baseline)

This framework is very good for Super Resolution practicers because it provides many functionals such as load/save network, update learning rate and network printing, etc. It also supports changing the loss functions, network architectures and dataloaders very conveniently. We are building our blocks on the BasicSR base class and **altered** the **loss functions, network architectures, training pipelines and dataloaders**.

The raw module (ESRGAN with iterative-RCAN) is set as the baseline model.

2. **Yulun Zhang et. al, RCAN** (state-of-the art SR module)

RCAN is the state-of-the-art Super Resolution Generator module and is often used as backbone of . However, as it is primarily designed for 2x/3x/4x Super Resolution, the original RCAN uses the LR image as input and provides *only one convolutional layer* for each upsampling step(2x) for 4x/8x/16x/... **Large Scale Super Resolution**. This will largely reduce the *learning ability* of multi-scale features, especially intermediate-scale features. As we are training on 8x/16x Large Scale Super Resolution, we must make sure that the intermediate scales are well-built, and we added a residual group (with 6 residual blocks and 12 convolutional layers) for Super Resolution.

This iterative-RCAN is also set as the baseline model.

3. **Timothy Haoning Wu, DMDR** (a workflow on real-world SR based on BasicSR)

The dataset generator is based on my DMDR and added the four channel concatenate input for *Spatial Attention Mechanism* learning. The train.py is also based on this repo finished earlier this year.

This method is far from the DMDR, and I used the DMDR just because I'm familiar with the general pipelines I built on this work.

4. Torchvision

While building the pre-training network, we used the VGG19-Pretrained architecture from Torchvision and removed all fc-layers, changing them to Global Average Pooling (GAP) and Convolution (Kernel-Size=1).


## Contributions of Team Members

####  Phase 1:  Pipeline Construcion

This part is finished together.

Idea and Architecture Construction: Haoning Wu and Jiaheng Han

Dataset Acquiring and Pre-processing: Jiaheng Han

Kernel Pipeline Construction: Haoning Wu


#### Phase 2: Method Building

This part is finished together.


Network Code Building: Haoning Wu

Model Pipeline Building (Stage 2,3,4): Haoning Wu

Stage 1 Training Framework and Experimenting: Jiaheng Han

#### Phase 3: Experimenting

This part is finished together.


CAM Generating: Jiaheng Han (Coding) and Haoning Wu (Debugging)

Experimental Debugging on Model Pipeline: Jiaheng Han

Major-flow Experimenting on Stage 2,3,4: Haoning Wu

#### Phase 4: Method Review (Report and Presentation)

This part is finished together.

## Experimental Results

### Qualitative Results (Bird)

![Fig_1](figs/fig_1.png)

Remark the **head and legs** of the two birds, SFL has obviously better reconstruction on these parts.

The SAM is relatively weaker on reconstructing these parts, however it produces more realistic bird body, comparing to the baseline ESRGAN algorithm.

### Quantitive Comparison

We have tested the following combinations so far, showing the previledge of our method on preserving the structural similarity and obtain better image quality on IQA (Image Quality Assessment) modules. With ablation study, the Share Features Module should become the one that contributes most in the three, while the only CLSLoss even degrades the performance. 

The result of only CLSLoss (Stage 2) does not match our expectation, so we removed that module in the combination evaluation and will continue on optimizing this part.

| **Stage/Scale**                       | **PSNR**⬆️ | **SSIM**⬆️ | **NIQE**⬇️ |
| ------------------------------------- | --------- | --------- | --------- |
| Baseline Iterative RCAN               | **18.99** | **0.502** |           |
| Baseline ESRGAN                       | **17.77** | **0.446** |           |
| Only CLSLoss                          | **16.88** | **0.403** |           |
| Only Share Features                   | **18.46** | **0.479** |           |
| Only Spatial Attention                | **18.29** | **0.452** |           |
| *Shared Features + Spatial Attention* |           |           |           |

However, traditional benchmarks are not capable of representing the semantics, so we may introduce some semantic-aware deep IQA methods into evaluation in the future. That would be another work.

