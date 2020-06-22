# Shared Features Learning on Semantic-Favorable Super Resolution
CVDL2020 Project, Contributed by Haoning Wu, Jiaheng Han
Mentor: Yadong Mu

## Introduction
Super Resolution task this year has been progressed by Deep End-to-end Neural Networks. When talking about Single Image Super Resolution (SISR), we are always wondering where the implicit mechanism of the encoder-decoder structure of such end-to-end networks lies. 

Most common view is that the SR generator (encoder-decoder) becomes some feature extractor which learns "add what details on what original textures", and among these years, GAN-based methods are proposed to learn the **latent distribution** of real images and generate more photo-realistic SR result instead of just trying to improve the PSNR/SSIM performance.

So, our baseline methods are set as EDSR in 2017 and ESRGAN pipeline (with EDSR generator) in 2018.

However, GAN surely can make "realistic" textures on SR domain, but it fails to decode the distinct semantics into different textures, which often generates statistically real but not semantically reasonable textures. This thing fails especially when we try to reconstruct large scale (e.g. *32 by 32 -> 256 by 256, 8x*)  ground truth of faces or other fine-grained specific contents: we may SR a smiling face into something very strange, maybe an angry face, or something not a face.

Another problem often confusing researchers is the Low Resolution fine-grained classification, focusing on more real-world fine-grained classification usage where the recognition objective is a small part of the whole input and is detected through some coarse-grained detection algorithm. When often **the higher-scale fine-grained classification** networks are learnt, these networks do not transfer well on small samples.

Now the branches are set together and consider how a human see something far: we're doing the "what this is (classfication)" and "human eye super resolution" synchronously, where the two task help each other in this mechanism. 

**In order to adverse the loss of fine-grained semantics of such super resolutions tasks,** we have decided to indirectly use our pretrained higher-scale fine-grained classfication models, and use this thing to both help the Semantic-Favorable Super Resolution and Low-Scale Fine-Grained Classification. Our contributions are mainly as follows:

- First, without introducing real labels of the Low Resolution fine-grained images,  we designed a classifier module into the common SR pipeline as a prior and help the Super Resolution network to learn more "realistic" results not only in textures, but in semantics as well.
- Second, we carefully designed the **shared features** learning for *Super Resolution* and *Low Resolution Classfication*, enforcing the encoder to learn both semantic and textural features and helps extract the latent distribution of the image space. 


## How to run
There are generally three stages of the whole experiment pipeline.

### Stage 1: Pre-train a fine-grained classification
```shell
python3 pre_train.py
```

### Stage 2: Train Semantic-Favorable Super Resolution
Not finished yet.

### Stage 3: Train Low-Scale Fine-Grained Classification
Not finished yet.
