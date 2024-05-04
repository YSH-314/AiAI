# Image generation for the radiography of dog hip.
The aim of this study is to utilize the diffusion model to augment the original dataset, thereby improving the performance of the Norberg Angle prediction model. The Norberg Angle is illustrated in the figure below. The study involved fine-tuning the DDPM, DreamBooth, and Stable Diffusion models to produce high-quality images. Moreover, a tailored model that amalgamated the EfficientNet and ViT_Gigantic_Patch14_Clip_224 architectures achieved the lowest prediction loss.

<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/3c3fd898-7857-4f2a-88fd-723165ddfb4f" width="450" height="250">

## The workflow of the study
<img src="https://github.com/YoushanZhang/AiAI/assets/74528993/8ce23469-dc6c-4eb3-8fa9-781e1f20cb92" width="550" height="350">


## Image generation from Diffusion models 
You can find the diffusion model weights through the [link](https://shorturl.at/pAY49)

Also, you can find all the generated images [here](https://drive.google.com/drive/folders/1Y_rxgAFNPX2thpiiFbtdiF52q3OT3mbK?usp=drive_link) 

In total, our dataset comprises 1047 real-world images and 1474 generated images. To prevent data leakage and ensure unbiased evaluation, we
meticulously partitioned the dataset into distinct training, testing, and validation sets. Specifically, the training set encompasses Set1, Set2,
Set3, and a subset of Set4, as delineated in the following table. The testing and
validation sets were both split from real-word image Set4. This selection
rationale ensures that the training data encapsulates a diverse
array of visual contexts, fostering robust model learning and adaptation

| Source           | # Images |
|------------------|---------:|
| Set1             |      219 |
| Set2             |       31 |
| Set3             |       84 |
| Set4             |      713 |
| DDPM             |      307 |
| DreamBooth 1     |      467 |
| DreamBooth 2     |      500 |
| Stable Diffusion 1 |     100 |
| Stable Diffusion 2 |     100 |

### Examples of generated images_1:
<img src="https://github.com/YSH-314/AiAI/assets/74528993/236c09c6-2e4d-43b7-af1f-6313e279c3b5" width="750" height="400">

### Examples of generated images_2:
<img src="https://github.com/YSH-314/AiAI/assets/74528993/1daddc41-df10-4945-ae24-269611550c6a" width="650" height="450">

## Inference from diffusion models:
#### import the packages:
```python
# Sample Python code
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from IPython.display import display

