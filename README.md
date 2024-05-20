# ASMR: Activation-Sharing Multi-Resolution Coordinate Networks for Efficient Inference

> __ASMR - Activation-Sharing Multi-Resolution Coordinate Networks for Efficient Inference__  
> [JCL Li](https://www.linkedin.com/in/jason-chun-lok-li-0590b3166), [STS Luo](https://www.cs.toronto.edu/~stevenlts), [Le Xu](https://scholar.google.com/citations?user=3ZHoWOMAAAAJ&hl=zh-CN), [Ngai Wong](https://www.eee.hku.hk/~nwong)  
> _The Twelfth International Conference on Learning Representations (__ICLR__), 2024_  
> __[Project page](https://github.com/stevolopolis/taco_temp)&nbsp;/ [Paper](https://openreview.net/pdf?id=kMp8zCsXNb)&nbsp;/ [Presentation](https://drive.google.com/file/d/10MT_wnoEnvn4RJcQgpneV7TIXbLQqr8u/view?usp=sharing)__

## General code structure
`config/` - Contains the default configurations for all our experiments and models. Model configurations are stored in the subdirectory `config/model_config`. Each model configuration file contains a placeholder set of configs with headers `INPUT_OUTPUT`, which specifies the model input/output formats, and `NET`, which specifies the model hyperparameters. Following that are modality specific configurations with headers `INPUT_OUTPUT_<modality>` and `NET_<modality>`. For example, configurations for ASMR on megapixel images wuld have headers `INPUT_OUTPUT_MEGA` and `NET_MEGA`.

`models/`

`scripts/`

`train_<data>.py`

`dataset.py`

`profiler.py`

## Reproducing results
For all experiments, you may reproduce the results by running the bash scripts in `scripts`. These scripts will train all the models on the respective datasets, each with 3 different unique seeds. Please be reminded to change the `DATASET_CONFIGS.file_path` values in the respective bash scripts according to where you stored the dataset.

**Ultra low-mac**

**Audio**

**Image**

**Megapixel**
We trained our models with an Nvidia RTX3090. Each model takes around 8 hours to train. We resized the pluto image to 8192x8192. The image could be found [here](https://drive.google.com/file/d/1BUvsJXeoXqOJyrTf2hAQHd02r3vZvebu/view?usp=sharing). Download the image and change the path to the image in `config/train_megapixel.yaml` under the header `DATASET_CONFIGS` at entry `file_path`.

To reproduce the results in _Table 3: Megapixel image fitting on the 8192×8192_, use the command line:
```
bash scripts/train_megapixel.sh
```
This bash script calls `train_megapixel.py` with custom configuration parameters for each model. It automatically trains our default ASMR model, alongside our tuned SIREN, KiloNeRF, and NGP model for 3 unique seeds. To retrain our ASMR model with nonlinear modulations, you will have to change the `NET_MEGA.modulator` entry in `config/model_config/asmr.yaml` from `linear` to `nonlinear`.

DataLoader details
- `uniform_grid_sampler()` - since we cannot fit the entire megapixel image into a consumer grade GPU such as RTX3090 at once, we need to partition the image into smaller batches ($2^{18}=212144$) for training. In particular, we partition the image into a (16x16) grid and sample points uniformly within each grid. Note that all points will only be sampled __once__ in each epoch. This sampling strategy is inspired by kiloNeRF and our experiments show that it is superior to generic random sampling.

**Video**

**3D shapes**