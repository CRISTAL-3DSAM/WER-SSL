# Transformer-based Self-supervised Multimodal Representation Learning for Wearable Emotion Recognition

This repository contains official implementation of the paper: [Transformer-Based Self-Supervised Multimodal Representation Learning for Wearable Emotion Recognition](https://arxiv.org/abs/2107.13669)

## Model Architecture

Overview of our self-supervised multimodal representation learning framework. The proposed self-supervised learning (SSL) model is first pre-trained with signal transform recognition as the pretext task to learn generalized multimodal representation. The encoder part of the resulting pre-trained model is then served as a feature extractor for downstream tasks which is frozen or fine-tuned on the labeled samples to predict emotion classes.

<img src="https://github.com/YWU726/code-ssl-test/blob/main/Figures/Overview.JPG" width="70%" height="70%">

## Usage

**1. Set up conda environment**

```commandline
conda env create -f environment.yml
conda activate SSL
```

**2. Datasets**

The pre-trained SSL model was evaluated on three multimodal datasets: [WESAD](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf), [CASE](https://www.nature.com/articles/s41597-019-0209-0.pdf) and [K-EmoCon](https://www.nature.com/articles/s41597-020-00630-y.pdf). Please cite the creators.

**3. Train  the SSL model**

```commandline
python SSL.py --path=<path to the downloaded codes> --data_pat=<path to the unlabeled data>
```

In the paper, we use the PRESAGE dataset that we collected at the [Presage Training Center](https://medecine.univ-lille.fr/presage) in Lille, France, for self-supervised learning. Discussions with the funders and the University of Lille are underway to make this dataset publicly accessible. In this case, the pre-trained models are shared in the folder ```pretrained_models```. You can also use your own data at hand for pre-training.

**4. Evaluate the SSL model on supervised emotion datasets** 

For WESAD:

```commandline
python SL.py --path=<path to the downloaded codes> --dataset_opt='WESAD' --data_path=<path to data> --best_model_dir=<path to the pretrained model> --sl_num_classes=<number of emotion categories: 2 or 3> --mode=<training mode: 'freeze' or 'fine_tune'>
```

For CASE/K-EmoCon, you need to specify the emotional dimension, i.e., valence or arousal:

```commandline
python SL.py --path=<path to the downloaded codes> --dataset_opt='CASE'/'KemoCon' --data_path=<path to data> --best_model_dir=<path to the pretrained model> --sl_num_classes=<number of emotion categories: 2 or 3> --mode=<training mode: 'freeze' or 'fine_tune'> --av_opt=<emotional dimension: 'valence' or 'arousal'>
```

## Acknowledgements

The proposed work was supported by the French State, managed by the National Agency for Research (ANR) under the Investments for the future program with reference ANR16-IDEX-0004 ULNE.

## Citation

If this paper is useful for your research, please cite us at:

```
@ARTICLE{10091193,
  author={Wu, Yujin and Daoudi, Mohamed and Amad, Ali},
  journal={IEEE Transactions on Affective Computing}, 
  title={Transformer-Based Self-Supervised Multimodal Representation Learning for Wearable Emotion Recognition}, 
  year={2023},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TAFFC.2023.3263907}}
```



