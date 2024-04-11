<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">HiLo: Detailed and Robust 3D Clothed Human Reconstruction with High-and Low-Frequency Information of Parametric Models</h1>
  <p align="center">
    <strong>Yifan Yang</strong></a>
    ·
    <strong>Dong Liu</strong></a>
    ·
    <strong>Shuhai Zhang</strong></a>
    ·
    <strong>Zeshuai Deng</strong></a>
    .
    <strong>Zixiong Huang</strong></a>
    .
    <strong>Mingkui Tan</strong></a>
  </p>
  <h2 align="center">CVPR 2024</h2>

  <p align="center">
  <br>
    <a href='https://arxiv.org/abs/2404.04876'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction-to-HiLo">Introduction to HiLo</a>
    </li>
    <li>
      <a href="#instructions">Instructions</a>
    </li>
    <li>
      <a href="#running-demo">Running Demo</a>
    </li>
    <li>
    <a href="#training-and-testing">Training and testing</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>
<br />
<br />

<br>


## Introduction-to-CR-NeRF 
|              ![Pipeline](images/pipeline.png)               |
| :----------------------------------------------------------: |
|       _Pipeline of HiLo_       |        
- If you want to **Train & Evaluate**, please check [installation.md](https://github.com/YuliangXiu/ICON/blob/master/docs/installation.md) to prepare environment, required models and extra data. Please check [dataset.md](https://github.com/YuliangXiu/ICON/blob/master/docs/dataset.md) to prepare THuman2.0 and CAPE dataset, see
<a href="#training-and-testing">Training and testing</a> to train and benchmark HiLo using the prepared datasets.

- If you want to **Running Demo**, please see
<a href="#running-demo">.

Giving a RGB image of clothed human, with our HiLo, you will get:
  - image:
    - with the normals of smpl and cloth
  - mesh:
    - with the 3d objects of smpl, reconstructed and refined cloth
  - video:
    - showing the reconstructed human from all angles


<br/>


<br>

## Instructions

- See [docs/installation.md](docs/installation.md) to install all the required packages and setup the models
- See [docs/dataset.md](docs/dataset.md) to prepare the in-the-wild datasets
- See <a href="#training-and-testing">Training and testing</a> to train and benchmark CR-NeRF using Brandenburg Gate tainingset

<br/>



<br>


<br>


## Running Demo

```bash
#Set $in_dir, $out_dir and cuda devices in command/infer.sh
bash command/infer.sh
```
The reconstructed results (mesh, image, video) will be in path "{$out_dir}".


## Training and testing

**If you want to train and test the model**

```bash
#Set experiment name and cuda devices in train_and_test.sh 
bash command/train_and_test.sh
```

**If you only want to test the model**

```bash
#Set the experiment name to match the training name, and set cuda devices in test_only.sh  
bash command/test_only.sh
```

<br/>
<br/>

## Citation

```bibtex
@inproceedings{yang2024hilo,
      title={HiLo: Detailed and Robust 3D Clothed Human Reconstruction with High-and Low-Frequency Information of Parametric Models}, 
      author={Yifan Yang and Dong Liu and Shuhai Zhang and Zeshuai Deng and Zixiong Huang and Mingkui Tan},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      year={2024}
}
```

## Acknowledgments

Here are some great resources we benefit from:

- [ICON](https://github.com/YuliangXiu/ICON.git) 
- [ECON](https://github.com/YuliangXiu/ECON.git)
- [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset.git)

<br>

## License

By downloading and using the code and model you agree to the terms in the [LICENSE](LICENSE).


