# CNN Architectures Benchmarking in Edge Devices 

Edge devices have become cutting-edge technology to put the ML model closer to the final user. However, their performance in harsh environments plays a fundamental role in designing the electronic device for collecting data, tailoring data to the model's requirements and inferring the decision locally. Therefore, the CNN architecture used to train the ML model is a crucial element since it delivers the power consumption, the execution time, and the system time response. To save time in training ML models, transfer learning is a technique that enables the transfer of knowledge learned from one domain to another, allowing for the development of efficient and accurate models in a new domain. Then, the pre-trained model can be optimized by quantization techniques, which are focused on reducing the number of bits used to represent the weights and activations of a neural network or removing the redundant neurons or layers
from the neural network. In this project, we tested several CNN architectures with some IoT datasets to check their performance with software-centric and hardware-centric metrics when the ML model was or not optimized. 

## Models

The following CNN architectures were tested in this project:

* Resnet 50
* VGG16
* MobileNet V2
* Inception V3
* EfficientNet B0

## Datsets

We selected the following datasets since they represent several harsh environments that Edge devices must face.

* [Leaf illness detection](https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf): This data is used to diagnose plant diseases, and it is taken from experimental research stations associated with Land Grant Universities in the US. The database
contains images of both infected and healthy leaves of 14 crops, such as tomatoes, apples, and others. In this paper, only a set of tomato crop images were used.
* [Waste classification](https://www.kaggle.com/datasets/techsash/waste-classification-data): This targets classification in buildings’ trash bins, which is challenging; smart detectors can either recognize the trash belonging to the bin or have one hole to receive the trash
* [Birds Danish detection](https://www.dof.dk/images/udvalg/su/dokumenter/su_listen/DenDanskeFugleliste_oktober2014.pdf): Denmark has a very active bird spotting community that organizes bird spotting trips every month. One of the more active organizations is Dansk Ornitolo- gisk Forening, DOF (Danish Ornithological Association),
which has more than 17,500 members spread across 13 local branches nationwide. DOF regularly updates the Danish Birds list.
* [Solar panels cracks detection](): Solar modules are composed of many solar cells. The solar cells are subject
to degradation, causing many different types of defects. Defects may occur during the transportation, installation of modules, and operation. The size of cracks may range from tiny to large cracks covering the whole cell.
* [Satellite imagery](): The global dataset for landcover classification is based on annual timeseries data from three different satellites: Sentinel-1, Sentinel-2, and Landsat- 8. The LandCoverNet dataset consists of 300 ortho-
images of size 100x100km2 (referred to as tiles) that are spread across six different regions; (1) Africa, (2) Asia, (3) Australia, (4) Europe, (5) North America an (6) South America. The 300 tiles are distributed among
the regions based on their relative area.

The following table summarizes the datasets' characteristics.

| IoT application | Dataset                | Total Images | Dimension     | Full size | labels |
|-----------------|------------------------|--------------|---------------|-----------|--------|
| Smart farming    | Leaf illness detection | 11000        | 256 × 256 × 3 |  200 MB   |   10   |
| Smart buildings  | Waste classification   | 25000        | 256 × 256 × 3 |  260 MB   |    2   |
| Wildlife detection | Birds detection  | 80000 | 224 × 224 × 3 | 1.5 GB | 18 |
| Industrial operations | Solar panels cracks detection | 2624 | 300 × 300 × 3  | 266 MB | 2 |
| Remote sensing | Satellite imagery | 3500  | 256 × 256 × 3  | 350 MB | 7 |


## Edge devices 

Several Edge devices were taken in consideration in the benchmarking and they are described as follows: 

|Specifications | Raspberry Pi 4 | Raspberry Pi zero | Jetson TX2 | Jetson Nano | Coral Dev Board |
|---------------|----------------|-------------------|------------|-------------|-----------------|
| CPU | Quad core Cortex-A72 64-bit SoC | Quad-core Arm Cortex-A53 | Dual-Core NVIDIA Denver, Quad-Core ARM Cortex-A57 | ARM Cortex-A57 MPCore | Quad Cortex-A53 Cortex-M4F|
| RAM | 4GB SDRAM | 512MB SDRAM | 8GB LPDDR4 | 4GB LPDDR4  | 1 GB LPDDR4 |
| Storage | Micro-SD card (64 GB) | Micro-SD card (64 GB) | 32GB eMMC | Micro-SD card (64 GB) | Micro-SD card (64 GB) |
| Wireless Connectivity | 2.4 GHz and 5.0 GHz IEEE 802.11ac | 2.4GHz 802.11 b/g/n | 2.4GHz 802.11 b/g/n | NN | 802.11a/b/g/n/ac 2.4/5GH |
| Camera Connector | 2-lane MIPI CSI camera port | CSI-2 camera connector | CSI2 D-PHY 1.2 (2.5 Gbps/Lane) | MIPI-DSI x2 | MIPI-CSI2 camera input (4-lane)|
| Hardware accelerator | None | None | NVIDIA Pascal GPU with 256 CUDA cores | 128-core GPU | Google Edge TPU: 4 TOPS (int8)|

## Environment

All models were trained, deployed and tested with 

```bash
TensorFlow 12
```

## Results

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```
