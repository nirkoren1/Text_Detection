# Project Overview

Welcome to my text detection and recognition project! My primary goal is to develop a robust system capable of detecting and recognizing text in natural scene images. The process involves two key phases: text region detection and subsequent recognition using specialized models. <br>
<br>
## [Go to demo website](https://nirkoren1.github.io/Text_Detection/)


https://github.com/nirkoren1/Text_Detection/assets/72827063/2d644a37-de0f-4dfc-b07e-2c3026000329



## Models Structure

### Detection Model
My text detection model adopts a segmentation-based approach with a U-Net structure, leveraging VGG for efficient feature extraction.

### Recognition Model
For text recognition, I employ the innovative Vision Transformer architecture, using character tokens for effective processing.

## Data Collection and Preparation:

### Detection
To train my detection model, I utilized 5,000 samples from the TextOcr dataset on Kaggle for training and an additional 1,000 samples for testing.

### Recognition
For training my recognition model, I employed a diverse dataset comprising 16,000 samples from the TextOcr dataset and 100,000 samples from the synthtiger dataset. The test set consists of 1,000 samples from the TextOcr dataset.

## Text Detection Information:

I delve into refining existing text detection models such as EAST and CRAFT. My exploration includes experimenting with various backbone architectures and loss functions. I tackle challenges like overlapping text, complex backgrounds, and varying text orientations.

### Papers Referenced:
- [Character Region Awareness for Text Detection](https://arxiv.org/pdf/1904.01941v1.pdf)
- [Arbitrary Shape Text Detection via Segmentation with Probability Maps](https://arxiv.org/pdf/2208.12419.pdf)

## Text Recognition Information:

The text recognition aspect focuses on training a model using the Vision Transformer architecture. I prioritize the ability to handle diverse text fonts and sizes, emphasizing proficiency in the English language.

### Papers Referenced:
- [MaskOCR: Text Recognition with Masked Encoder-Decoder Pretraining](https://arxiv.org/pdf/2206.00311v3.pdf)
- [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/pdf/2109.10282v5.pdf)


