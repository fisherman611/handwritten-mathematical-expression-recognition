# **Handwritten Mathematical Expression Recognition**

## **Project Overview**
This project focuses on recognizing handwritten mathematical expressions and converting them into LaTeX format. The system leverages deep learning techniques to process images of handwritten equations, interpret their structure, and generate corresponding LaTeX code. The primary goal is to achieve high accuracy in recognizing complex mathematical expressions, addressing challenges such as varying handwriting styles and intricate symbol arrangements. The project is built using PyTorch and incorporates advanced neural network architectures tailored for this task.

## **Dataset**

The project utilizes the **CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions)** dataset, a widely used benchmark for handwritten mathematical expression recognition. The dataset is organized into several subsets, each containing images and their corresponding LaTeX annotations.

Download the splitted dataset: [CROHME Splitted](https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_lh225458_sis_hust_edu_vn/EviH0ckuHR9KiXftU5ETkPQBHvEL77YTscIHvfN7LBSrSg?e=CHwNxv) and then place in the `data/` directory.

## **Methods and Models**

### **Preprocessing** 
Steps to clean and standardize images: 
* Load in grayscale.
* Use Canny edge detection, dilate with $7 \times 13$ kernel to connect edges.
* Crop with F1-score method to focus on the expression.
* Binarize with adaptive thresholding; set background to black if needed.
* Apply median blur (kernel 3) multiple times to reduce noise.
* Add 5-pixel padding, resize to $128 \times 384$, pad with black if needed.

### **Augmentation**
Augmentation to handle handwriting variations: 
* Rotate up to 5 degrees, border replication. 
* Elastic transform for stroke variations.
* Random morphology: erode or dilate to change stroke thickness.
* Normalize and convert to tensor.

### **Model: Counting-Aware Network (CAN)**

CAN is an end-to-end model for HMER, combining recognition and symbol counting: 
* **Backbone:** 

    * DenseNet (or ResNet)
    * Takes grayscale image $H' \times W' \times 1$, outputs feature map $\mathcal{F} \in \mathbb{R}^{H \times W \times 684}$, where ($H = \frac{H'}{16}$), ($W = \frac{W'}{16}$).

* **Multi-Scale Counting Module (MSCM):** 

    * Uses $3 \times 3$ and $5 \times 5$ conv branches for multi-scales features.
    * Channel attention: $$\mathcal{Q} = \sigma(W_1(G(\mathcal{H})) + b_1)$$ $$\mathcal{S} = \mathcal{Q} \otimes g(W_2 \mathcal{Q} + b_2)$$
    * Concatenates features, $1 \times 1$ conv to counting map $$\mathcal{M} \in \mathbb{R}^{H \times W \times C}$$
    * Sum-pooling gives counting vector $$\mathcal{V}i = \sum{p=1}^H \sum_{q=1}^W \mathcal{M}_{i,pq}$$

* **Counting-Combined Attentional Decoder (CCAD):**

    * $1 \times 1$ conv on $\mathcal{F}$ to $\mathcal{T} \in \mathbb{R}^{H \times W \times 512}$, adds positional encoding.
    * GRU gives hidden state $h_t \in \mathbb{R}^{1 \times 256}$, attention weights: $$e_{t,ij} = w^T \tanh(\mathcal{T} + \mathcal{P} + W_a \mathcal{A} + W_h h_t) + b$$ $$\alpha_{t,ij} = \frac{\exp(e_{t,ij})}{\sum_{p=1}^H \sum_{q=1}^W \exp(e_{t,pq})}$$
    * Context vector $\mathcal{C} \in \mathbb{R}^{1 \times 256}$, predicts token: $$p(y_t) = \operatorname{softmax}(w_o^T (W_c \mathcal{C} + W_v \mathcal{V}^f + W_t h_t + W_e E(y_{t-1})) + b_o)$$
    * Beam search (width = 5) for inference.

* **Loss:** 
    
    * Loss class: $$\mathcal{L}{\text{cls}} = -\frac{1}{T} \sum{t=1}^T \log(p(y_t))$$
    * Loss counting: $$\mathcal{L}{\text{counting}} = \operatorname{smooth}{L_1}(\mathcal{V}^f, \hat{\mathcal{V}})$$
    * Total loss:  $$\mathcal{L} = \mathcal{L}{\text{cls}} + \lambda \mathcal{L}{\text{counting}}$$ $$\lambda = 0.01$$

## **Results**
## **Conclusion**
CAN works well for handwritten math recognition on CROHME dataset. It handles complex expressions with counting and attention. Future ideas: try transformer decoders, add synthetic data, improve preprocessing for noisy images.

## **Installation**
Clone the repository and naviagate to the project directory: 
```bash
git clone https://github.com/fisherman611/handwritten-mathematical-expression-recognition.git
```

Navigate to the project directory:
```bash
cd handwritten-mathematical-expression-recognition
```

Install the required dependencies: 
```bash
pip install -r requirements.txt
```

## **Download the pretrained model**
Download the pretrained model checkpoints from this [OneDrive link](https://husteduvn-my.sharepoint.com/:f:/g/personal/thanh_lh225458_sis_hust_edu_vn/EvWQqIjJQtNKuQwwH1G8EMkBcRPM8s3msiI7-IBERbve1A?e=6SeGHB)

Place the downloaded checkpoint in the `checkpoints/` directory within the repository.

## **Inference**
## **References**
[1] B. Li, Y. Yuan, D. Liang, X. Liu, Z. Ji, J. Bai, W. Liu, and X. Bai, "When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition," arXiv preprint arXiv:2207.11463, 2022. [Online]. Available: https://arxiv.org/abs/2207.11463
## **License**
This project is licensed under the [MIT License](LICENSE).
