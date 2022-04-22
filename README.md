# GANs_Literature

[GANs Theory: How it works internally - presentation file by Hyosun Choi 12/04/2022](https://1drv.ms/b/s!AhCVuY1b1tq1i7RtRoOl5iHMsA0_nA?e=blTRgm)

[GANs & Classification for peatlands images presentation file by Hyosun Choi 10/03/2022 @Nobel House](https://1drv.ms/b/s!AhCVuY1b1tq1i7Rsc5fFCd_iu93pCQ?e=Sv9YQF)

## 01. Traditional GANs : to generate more datasets of peatlands as we have only small datasets.
We decided to explore this literature first and then see if we can move onto the modern collaborated GANs literature.
### Hyosun 
01. List from my ppt in Nobel House 10/03/2022 (Updates Done_22/04/2022 from my ppt in Nobel House_10/03/2022)\
1.1. [The latest trend of GANs](https://medium.com/sciforce/whats-next-for-gans-latest-techniques-and-applications-3be06a7e5ab9) \
1.2. [18 Impressive Applications of Generative Adversarial Networks(GANs)](https://machinelearningmastery.com/impressive-applications-of-generative-adversarial-networks/): categories of GANs, very well organised
* Generate Examples for Image Datasets
* Generate Photographs of Human Faces
* Generate Realistic Photographs
* Generate Cartoon Characters
* Image-to-Image Translation
* Text-to-Image Translation
* Semantic-Image-to-Photo Translation
* Face Frontal View Generation
* Generate New Human Poses
* Photos to Emojis
* Photograph Editing
* Face Aging
* Photo Blending
* Super Resolution
* Photo Inpainting
* Clothing Translation
* Video Prediction
* 3D Object Generation 

&ensp;&ensp;&ensp;&ensp;1.3. [really-awesome-gan](https://github.com/nightrome/really-awesome-gan): useful list
* [ARIGAN](https://arxiv.org/pdf/1702.03410): Synthetic Arabidopsis Plants using Generative Adversarial Network
* [Conditional Image Synthesis with Auxiliary Classifier GANs](https://c4209155-a-62cb3a1a-s-sites.googlegroups.com/site/nips2016adversarial/WAT16_paper_7.pdf)
* [GANs for Biological Image Synthesis](https://arxiv.org/abs/1708.04692)
* [Learning a Generative Adversarial Network for High Resolution Artwork Synthesis](https://arxiv.org/pdf/1708.09533):Improved ArtGAN for Conditional Synthesis of Natural Image and Artwork
* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network(SRGAN)](https://arxiv.org/pdf/1609.04802)
* [Learning from Simulated and Unsupervised Images through Adversarial Training](https://arxiv.org/pdf/1612.07828)
* [Megapixel Size Image Creation using Generative Adversarial Networks](https://arxiv.org/pdf/1706.00082)
* [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/pdf/1604.04382)
* [Semi-Latent GAN](https://arxiv.org/pdf/1704.02166): Learning to generate and modify facial images from attributes
* [Texture Synthesis with Spatial Generative Adversarial Networks](https://arxiv.org/pdf/1611.08207v3)
* [Synthetic Medical Images from Dual Generative Adversarial Networks](https://arxiv.org/pdf/1709.01872)
* [StackGAN++](https://arxiv.org/pdf/1710.10916): Realistic Image Synthesis with Stacked Generative Adversarial Networks
* [Synthesis of Positron Emission Tomography (PET) Images via Multi-channel Generative Adversarial Networks (GANs)](https://arxiv.org/pdf/1707.09747)
* [Synthetic Iris Presentation Attack using iDCGAN](https://arxiv.org/pdf/1710.10565)
* [Image Generation and Editing with Variational Info Generative AdversarialNetworks](https://arxiv.org/pdf/1701.04568)
* [GP-GAN](https://arxiv.org/pdf/1703.07195): Towards Realistic High-Resolution Image Blending
* [Generative Adversarial Network based on Resnet for Conditional Image Restoration](https://arxiv.org/pdf/1707.04881)
* [GANs for Biological Image Synthesis](https://arxiv.org/pdf/1708.04692)
* [Efficient Super Resolution For Large-Scale Images Using Attentional GAN](https://arxiv.org/pdf/1812.04821)
* Depth Structure Preserving Scene Image Generation
* [DeLiGAN](https://arxiv.org/pdf/1706.02071) : Generative Adversarial Networks for Diverse and Limited Data
* [ArtGAN](https://arxiv.org/pdf/1702.03410): Artwork Synthesis with Conditional Categorial GANs
* [ARIGAN](https://arxiv.org/pdf/1709.00938): Synthetic Arabidopsis Plants using Generative Adversarial Network
* [AlignGAN](): Learning to Align Cross-Domain Images with Conditional Generative Adversarial Networks
* [Amortised MAP Inference for Image Super-resolution]()
* [Analyzing Perception-Distortion Tradeoff using Enhanced Perceptual Super-resolution Network]()
* [A Novel Approach to Artistic Textual Visualization via GAN]()


&ensp;&ensp;&ensp;&ensp;1.4. [How to Develop a Pix2Pix GAN for Image-to-Image Translation](https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/): The Pix2Pix model is a type of conditional GAN, or cGAN \
&ensp;&ensp;&ensp;&ensp;1.5. [etc.](https://imgur.com/t/artificial_intelligence/vO8gVBF) 

02. [Various GANs related to satellite images: https://github.com/robmarkcole/satellite-image-deep-learning](https://github.com/robmarkcole/satellite-image-deep-learning) \
2.1. [Generating synthetic data](https://github.com/robmarkcole/satellite-image-deep-learning#synthetic-data) \
2.1.1. [The Synthinel-1 dataset: a collection of high resolution synthetic overhead imagery for building segmentation](https://arxiv.org/ftp/arxiv/papers/2001/2001.05130.pdf) with [repo](https://github.com/timqqt/Synthinel) : \
&ensp; Dependencies: The dependencies to run the codes are CityEngine 2019.0 \
&ensp; Third-Party Software: CityEngine used by Synthinel is a tool for rapidly generating large-scale virtual urban scenes. \
2.1.2. [RarePlanes](https://www.cosmiqworks.org/RarePlanes/) -> incorporates both real and synthetically generated satellite imagery including aircraft. Read [the arxiv paper](https://arxiv.org/abs/2006.02963) and checkout [the repo](https://github.com/aireveries/RarePlanes). Note the dataset is available through the AWS Open-Data Program for free download \
2.1.3. Read [this article from NVIDIA](https://developer.nvidia.com/blog/preparing-models-for-object-detection-with-real-and-synthetic-data-and-tao-toolkit/) which discusses fine tuning a model pre-trained on synthetic data (Rareplanes) with 10% real data, then pruning the model to reduce its size, before quantizing the model to improve inference speed \
2.1.4. [Combining Synthetic Data with Real Data to Improve Detection Results in Satellite Imagery](https://one-view.ai/combining-synthetic-data-with-real-data-to-improve-detection-results-in-satellite-imagery-case-study/) \
&ensp;&ensp; 2.1.4.0. https://one-view.ai/ \
&emsp;&ensp;&ensp;&ensp; 2.1.4.1. https://one-view.ai/combining-synthetic-data-with-real-data-to-improve-detection-results-in-satellite-imagery-case-study/ \
&emsp;&ensp;&ensp;&ensp; 2.1.4.2. https://one-view.ai/mixing-it-up-the-benefits-of-blending-synthetic-and-real-world-data/ \
&emsp;&ensp;&ensp;&ensp; 2.1.4.3. https://one-view.ai/the-transition-of-real-world-imagery-to-synthetic-data/ \
&emsp;&ensp;&ensp;&ensp; 2.1.4.4. https://one-view.ai/five-key-insights-into-synthetic-data-for-geospatial-imagery/ \
2.1.5. [BlenderGIS](https://github.com/domlysz/BlenderGIS) could be used for synthetic data generation \
2.1.6. [bifrost.ai](https://www.bifrost.ai/) -> simulated data service with geospatial output data formats \
2.1.7. [oktal-se](https://www.oktal-se.fr/deep-learning/) -> software for generating simulated data across a wide range of bands including optical and SAR \
2.1.8. [The Nuances of Extracting Utility from Synthetic Data](https://www.iqt.org/synthesizing-robustness-yoltv4-results-part-1/) -> We find that strategically augmenting the real dataset is nearly as effective as adding synthetic data in the quest to improve the detection or rare object classes, and that fully extracting the utility of synthetic data is a nuanced process \
2.1.9. [Synthesizing Robustness](https://www.iqt.org/synthesizing-robustness/) -> explores how to best leverage and enhance synthetic data \
2.1.10. [rendered.ai](https://rendered.ai/) -> The Platform as a Service for Creating Synthetic Data \
2.1.11. [synthetic_xview_airplanes](https://github.com/yangxu351/synthetic_xview_airplanes) -> creation of airplanes synthetic dataset using ArcGIS CityEngine \
2.1.12. [Combining Synthetic Data with Real Data to Improve Detection Results in Satellite Imagery: Case Study](https://one-view.ai/combining-synthetic-data-with-real-data-to-improve-detection-results-in-satellite-imagery-case-study/) \
2.1.13. [SynImageAnalysis](https://github.com/FlorenceJiang/SynImageAnalysis) -> comparing syn and real sattlelite images in the latent feature space (embeddings) \
2.1.14. [Import OpenStreetMap data into Unreal Engine 4](https://github.com/ue4plugins/StreetMap) \
2.1.15. [DIRSIG](http://dirsig.cis.rit.edu/) 
&ensp; -> The Digital Imaging and Remote Sensing Image Generation model is a physics-driven synthetic image generation model: The Digital Imaging and Remote Sensing Image Generation (DIRSIG™) model is a physics-driven synthetic image generation model developed by the Digital Imaging and Remote Sensing Laboratory at Rochester Institute of Technology. The model can produce passive single-band, multi-spectral or hyper-spectral imagery from the visible through the thermal infrared region of the electromagnetic spectrum. The model also has an very mature active laser (LIDAR) capability and an evolving active RF (RADAR) capability. The model can be used to test image system designs, to create test imagery for evaluating image exploitation algorithms and for creating data for training image analysts. \
2.2. [Online-communities](https://github.com/robmarkcole/satellite-image-deep-learning#online-communities) \
&ensp;&ensp;2.2.1. https://forums.fast.ai/t/geospatial-deep-learning-resources-study-group/31044 \
&ensp;&ensp;&ensp;&ensp;2.2.1.1. Stanford’s Sustainability & AI Lab (projects on crop yield analysis & poverty prediction): http://sustain.stanford.edu/projects/ \
&ensp;&ensp;2.2.2.[Kaggle](https://www.kaggle.com/getting-started/131455) \
2.3. GANs list by Hyosun \
&ensp;&ensp;2.3.0. [GANs](https://github.com/robmarkcole/satellite-image-deep-learning#gans) \
&ensp;&ensp;2.3.1. [SCALAE](https://github.com/LendelTheGreat/SCALAE) [[paper](https://arxiv.org/pdf/2101.05069.pdf)] : Formatting the Landscape: Spatial conditional GAN for varying population in satellite imagery  [[ALAE](https://github.com/LendelTheGreat/SCALAE/blob/master/README_ALAE.md): [paper](https://arxiv.org/pdf/2004.04467.pdf)] \
&ensp;&ensp;2.3.2. [BigGANs](https://arxiv.org/pdf/1809.11096.pdf) [[code](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb)]: LARGE SCALE GAN TRAINING FOR
HIGH FIDELITY NATURAL IMAGE SYNTHESIS(ICLR 2019)[[trained models](https://tfhub.dev/s?q=biggan)] reference of 2.3.1. \
&ensp;&ensp;2.3.3. [Semantic Image Synthesis With Spatially-Adaptive Normalization](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html)[CVPR [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.pdf)][[code](https://github.com/NVlabs/SPADE)] \
&ensp;&ensp;2.3.4. [Useful search results: 'Large scale GAN training for high fidelity natural image synthesis' cited papers](https://scholar.google.co.uk/scholar?cites=9573828555610570748&as_sdt=2005&sciodt=0,5&hl=en) \
&ensp;&ensp;&ensp;&ensp;2.3.4.1. [Self-Attention Generative Adversarial Networks](https://proceedings.mlr.press/v97/zhang19d.html):Ian Goodfellow \
&ensp;&ensp;2.3.5. [PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/) [[paper](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf)] \
2.4. [SynImageAnalysis](https://github.com/FlorenceJiang/SynImageAnalysis) \
2.5. [ML metrics](https://github.com/robmarkcole/satellite-image-deep-learning#ml-metrics) \
2.6. [Data Augmentation]() \
&ensp;&ensp;2.6.1. [A survey on Image Data Augmentation for Deep Learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0) \
&ensp;&ensp;&ensp;&ensp;2.6.1.1. Our future work intends to explore performance benchmarks across geometric and color space augmentations across several datasets from different image recognition tasks. These datasets will be constrained in size to test the effectiveness with respect to limited data problems. Zhang et al. [135] test their novel GAN augmentation technique on the SVHN dataset across 50, 80, 100, 200, and 500 training instances. Similar to this work, we will look to further establish benchmarks for different levels of limited data. \
&ensp;&ensp;&ensp;&ensp;2.6.1.2. Improving the quality of GAN samples and testing their effectiveness on a wide range of datasets is another very important area for future work. We would like to further explore the combinatorics of GAN samples with other augmentation techniques such as applying a range of style transfers to GAN-generated samples. \
&ensp;&ensp;&ensp;&ensp;2.6.1.3. Super-resolution networks through the use of SRCNNs, Super-Resolution Convolutional Neural Networks, and SRGANs are also very interesting areas for future work in Data Augmentation. We want to explore the performance differences across architectures with upsampled images such as expanding CIFAR-10 images from 32 × 32 to 64 × 64 to 128 × 128 and so on. One of the primary difficulties with GAN samples is trying to achieve high-resolution outputs. Therefore, it will be interesting to see how we can use super-resolution networks to achieve high-resolution such as DCGAN samples inputted into an SRCNN or SRGAN. The result of this strategy will be compared with the performance of the Progressively Growing GAN architecture. \
&ensp;&ensp;2.6.2. [DADA: Deep Adversarial Data Augmentation for Extremely Low Data Regime Classification](https://arxiv.org/pdf/1809.00981.pdf) \
&ensp;&ensp;2.6.3. [Multiclass non-Adversarial Image Synthesis with Application to Classification from Very Small Sample](https://www.cs.huji.ac.il/w~daphna/papers/Winter_Arxiv2020.pdf) \
&ensp;&ensp;2.6.4. [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125/pdf?version=1582551862) \
&ensp;&ensp;2.6.5. [Exploring Bias in GAN-based Data Augmentation for Small Samples](https://arxiv.org/pdf/1905.08495v1.pdf) \
&ensp;&ensp;2.6.6. [How to implement augmentations for Multispectral Satellite Images Segmentation using Fastai-v2 and Albumentations](https://towardsdatascience.com/how-to-implement-augmentations-for-multispectral-satellite-images-segmentation-using-fastai-v2-and-ea3965736d1) \
&ensp;&ensp;2.6.7. [AutoGAN: Neural Architecture Search for Generative Adversarial Networks
](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gong_AutoGAN_Neural_Architecture_Search_for_Generative_Adversarial_Networks_ICCV_2019_paper.pdf) \
&ensp;&ensp;2.6.8. [Troika Generative Adversarial Network (T-GAN): A Synthetic Image Generator That Improves Neural Network Training for Handwriting Classification](https://philjournalsci.dost.gov.ph/images/pdf/pjs_pdf/vol149no3a/Troika_generative_adversarial_network_.pdf)\
03. [SEN12MS Toolbox:](https://github.com/schmitt-muc/SEN12MS) Schmitt M, Hughes LH, Qiu C, Zhu XX (2019) SEN12MS - a curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. In: ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences IV-2/W7: 153-160



4. 
5. https://www.intelligence-airbusds.com/imagery/constellation/?gclid=CjwKCAjwrqqSBhBbEiwAlQeqGj8ePK3YtnfXu2WEP2ywQRnK4II6WCGxIU7UpAsg5xwchXimWd_ygxoCMvcQAvD_BwE
* My previous code: My MSc dissertation code 2019, Hyosun Choi => My MSc codes are owned by © 2019 Royal Holloway, University of London reserved.


### From the team
01. [Generating synthetic multispectral satellite imagery from sentinel-2](https://arxiv.org/pdf/2012.03108.pdf)
02. [Data augmentation using CycleGANs to improve CT segmentation](https://www.nature.com/articles/s41598-019-52737-x.pdf) 
