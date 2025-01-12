# Covid-19 Severity Classification

This repository contains the notebook and files used to develop the model discussed in the thesis titled "Covid-19 Severity Classification Using Hybrid Feature Extraction Techniques" 
The model was developed to classify the severity of covid-19 from chest X-ray images. It utilizes a hybrid approach combining convolutional neural networks, vision transformers, and persistent homology.
Files
 - "Copy of coatnet lung severity score classification on minimized and non segemented" and "coatnet lung severity score classification on minimized and non segemented": These files likely contain variations of the CoatNet model applied to classify COVID-19 severity. One variation may use segmented lung images as input, while the other may use the full chest X-ray images without segmentation.

- "PH for image.ipynb" and "PH_Feature_to_files.ipynb": These Jupyter Notebook files probably pertain to the extraction and processing of persistent homology features from the images. "PH for image.ipynb" might contain the code for calculating persistent homology from individual images, while "PH_Feature_to_files.ipynb" could be responsible for extracting these features for the entire dataset and saving them to files.
- "coatnet lung severity score classification": This file could contain the implementation of the CoatNet model for classifying COVID-19 severity, potentially utilizing the segmented lung images.
- "metadata_consensus_v1.xlsx" and "metadata_global_v2.xlsx": These spreadsheet files likely contain metadata associated with the chest X-ray images used in the study. They might include information such as patient demographics, image acquisition parameters, and severity labels provided by radiologists.
- "modified combined model.ipynb": This Jupyter Notebook file likely contains the implementation of the final proposed model, which combines the CoatNet features with the persistent homology features. The notebook might detail the model architecture, training process, and evaluation results.
- "u-net-lung-segmentation-montgomery-shenzhen.ipynb": This Jupyter Notebook file probably contains the implementation of the U-Net model used for segmenting the lung regions from the chest X-ray images. It might include the code for training, evaluating, and utilizing the U-Net model for segmentation.

The file list suggests a structured approach to developing the COVID-19 severity classification model. The presence of files related to persistent homology, CoatNet, U-Net segmentation, and a combined model indicates a comprehensive implementation of the hybrid approach discussed in the thesis.
