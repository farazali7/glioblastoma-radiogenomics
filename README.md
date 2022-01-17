[mri-planes]: ./assets/mriplanes.jpg "MRI Planes"
[mri-seqs]: ./assets/mriseqs.png "MRI Sequences"
[imbalance]: ./assets/imbalance.png "Data imbalances"
[model]: ./assets/modelstruct.png "Model Architecture"

# Glioblastoma Radiogenomic Classification
A machine learning model for detecting presence of MGMT promoter methylation genetic sequence from brain MRI scans.

## Problem Definition
Chemotherapy drugs tend to damage tumour cell DNA. O[6]-methylguanine-DNA methyltransferase (MGMT) is a protein found in cells, such as tumour cells that repairs the cell's DNA. The more MGMT protein that the tumour cells produce, the less effective the chemotherapy drug is expected to be, as the protein will repair the damage to the tumour. Thus, determination of MGMT promoter methylation status in newly diagnosed GBM can influence treatment decision making. Current methods for genetic analysis of cancer are invasive, such as requiring surgery to extract sample tissue. By leveraging MR images, there is potential to non-invasively assess the pertinent genetics to make informed medical treatment decisions. 

## MRI Data Background
Magnetic resonance imaging (MRI) is one of the most commonly used tests in neurology and neurosurgery. MRI provides significant detail of the brain, spinal cord and vascular anatomy, and has the advantage of being able to visualize anatomy in all three planes: axial, sagittal, and coronal.

![mri-planes]

The most common type of MRI sequences are the following:
- Note: TR = Repetition Time: time between successive pulse sequences applied to the same slice. TE = Time to Echo: time between the delivery of the RF pulse and the receipt of the echo signal.
1. T1-weighted
    - Produced by using short time TE and TR times
2. T2-weighted
    - Produced by using longer TE and TR times
3. Fluid Attenuated Inversion Recovery (FLAIR)
    - Similar to T2 except TE and TR are very long. Abnormalities remain bright but normal CSF fluid is attenuated and made dark.
4. T1-weighted Gadolinium (Gad) Post Constrast
    - Gad is a non-toxic paramagnetic contrast enhancement agent that changes signal intensities by shortening T1, thus, it is very bright on T1-weighted images.

![mri-seqs]

In order to start tackling this problem, the data available
was first analyzed.

## Exploratory Data Analysis
The data analysis began with looking at the distribution
of targets within the dataset.

![imbalance]

The training dataset contains about 585 studies, each with numerous (~20-400) scans in the four sequences mentioned above. As shown above, the instances of positive to negative indications are almost the same.

The test dataset contains about 87 studies, each with numerous (~20-400) scans in the four sequences mentioned above.

Three studies have black images only and can be ignored from training (00109, 00123, 00709).

## Preprocessing + Data Cleaning + Augmentation
These are some approaches that were taken to help clean
the dataset a bit further and make it more plausible
to train given limited computation resources. For context, roughly 30 slices (images), from the middle of the respective set of scans, were chosen from each study in each sequence and supplied together as input to a 3D CNN model.
* Remove studies from training set that had only black images
* Normalize (scale) image data
* Crop black border from images
* Resize the image 120 x 120
* Upsample streams that have less than 30 slices

Additional random image augmentation techniques were applied to expand the training dataset. 
* Left-Right Flips
* Up-Down Flips
* Contrast, Brightness, Saturation, and Hue Changes
* Rotations
* Cutouts

## Model Structure & Performance
The model was based primarily on a pre-trained
EfficientNetB0 model. An input layer and a 3D convolution layer were prepended to it. A global average pooling layer and a final dense layer for predictions were appended.

![model]

The model using L2 regularization loss to help with overfitting and Adam's optimizer. Since a large input was being given at each step (a packaged bundle of 30 slices in each of the 4 sequences), to avoid memory problems, gradient accumulation was used to iterate the model's trainable parameters (weights and biases).

After about 5 epochs, the model achieves roughly a 0.704 AUC score but was not trained using the entire dataset yet due to limitations.

## Future Work
* Train model on full dataset entirely
* Test other pre-trained models
* Try a windowing and look-up-table preprocessing technique used by radiologists