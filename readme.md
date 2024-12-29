# AMS IZZIV - final report
MARINA BABIĆ

MODETV2 (Motion Decomposition Transformer version 2)

https://github.com/marinaa05/AMS-CHALLENGE2.git
(Final code is on branch dev)

*My first model was called XMorpher (https://github.com/marinaa05/AMS-CHALLENGE.git), which I had to change due to insufficient dataset for training.*

## Method Explanation
**ModeTv2** method in my project is a deep learning-based approach designed for deformable medical image registration, which uses pyramidal structure to refine deformation fields progressively, addressing global deformations at low resolutions and local adjustments at higher resolutions. Key to this method is the Motion Decomposition Transformer (ModeTv2) module, which uses neighborhood attention to capture localized motion patterns and a Registration Head (RegHead) to fuse these patterns into a cohesive deformation field.

So, first step is feature extraction from moving and fixed images using a shared encoder. The extracted features are passed to ModeTv2, which uses neighborhood attention to calculate subfields representing motion patterns. After that, the RegHead combines subfields into a refined deformation field. The final deformation field is applied to the moving image, aligning it with the fixed image. The model is trained to minimize differences between the fixed and warped images (using a loss function), with pairwise optimization (PO) which helps further improve alignment for specific image pairs.

![Illustration of the proposed deformable registration network.](images_readme/modetv2.png)
*Image 1: Illustration of the proposed deformable registration network. The encoder takes the fixed image I<sub>f</sub> and moving image I<sub>m</sub>
as input to extract hierarchical features F<sub>1</sub>-F<sub>5</sub> and M<sub>1</sub>-M<sub>5</sub>. The ModeTv2 consists of a GPU-accelerated motion decomposition Transformer (ModeT) and a registration head (RegHead). The ModeTv2 is used to generate multiple deformation subfields and then fuses them. Finally the decoding pyramid outputs the total deformation field ϕ.*

## Results
**Firstly** I have trained my model with FBCT as the moving and CBCT as the fixed image. The training process consisted of 55 epochs, each comprising 11 iterations, corresponding to the 11 patients in the training dataset.

Validation of trained model with 55 epochs:
```bash
 aggregated_results:
        LogJacDetStd        : 0.00676 +- 0.00177 | 30%: 0.00790
        TRE_kp              : 10.13131 +- 2.69525 | 30%: 11.74156
        TRE_lm              : 10.73650 +- 3.44077 | 30%: 11.36809
        DSC                 : 0.26955 +- 0.07639 | 30%: 0.22731
        HD95                : 49.13435 +- 13.35092 | 30%: 37.64458
```

An example of registration (scans of patient 12). On the images below we can see the alignment of the two images (fixed and moving) in all three axes (X, Y, Z) and deformation fields in all axes for this certain registration:

![Fixed image (FBCT), moving image (CBCT at the end of therapy), transformed moving image in axis X](images_readme/patient12_x_axis.PNG) 
*Image 2: Fixed image (FBCT), moving image (CBCT at the end of therapy), transformed moving image in axis X*

![com](images_readme/def_polje_x.PNG)![com](images_readme/def_polje_x2.PNG)    
*Image 3: Deformation fields for Image 2.*

![Fixed image (FBCT), moving image (CBCT at the end of therapy), transformed moving image in axis Y](images_readme/patient12_y_axis.PNG)
*Image 4: Fixed image (FBCT), moving image (CBCT at the end of therapy), transformed moving image in axis Y*

![com](images_readme/def_polje_Y.PNG)![com](images_readme/def_polje_Y2.PNG)    
*Image 5: Deformation fields for Image 4.*

![Fixed image (FBCT), moving image (CBCT at the end of therapy), transformed moving image in axis Z](images_readme/patient12_z_axis.PNG)
*Image 6: Fixed image (FBCT), moving image (CBCT at the end of therapy), transformed moving image in axis Z*

![com](images_readme/def_polje_z.PNG)![com](images_readme/def_polje_z2.PNG)    
*Image 7: Deformation fields for Image 6.*

Looking at the results, I aimed to get better ones, so I increased number of epochs to 150 and I got very similar values:
```bash
aggregated_results:
        LogJacDetStd        : 0.00626 +- 0.00275 | 30%: 0.00840
        TRE_kp              : 10.12429 +- 2.70181 | 30%: 11.72799
        TRE_lm              : 10.73639 +- 3.46249 | 30%: 11.42628
        DSC                 : 0.26953 +- 0.07641 | 30%: 0.22729
        HD95                : 49.11905 +- 13.38340 | 30%: 37.56922
```
**Secondly**, I have trained my model (epochs=50) with CBCT as the fixed image and FBCT as the moving image. I have created two files for training and inference (train3.py-infer3.py and new_test.py-new_infer.py) which are slightly different. The results are stated below:

Using *train3.py* and *infer3.py*:
```bash
aggregated_results:
        LogJacDetStd        : 0.00342 +- 0.00045 | 30%: 0.00370
        TRE_kp              : 10.11501 +- 2.70331 | 30%: 11.72159
        TRE_lm              : 10.74309 +- 3.57581 | 30%: 11.45996
        DSC                 : 0.26895 +- 0.07659 | 30%: 0.22596
        HD95                : 48.96671 +- 13.29238 | 30%: 37.53550
```

Using *new_train.py* and *new_infer.py*:
```bash
aggregated_results:
        LogJacDetStd        : 0.00057 +- 0.00057 | 30%: 0.00114
        TRE_kp              : 10.10998 +- 2.72587 | 30%: 11.73434
        TRE_lm              : 10.72368 +- 3.47276 | 30%: 11.47623
        DSC                 : 0.27010 +- 0.07683 | 30%: 0.22781
        HD95                : 48.99913 +- 13.29813 | 30%: 37.38852
```
The results are quite similar, however I expected better values for each metric, considering that I reversed the roles of the moving and fixed images.

An example of image registration on patient 11:

![Fixed image (CBCT at the end of therapy), moving image (FBCT), transformed moving image in axis X](images_readme/new_infer.PNG)
*Image 8: Fixed image (CBCT at the end of therapy), moving image (FBCT), transformed moving image in axis X, deformation fields for each image.*

![Fixed image (CBCT at the end of therapy), moving image (FBCT), transformed moving image in axis Y](images_readme/new_infer_y.PNG)
*Image 9: Fixed image (CBCT at the end of therapy), moving image (FBCT), transformed moving image in axis Y, deformation fields for each image.*

![Fixed image (CBCT at the end of therapy), moving image (FBCT), transformed moving image in axis Z](images_readme/new_infer_z.PNG)
*Image 10: Fixed image (CBCT at the end of therapy), moving image (FBCT), transformed moving image in axis Z, deformation fields for each image.*

In both training and inference files, deformation field was very weak (mean, max and min values on oder of 10<sup>-7</sup>-10<sup>-9</sup>), which is the reason of poor alignment between the two images. So far, I have not been able to determine the cause of such low values.

**ADDITIONAL**

Out of curiousity, I have trained ModeTv2 model on our dataset (CT scans of 14 patients) in a way that I have trained it using dataset mentioned in the article (LPBA dataset - MR images of 40 patients). This way, the model compares each patient with others, creating (in our case) 10x9 image pairs. What is "bad" in this approach is that CBCTs are ignored.

My results:

![Fixed image (FBCT of patient 11), moving image (FBCT of anoter patient), transformed moving image in axis Z](images_readme/patient11_pre_X.PNG)
*Image 11: Fixed image (FBCT of one patient), moving image (FBCT of another patient), transformed moving image in axis X, deformation fields for each image.*

![Fixed image (FBCT of patient 11), moving image (FBCT of anoter patient), transformed moving image in axis Z](images_readme/patient11_pre_Y.PNG)
*Image 12: Fixed image (FBCT of one patient), moving image (FBCT of another patient), transformed moving image in axis Y, deformation fields for each image.*

![Fixed image (FBCT of patient 11), moving image (FBCT of anoter patient), transformed moving image in axis Z](images_readme/pacient11_pre.PNG)
*Image 13: Fixed image (FBCT of one patient), moving image (FBCT of another patient), transformed moving image in axis Z, deformation fields for each image.*

## Docker Information
**Step 1: Building docker image**:
```bash
docker build -f Dockerfile.modet -t <docker-image> .
```
Dockerfile.modet is in /home/marinab/modetv2/Dockerfile.modet

*Dockerfile.modet:*
```bash
FROM nvcr.io/nvidia/pytorch:24.06-py3

COPY ModeTv2 /workdir/ModeTv2

WORKDIR /workdir/ModeTv2/modet

RUN pip install pystrum natsort
RUN pip install scikit-image
RUN pip install .
```

**Step 2: Create and run new container:**
```bash
docker run \
    -it -d \
    --runtime=nvidia \
    --shm-size=1g \
    --ulimit memlock=1 \
    --ulimit stack=67108864 \
    --name <container-name> \
    -v /home/marinab/modetv2/ModeTv2:/workspace/modetv2 \
    -w /workspace/modetv2/ModeTv2 \
    <docker-image> python <file_name>
```                                  
**Step 3: Run the container interactively:**
```bash
docker run \
    -it --rm \
    --runtime=nvidia \
    -v /home/marinab/modetv2/ModeTv2:/workspace/ModeTv2 \
    -w /workspace/ModeTv2 \
    <docker-image> bash    
```  
**Step 4: Preprocessing Images:**

When inside of the container, run the following command to preprocess the data and prepare it for training:
```bash
python preprocessing.py --input_dir <path_to_input_directory> --output_dir <path_to_output_directory>
``` 
IMPORTANT: <path_to_input_directory> should be data of type .nii.gz

Example usage:
```bash
python preprocessing.py --input_dir ./Release_06_12_23/imagesTr --output_dir ./processed_files
``` 

**Step 5: Running the training:**
```bash
python train.py \
    --train_dir path/to/preprocessed/train_data/ \
    --val_dir path/to/preprocessed/val_data/ \
    --lr 0.001 \
    --weights 1 1 \
    --max_epoch 50 \
    --batch_size 1 \
    --save_dir path/to/file/for/saving/models/ \
    --task <choice>
```
**The values of lr, weights, max_epoch, batch_size in the command are the ones with which I trained my model.

IMPORTANT: save_dir will be saved inside experiments folder.

**Step 6: Running the inference:**
```bash
python infer.py \
    --output_dir folder_with_results \
    --val_dir path/to/preprocessed/val_data/ \
    --model_folder folder_with_models/ \
    --task <choice>
```

## Data Preparation
To prepare the data for training, it  to:

1. Convert the original data into .pkl format,
2. Resize each image by dividing its dimensions by 1.5 (x//1.5, y//1.6, z//1.5). This step was necessary to reduce the image size, as the original dimensions were too large for the training process (we devide each dimension by same factor to ensure that the image maintains its original proportions and does not become distorted),
3. Normalize the images to the range [0, 1] to ensure uniformity in pixel intensity values.

The data was divided into two classes, which are utilized during training:
1. For first part of task, we ignore CBCT at the end of therapy. FBCT and CBCT at the beginning of therapy are used as fixed and moving images.
2. For second part of task, we ignore CBCT at the beginning of therapy. FBCT and CBCT at the end of therapy are used as fixed and moving images.

## Train Commands
To run the training, one should run the following command (when inside of the container):
```bash
python train.py \
    --train_dir path/to/preprocessed/train_data/ \
    --val_dir path/to/preprocessed/val_data/ \
    --lr 0.001 \
    --weights 1 1 \
    --max_epoch 50 \
    --batch_size 1 \
    --save_dir path/to/file/for/saving/models/ \
    --task <choice>
```
where train_dir is directory of processed data for training, val_dir is directory of processed data for validation, l_r lreaning rate, max_epoch maximal number of epochs (min_epochs is set to 0 inside the train file), save_dir is a folder where models will be saved and choice is either "pre" or "post" depends on which images we are training (if training on FBCT and CBCT pre therapy use "pre", if training on FBCT and CBCT post therapy use "post").

**IMPORTANT**: There are three files for training:
1. train2.py - CBCT fixed image, FBCT moving image, 
2. train3.py and new_train.py (FBCT fixed image, CBCT moving image).

## Test Commands
To test the model, one needs to run command (when inside of the container):

```bash
python infer.py --output_dir folder_with_results --val_dir path/to/preprocessed/val_data/ --model_folder folder_with_models/ --task <choice>
```
where output_dir is folder where moving, fixed image and deformation field (under name *flow*) in original size (256, 192, 192, 3) and format (.nii.gz) will be saved. val_dir is path to procesed data for testing and model_folder is the folder where models from training are saved (in training that was under name save_dir), choice is either "pre" or "post" depends on which images we have trained our model (if trained on FBCT and CBCT pre therapy use "pre", if trained on FBCT and CBCT post therapy use "post")

**IMPORTANT**: There are three files for training and use inference files accordingly:
1. if using train2.py, run infer2.py,
2. if using train3.py, run infer3.py,
3. if using new_train.py, run new_infer.py

The model I have trained are saved at /home/marinab/data/experiments, where model, trained with:

1. train2.py (55 epochs) is saved under /home/marinab/data/experiments/New_Post_55_epoh_ModeTv2_cuda_nh(84211)_hd_6_ncc_1_reg_1_lr_0.0001_54r and /home/marinab/data/experiments/New_Pre_55_epoh_ModeTv2_cuda_nh(84211)_hd_6_ncc_1_reg_1_lr_0.0001_54r;
(150 epochs): /home/marinab/data/experiments/Post_150_epoh_ModeTv2_cuda_nh(84211)_hd_6_ncc_1_reg_1_lr_0.0001_54r and /home/marinab/data/experiments/Pre_150_epoh_ModeTv2_cuda_nh(84211)_hd_6_ncc_1_reg_1_lr_0.0001_54r,
2.  train3.py: /home/marinab/data/experiments/New_Post_55_epoh_ModeTv2_train3 and /home/marinab/data/experiments/New_Pre_55_epoh_ModeTv2_train3,
3. new_train.py: /home/marinab/data/experiments/New_Train_Post_ModeTv2 and /home/marinab/data/experiments/New_Train_Pre_ModeTv2

*If something is not working, please feel free to contact me on mb1928@studet.uni-lj.si*
