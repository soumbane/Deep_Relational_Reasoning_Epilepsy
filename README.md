# Deep Relational Reasoning for the Prediction of Language Impairments and Postoperative Seizure Outcomes of Children With Focal Epilepsy
![](networks/Deep_Relational_Reasoning_Epilepsy_TMI.jpg)

This is the official TensorFlow (Keras) implementation of the paper "[Deep Relational Reasoning for the Prediction of Language Impairment and Postoperative Seizure Outcome Using Preoperative DWI Connectome Data of Children With Focal Epilepsy](https://ieeexplore.ieee.org/document/9252947)" that was published in IEEE Transactions on Medical Imaging (TMI).

## Requirements
* Python >= 3.6
* [TensorFlow](https://www.tensorflow.org/) >= 2.0
* [keras-vis](https://raghakot.github.io/keras-vis/) (for GRAD-CAM visualization)

## Get Started (Preprocessing)
The following steps are required to replicate our work:

* The data of 51 patients has already been augmented to 510 augmentations per patient (per vector) using SMOTE augmentation.
* Preprocess dataset and generate the five folds (both training and testing) of all 51 patients and their augmentations - Use the raw data (private) of different densities (Densities used in paper: 01-High; 04-Medium; 08-Low) consisting of vectors for each of the 51 patients and 510 augmentations per patient (i.e., 510*51 + 51 = 26061 vectors). First divide the data and the corresponding augmentations into training (41 patients + augmentations) and testing (10 patients and augmentations). Then, convert each vector to a 116x116 connectome matrix for each patient. 
* Language Impairment Dataset (Regression) - Use the script `convert_script_new_data_regression_to_divide_into_5_folds.m` inside `Connectome_regression_data` folder to divide into training and testing (for the 5-fold cross validation) and convert the vectors into 116x116 shaped connectome matrices for both expressive and receptive scores. The data is saved to `Connectome_regression_data/processed_data/expressive_data_or_receptive_data/foldk`, where `k` is the fold number.     
* Seizure Outcome Dataset (Classification) - Use the script `convert_script_new_data_classification_to_generate_five_folds.m` inside `Connectome_classification_data` folder to divide into training and testing (for the 5-fold cross validation) and convert the vectors into 116x116 shaped connectome matrices for seizure outcome classification. The data is saved to `Connectome_classification_data/new_classification_data/foldk`, where `k` is the fold number.   

## Training

1. Define paths and hyper-parameters in configuration files.
* Refer to the files `config/COVID_JHU.conf` and `config/COVID_NYT.conf` for the data paths, hyper-parameters and model configurations used for training and testing. 
* The `sensors_distance` in the config files indicate the path to the adjacency matrix W.

2. Train the model
```
python train.py --epochs 100 --learning_rate 0.001 --expid 1 --print_every 20
```

## Testing

1. The pre-trained models could be found in `checkpoints/pretrained_models`
* Refer to the required folder `JHU or NYT`, `Infected or Deaths` for infected or death cases respectively and our model is in folder `STST`

2. Test the model
* An example for testing with `COVID_JHU` dataset's daily infected cases and `COVID_NYT` dataset's daily death cases with our model `STST` (name in code for STSGT model) is given here. The `... _best_model.pth` indicates the model with the lowest Mean Absolute Error (MAE) on the validation set. 
```
# For JHU Daily Infected cases data with our trained model
python test.py --checkpoint "checkpoints/pretrained_models/JHU_States_Infected/STST/exp_2_1654.67_best_model.pth"

# For NYT Daily Death cases data with our trained model
python test.py --checkpoint "checkpoints/pretrained_models/NYT_States_Deaths/STST/exp_1_19.06_best_model.pth"
```

## Notes
* Please choose the correct configuration file with the `DATASET` variable in both `train.py` and `test.py`.

## Cite
Please cite our paper if you find this work useful for your research:
```
@article{banerjee2020deep,
  title={Deep relational reasoning for the prediction of language impairment and postoperative seizure outcome using preoperative DWI connectome data of children with focal epilepsy},
  author={Banerjee, Soumyanil and Dong, Ming and Lee, Min-Hee and O’Hara, Nolan and Juhasz, Csaba and Asano, Eishi and Jeong, Jeong-Won},
  journal={IEEE transactions on medical imaging},
  volume={40},
  number={3},
  pages={793--804},
  year={2020},
  publisher={IEEE}
}
```
