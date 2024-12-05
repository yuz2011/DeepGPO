# User guide
## Requirements and installation
The DeepGPO.yml lists all the dependencies of the DeepGPO. To quickly set up an environment for our model, use the following command
conda env create -f DeepGPO.yml
```
Installing a specific package usually takes a few minutes on a standard desktop and setting up the complete environment may take approximately an hour. Once the library is installed, you can verify the installation by importing the library in Python and checking the version number:
```
import numpy as np
print(np.__version__) 
```
## Example data
The example data can be found in the `demo_data` folder. 
Spectra files:
`mgf_demo/demo.mgf`

In order to obtain an MGF file, you will need to employ a program or tool with the ability to convert your data into MGF format. Here, the MGF files are from the searching process of pGlyco3 (release: pGlyco3.1, https://github.com/pFindStudio/pGlyco3/releases). When multiple MGF files are stored under a specified folder, they can be automatically retrieved in one go.

Sequence searching software results:
`pGlycoDB-GP-FDR-Pro-Quant-Site_demo.txt`

The `pGlycoDB-GP-FDR-Pro-Quant-Site_demo.txt` file contains the pGlyco3 results for glycopeptides. In practical application, the pGlycoDB-GP-FDR-Pro-Quant-Site.txt file of the pGlyco3 results can be accessed to obtain the desired file. Furthermore, if you have search results from other tools, they can be utilized by converting their format to match that of pGlyco3.
## Run the release version of DeepGPO using the command line
The entire process contains three steps: 
1.	Data processing
2.	Model training
3.	Prediction 

The related code files are appropriately numbered for ease of use, such as `1_dataset_format_NO.py`.

These scripts are executed in a command-line interface. Advanced users can also adapt these commands for other command-line interfaces.
### Glycopeptide MS/MS spectra: pre-processing, training, and prediction
(1)	Entry to the folder including DeepGPO code files.
Users can navigate to the relevant folder using a command such as cd D:\code\DeepGPO_code. The path “D:\code\DeepGPO_code” signifies the directory containing the Python scripts for DeepGPO.

(2)	Pre-processing: Convert the library search results (.txt) and experimental glycopeptide spectra (.mgf) into files containing spectral data (.csv).

```
python 1_dataset_format_NO.py --datafold D:/code/demo_data/NO/demo/ --dfname pGlycoDB-GP-FDR-Pro-Quant-Site_demo.txt --mgfdatafold mgf_demo --output_name demo_data_1st.csv --mgfsourceorign pGlyco3  --only_duplicated Retained_all  --enzyme None
```

The description of the parameters of the command line:

`--datafold`: This parameter denotes the directory where both the pGlyco3 identification results (pGlycoDB-GP-FDR-Pro-Quant-Site.txt) and experimental spectra (.mgf) are stored.

`--dfname`: This parameter signifies the file name of the pGlyco3 identification results.

`--mgfdatafold`: This parameter corresponds to the folder name for all .mgf files. (The .mgf files are located within the folder indicated by datafold+mgfdatafold)

`--output_name`: This parameter sets the name for the output file.

`--mgfsourceorign`: This parameter allows you to select the format for .mgf files. The options are “pGlyco3” and “MsConvert”. The default is “pGlyco3”.

`--only_duplicated`: This parameter determines the method for removing duplicate identification results. It has three possible values: “Duplicated”, “Drop_duplicated”, and “Retained_all”. “Duplicated” only keeps duplicated identification results. “Drop_duplicated” retains a single identification result, choosing the one with the smallest “TotalFDR” if duplicates exist. “Retained_all” keeps all identification results. The default is “Drop_duplicated”.

`--enzyme`:This parameter specifies the O-glycoproteases used for calculating weights specific to O-glycoprotease-digested glycopeptides.Supported options include enzymes such as None([]), OgpA ([1]), SmE ([1, -1]), StcE ([1, -2]), and others, where the first element is the enzyme name and the second represents the cleavage sites. A negative number indicates counting from the end, where `-1` refers to the last amino acid. If additional cleavage patterns are needed, they can be added to the weight.py script.


`--not_use_weights`:To disable the loss re-weighting method, simply append --not_use_weights to the command. This will skip the weight calculation step.

(3)	Model Training: Train DeepGPO model for intact glycopeptide MS/MS prediction
For ease of use, users have the option to utilize our provided trained model for immediate rough testing, thereby bypassing the model training phase. We have uploaded this trained model (DeepGPO_model) to [Google Drive](https://drive.google.com/drive/folders/1KzJm4bE3RkdnZ1hLB2wjQyXoOtP69Zse?usp=drive_link).
For those with access to other datasets, model training can also be conducted using your own datasets or extensive datasets downloaded from public databases. This provides more flexibility and customization, allowing the model to better adapt to various types of data.

```
python 2_train_byBY.py  --model_ablation DeepGP --DeepGP_modelpath D:/code/DeepGP_model/epoch-124_step-47988_mediancos-0.927153.pt  --testdata alltest --task_name demo --folder_path D:/code/demo_data/  --type NO --trainpathcsv demo/train_combine.csv --ms2_method cos_sqrt --pattern *_data_1st.csv 
```

The description of the parameters of the command line:

`--model_ablation`: This parameter chooses the model to be used. Options include DeepGP, BERT and Transformer, with the default being “DeepGP”.

`--DeepGP_modelpath`: This parameter is used for DeepGP, a pre-trained model trained on human&mouse datasets previously published and available at [DeepGP GitHub Release Page](https://github.com/lmsac/DeepGP/releases). It can also be directly accesed through [Google Drive](https://drive.google.com/drive/folders/1yXYXO7MD4Mt8yxfTdFpk-SM-wjRkXYL_?usp=drive_link). Please download it and specify the model path for DeepGP. If the BERT or Transformer models are used instead of DeepGP, this parameter can be ignored.

`--testdata`: This parameter indicates the test data. If any files within the specified directory contain this keyword in their names, they will be excluded from the training set and set aside for testing purposes. If no file contains the term, all will be used for training. The default is “alltest”.

`--task_name`: This parameter sets the name for the task.

`--folder_path`: This parameter specifies the name of the main folder.

`--type`: This parameter identifies the type of the dataset. Since the data is organized as main_folder_name/type/folder_name/, data within the specified main_folder_name/type/ will be chosen as the training datasets. Multiple folders within the main_folder_name/type/ can be selected at once.

`--trainpathcsv`: This parameter denotes the output filename for the processed training data. Files within the specified directory that match a certain suffix (as defined by another parameter, "pattern") will be aggregated and processed. The resulting training dataset will be saved with the filename provided in this parameter.

`--ms2_method`: This parameter determines the metric used for DeepGPO. Options are “cos_sqrt”, “cos”, and “pcc”, representing cosine similarity with a square root transformation, cosine similarity, and Pearson correlation coefficient, respectively.

`--pattern`: This parameter denotes the suffix for the training datasets. Files bearing this suffix within the folder name will be employed as training datasets.

`--lr`: This parameter adjusts the learning rate, defaulting to 0.0001.

`--device`: This parameter sets the device number for CUDA. If no GPU is available, the CPU will be used by default. The default device number is 0.

Advanced users can also adjust the settings available in the code’s utilities (utils.py). The model architecture can be easily modified using keywords. For example, if you type `GNN_global_ablation=GIN`, you will change the GNN architecture for glycan global representation into GIN. If you type `GNN_edge_ablation=GIN`, it means the GNN architecture of glycan B/Y ions intensity prediction is GIN. Users can also change the dimension and layer number by inputting their self-defined number. For example, `GNN_edge_num_layers=7` means that the layer number of GNN for glycan B/Y ions intensity prediction is equal to 7. You can replace “7” with the number of layers you want.
We highly recommend training DeepGPO using larger datasets. The demo dataset provided contains only 5 unique spectra. While the code can be successfully implemented with this dataset, it is not large enough to effectively train a model. Therefore, for optimal performance and accuracy, consider using larger datasets.

(4)	Prediction: Predict MS/MS glycopeptide spectra with trained model
```
python 3_replace_predict_byBY.py --trainpathcsv D:/code/demo_data/NO/demo/demo_data_1st.csv --datafold D:/code/demo_data/NO/demo/ --bestmodelpath D:/code/DeepGPO_model/epoch-99_step-62172_mediancos-0.938538.pt  --savename demo --ms2_method cos --postprocessing off
```

The description of the parameters of the command line:

`--trainpathcsv`: This parameter specifies the input file name for the test dataset.

`--datafold`: This parameter denotes the directory name for the output files.

`--bestmodelpath`: This parameter sets the model path file name. Here, we use the trained model which can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1KzJm4bE3RkdnZ1hLB2wjQyXoOtP69Zse?usp=drive_link) as noted above.

`--device`: This parameter indicates the device number for CUDA. If no GPU is available, the CPU will be used by default. The default device number is 0.

`--savename`: This parameter provides the prefix for the output file names.

`--ms2_method`: This parameter decides the metric used for DeepGPO. Options include “cos_sqrt”, “cos”, and “pcc”. These represent cosine similarity with a square root transformation, cosine similarity, and Pearson correlation coefficient, respectively. The default is: "cos_sqrt".

`--postprocessing`: This parameter determines whether post-processing is required (“on/off”). If set to “on”, the output will include files containing all predicted fragments and their corresponding intensities, along with all experimental fragments and their intensities.

It takes about 1 second to predict 40 spectra on a single RTX 3090 GPU.

# Contact
All the demo data and code are provided. A sentence with gray background indicates it is a sentence of code. If you have any further questions, please don't hesitate to ask us via: liang_qiao@fudan.edu.cn. You could also go to the homepage of DeepGPO on GitHub to ask a question.
