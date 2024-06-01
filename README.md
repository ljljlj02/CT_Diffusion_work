Dataset
The dataset should contain pairs of high-noise and low-noise CT images in NIfTI format. 
Usage
Data Preparation
Ensure that you have your NIfTI files organized and update the file paths in the script accordingly.

Model Training
To train the model, run:
python train.py

Model Evaluation
To evaluate the model, run
python evaluate.py

Results
The model's performance is evaluated using the Gamma 3%/3mm/10% criterion. The results are printed in the console after evaluation.
