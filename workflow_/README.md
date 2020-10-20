---INTRODUCTION---
Age-Invariant Face Recognition with CNN architecture is utilized 

workflow folder includes: 
Brief EDA
Experiments folder for jupyter notebooks
Feature engineering script with explanations
Any results (see below)
Any conclusions (in light of other workflows; see below)

---RESULTS---
Brief EDA results: 
    "Unknown" may represent more than one individual across their lifetime
    Max Age range of any given individual in dataset may span from infant to elder
    
Experiment_2a:
    only 365 total faces detected out of 633 total images (same as original dataset)

---CONCLUSIONS---
Model will require feature to identify individuals over their lifetime

Set dlib simple_crop to False and run again, might detect more faces; there should be at least 633 detected faces