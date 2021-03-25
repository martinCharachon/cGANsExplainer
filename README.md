# Leveraging Conditional Generative Models in a General Explanation Framework of Classifier Decisions

Reproductible code for using the visual explanation methods of classifier decisions proposed in "*Leveraging Conditional Generative Models in a General Explanation Framework of Classifier Decisions*
"
## Summary

In this repository, we propose the codes for:
- Classification models used in the different task (optimization and architectures)
- The visual explanation techniques presented in our paper (optimization and architecture): CyCE and SyCE
- Visualization of explanation maps and generated images
- Generation and evaluation of our visual explanation methods

We offer the complete use case for the digits identification problem (see *mnist*). We describe in the following section how to run the different optimization and generate results.

Note that we only give the splits used for Pneumonia detection and Brain tumor localization. Users should create the database following the description from the paper and the dataset schema described in the so-called section.
- **Brain tumor** database used in the paper can be downloaded from the Medical Segmentation Decathlon (http://medicaldecathlon.com/)
    - Only T1gd sequence is used in our paper
    - Initial data comes with a four levels annotations mask: *"0" = background, "1" = edema, "2" = non-enhanced tumor, "3" = enhanced tumor*
    - We transform it into a binary mask: "0"+"1" to "0" (background) and "2"+"3" to "1" (tumor); from which we can derive the class label for each slice of the volume ("0": no tumor / "1": tumor).
- **Pneumonia** database can be downloaded from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    - Initial data comes with 3 class labels. We only used the healthy and pathological cases.

## Set up 

### Create an environnement
```
conda create -n cgan-explain-env python=3.7
```
### Work in the new environment
```
conda activate cgan-explain-env 
```

### Requirements
```
pip install -r requirements.txt
```
For MNIST digits problem, the optimization of classifier, interpreter and VAE embeddings can be run on NVIDIA GeForce MX 130

For the paper, the different optimizations were performed on 8 cpus Intel 52 Go RAM 1 V100.

## Train a Classifier
To train the classifier (to be interpreted), run the code `run_optimization_classifier.py`. Users must pass a configuration file that follows the structure of *./mnist/config_classifier.json*. For mnist, the training can be run from scratch with:
```
python run_optimization_classifier.py --config-file "./mnist/config_classifier.json"
```
Note: 
- For the digits task, we let a trained classifier in directory *./mnist/reference_models/classifier/*
## Optimize the Interpreter
To train the both generators and discriminators of SyCE technique, the user should run the following command (for mnist):
```
python run_optimization_interpreter.py --config-file "./mnist/config_syce.json" \
                                        --ref-model-settings-json-path "./mnist/classifier_settings.json" \ 
                                        --split-specific-path "./mnist/split/split_prediction_specific.json"
```
Notes :
- The user should pass the path to the classifier settings to interpret (here for mnist: *./mnist/classifier_settings.json*)
- If the split specific to the classifier's decision already exists (*./mnist/split/split_prediction_specific.json*), pass it to the configuration instead of the initial split (*./mnist/split/split_mnist_3_8.json*). Then you don't need the last argument ``--split-specific-path`` to start the optimization.
- Parameters of the different models architecture, losses or other training settings can be changed in the configuration file e.g. *./mnist/config_interpreter.json*.
- For the digits task, trained generators are found in directory *./mnist/reference_models/interpreter/*

Similarly to optimize CyCE run:
```
python run_optimization_interpreter.py --config-file "./mnist/config_cyce.json" \
                                        --ref-model-settings-json-path "./mnist/classifier_settings.json" \ 
                                        --split-specific-path "./mnist/split/split_prediction_specific.json"
```
Notes :
- Two slightly different optimizations can be run via the parameter `"optimization-type"` in the training configuration. Similar results are obtained (default="together"). Choose between:
    - "together": the two generators are optimize in a single step given a batch of images predicted as "0" and another predicted as "1".
    - "separated": step 1: the two generators are optimized given a batch of images predicted as "0". step 2: generators are optimized given a batch of images predicted as "1". 
## Optimize the Discriminative VAE
The VAE used for the evaluation of domain translation (and distribution matching between generated and real images), can be trained with:
```
python run_optimization_interpreter.py  --config-file "/mnist/config_vae.json" \
                                        --ref-model-settings-json-path "./mnist/classifier_settings.json"
```
Notes: 
- For the digits task, trained generators are found in directory *./mnist/reference_models/vae/*
- For the brain tumor localization, the basic vae encoder (``src.vae.vae_encoder.VAEEncoder``) achieves poor "class" discrimination. User should prefer ``src.vae.vae_inception_encoder.VAEInceptionEncoder`` passing *"VAEInceptionEncoder"* in the configuration file (see ``src.vae.vae_inception_encoder.py`` for the parameters). 

## Generate Interpretations
The same (or similar) configuration can be used to both generate, evaluate the localization or visualize the explanation produced by our techniques (see *./mnist/config_generate_interpretation.json* for mnist). 

Here we use the trained generators (see above) to interpret the classifier (see the configuration file for mnist).

Notes:
- In *./mnist/config_generate_interpretation.json*, we give a configuration to generate visual explanation for a trained SyCE method. Trained generators are found in *./mnist/reference_models/interpreter/SyCE*.
- The user can choose to either compute the visual explanation as:
    - |Stable - Adversarial| (as defined in the paper). Set in the configuration file `"explanation_def": "st - adv"`.
    - |Original - Adversarial|. Set in the configuration file `"explanation_def": "ori - adv"`.
- To generate visual explanations for CyCE, use the models from *./mnist/reference_models/interpreter/CyCE* and set `"explanation_def": "ori - adv"`.

### Generate and Evaluate Interpretations
```
python generate_and_evaluate_interpretation.py --config-file "./mnist/config_generate_interpretation.json" \
                                               --ref-model-settings-json-path "./mnist/classifier_settings.json" 
```
Notes:
- User can specify the split set with argument ``--indexes-name`` (by default set to *"test_indexes"*).
- For the digits problem, we do not have access to localization annotation so we can not perform the localization evaluation.


### Visualize Interpretations
```
python visualize_interpretation.py --config-file "./mnist/config_generate_interpretation.json" \
                                   --ref-model-settings-json-path "./mnist/classifier_settings.json" \
                                   --threshold 95
```
Notes:
- Idem: user can specify the split set with argument ``--indexes-name``.
- The threshold indicates the percentile at which the explanation map will be thresholded for visualization 
```
t = numpy.percentile(heatmap, threshold) 
binary_heatmap = numpy.where(heatmap > t, 1.0, 0.0)
```

## Domain Translation Evaluation
### Generate adversaries (and stable) images for the different methods.
Here we give a script to generate images from MGen [1], SAGen [2], CyCE and SyCE. 
The trained models used for MGen [1], SAGen [2], CyCE and SyCE are found in *./mnist/reference_models/interpreter/*.

To execute the generation for these methods, 
user can run:
```
python generate_multi_interpretation_techniques.py --config-file "./mnist/config_generate_multi_interpretations.json" \
                                                   --ref-model-settings-json-path "./mnist/classifier_settings.json"
```
Note:
- A version is saved at *./mnist/interpreter_outputs/multi_methods/multi_generation_saved.h5*

### Generate Embeddings
To generate embeddings with the trained VAE, run:
```
python generate_embeddings.py --config-file "./mnist/config_generate_embeddings.json"
```
Notes:
- We assume that you have access to different generated images (as well as real images)
- Here we use the generated images from from MGen [1], SAGen [2], CyCE and SyCE found in *./mnist/interpreter_outputs/multi_methods*
- A version is saved at *./mnist/embeddings_outputs/multi_generation_embeddings_saved.h5*

### Visualize and Evaluate Embeddings
To  visualize the 2-dimensional PCA from the embeddings of real and generated images:
```
python visualize_embeddings_pca.py --embeddings-path "./mnist/embeddings_outputs/multi_generation_embeddings_saved.h5" \
                                   --split-path "./mnist/split/split_mnist_3_8.json"
```
To Evaluate the Fréchet Distance on the embedded mean (from VAE embeddings), run:
```
python evaluate_embeddings.py --embeddings-path "./mnist/embeddings_outputs/multi_generation_embeddings_saved.h5" \
                              --results-path "./mnist/embeddings_outputs/fdmu_results.json" \
                              --evaluation-metrics "FD" \
                              --split-path "./mnist/split/split_mnist_3_8.json"
```
To Evaluate the Jenson-Shannon Distance from the PCA applied on the embedded mean of real and gen. images, run:
```
python evaluate_embeddings.py --embeddings-path "./mnist/embeddings_outputs/multi_generation_embeddings_saved.h5" \
                              --results-path "./mnist/embeddings_outputs/js_results.json" \
                              --evaluation-metrics "JS" \
                              --split-path "./mnist/split/split_mnist_3_8.json"
```

## Dataset and Split Schema

To be able to run our codes on new datasets (including those for Pneumonia detection and Brain tumor loc.), the users should follow the next key points:
- Use package `h5py` to store image data and annotations (class label, localization annotations ...)
- Data and annotations should be accessed as follow : 
```
import h5py

# Reference example for mnist
ref = ["mongo_id_11982", 0]

# Open hdf5 database
db = h5py.File(db_path, "r")

# Access to data and annotations 
data = db[f"data/{ref[0]}/{ref[1]}/data"][()]
label = db[f"data/{ref[0]}/{ref[1]}/label/classification"][()]
bbox = db[f"data/{ref[0]}/{ref[1]}/label/localization"][()]
label = db[f"data/{ref[0]}/{ref[1]}/label/segmentation"][()]

# Close hdf5 database
db.close()
``` 
- Follow the split schema of mnist digits found in *./mnist/split/split_mnist_3_8.json*. The split is composed of 3 lists: *train_indexes*, *val_indexes*, *test_indexes*. Each reference from those lists points to a single case (image + annotation(s)) with structure `["case_id", int(slice_number)]`. Except for Brain tumor localization (or other problems involving slices or similar definition) , the slice number is always 0.
- Note that for Brain tumor loc. the slices referenced in the split (*./BrainTumorLoc/split/split.json*) correspond to slices along the axial axis of the **resampled volume** (and not the original one). As described in the suppl. mat., we resampled the initial MRI (Tgd sequence) volumes from (155, 240, 240) to (145, 224, 224).
- For Pneumonia, we resized the original image from (1024, 1024) to (224, 224) and then stored them in the hdf5 database file.
- Note that for the optimization of the interpreter you can also directly use the split that include the classifier decisions specificity (see *./mnist/split/split_prediction_specific.json*). If you pass the regular split schema (use for classifier or vae training), the prediction specific split will be computed before the interpreter's optimization starts.
- Bounding box annotations (in Pneumonia detecion) should be strored in format:
```
bbox = numpy.array(
        [[xmin_1, ymin_1, xmax_1, ymax_1],
         [xmin_2, ymin_2, xmax_2, ymax_2]])  
```
- Segmentation annotation (for Brain tumor loc.) should be a binary mask.

## References

[1]: Dabkowski, Piotr, et Yarin Gal. « Real Time Image Saliency for Black Box Classifiers ». NIPS, 2017.

[2]: Charachon, M.,  Hudelot, C., Cournède, P.-H., Ruppli, C., and Ardon, R. « Combining similarity and adversarial learning to generate visual explanation: Application to medical image classification ». ArXiv, abs/2012.07332,2020..