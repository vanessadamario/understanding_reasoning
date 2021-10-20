# How Modular Should Neural Module Networks Be for Systematic Generalization

The code is entirely in Python. The requirements to run the code are at   
```` requirements.txt````.

This code is highly inspired by the work of Bahdanau *et al.*

"CLOSURE: Assessing Systematic Generalization of CLEVR Models" https://arxiv.org/pdf/1912.05783.pdf, from which we forked the repo https://github.com/rizar/CLOSURE

and 

"SYSTEMATIC GENERALIZATION: WHAT IS REQUIRED AND CAN IT BE LEARNED? https://arxiv.org/pdf/1811.12889.pdf, from with we forked the repo
https://github.com/rizar/systematic-generalization-sqoop 


The repo is divided in several sub-folders.
In all of them, we compare different variants of NMNs
with different modularity and further non-modular networks.

**CLOSURE-master**: all the experiments of the CLEVR datasets.

**experiment_1**: single object in the scene, single NMN block. E.g., VQA question: _Is the object red?_  

**experiment_2**: one object per scene, two scenes in total, comparison, 3 NMN blocks in a tree configuration. 
E.g., question: _Is 6 the same color of 7?_ (data generation in progress). (All the runs of experiment_2 are generated through the code in experiment_4)

**experiment_3**: multiple objects per scene, single NMN block, same question as experiment_1  

**experiment_4**: multiple objects per scene, single NMN block,
comparison as in experiment 2 (data generation in progress)

**original_library**: SQOOP experiments, from Bahdanau, and two objects per scene case.
## VQA-MNIST
We trained and tested several NMNs: we generate different NMNs by changing 
the parameters in the experiments.py files.
We use these architectures across all the experiment_*

### main.py
Through the main.py, for each case (experiment_*) we generate the dataset, 
we specify the hyper-parameters for the experiments (everything gets saved in a json file, 
each experiment in the json has its unique identifier), 
we train those networks and test them. 
The pipeline is customized to run on a cluster, but it can be easily adapt by changing the *sh files. You can modify the output_path in the main.py file.

#### Check the path in main.py
Change output_path

### Dataset generation 
Data can be generated using the *.sh scripts. The split from MNIST is needed, 
and can be found at this link https://www.dropbox.com/s/y4v5vmxqfqi42nz/MNIST_splits.zip?dl=0

### Generate new experiments
We go in the experiment_* folder (should better be called case_* folder, among the four) 

python main.py --host-filesystem ** --experiment_index 0 --run gen_exp

The output of this operation is a folder : results, containing a train.json file, with all the necessary params to train a specific network on a dataset
(In case of experiment_2 you need to create a new folder, as all the code for dataset generation and training is in experiment_4)
Before this call, we need to specify the architecture hyper-parameters and the datasets, in experiments.py (dict_method_type dictionary) 

#### Parameters to change in `dict_method_type` (`experiments.py`)
In the following   
**architecture_name**  
 _parameters to change_ in the variable `dict_method_type`
The **architecture_name** does not appear anywhere in the code, but helps as a reference.
The other keys in this dictionary do not change across **experiment**, we specify only those that must be modified.

**all - all - all**  
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": False,   
                      "separated_module": False,  
                      "separated_classifier": False  
                     }
```
                             
**all - group - group** 
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": False,  
                      "separated_module": True,  
                      "separated_classifier": True  
                     }
```
                                                                         
**group - group - group**  
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,   
                      "separated_stem": True,  
                      "separated_module": True,  
                      "separated_classifier": True  
                      }  
```

**sub-task - sub-task - sub-task**   
```
dict_method_type = {"use_module": "residual",  
                      "feature_dim": # input dimensions (depends on the dataset),    
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": True,  
                      "separated_module": True,  
                      "separated_classifier": True  
                     }
```

**all - sub-task - all**   
```
dict_method_type = {"use_module": "residual",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": False,  
                      "separated_module": False,  
                      "separated_classifier": False 
                      }  
```

**all(bn) - all - all(bn)**  
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 1,  
                      "classifier_batchnorm": 1,  
                      "separated_stem": False,   
                      "separated_module": False,  
                      "separated_classifier": False  
                      }
```


**all(bn) - sub-task - all(bn)**   
```
dict_method_type = {"use_module": "residual",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 1,  
                      "classifier_batchnorm": 1,  
                      "separated_stem": False,  
                      "separated_module": False,  
                      "separated_classifier": False  
                      }
```

**all - sub-task/group - all**
```
dict_method_type = {"use_module": "mixed",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": False,  
                      "separated_module": True,  
                      "separated_classifier": False  
                      }
```

**sub-task - sub-task/group - all**
```
dict_method_type = {"use_module": "mixed",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": True,  
                      "separated_module": True,  
                      "separated_classifier": False  
                      }
```

**group - all - all**
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": True,  
                      "separated_module": False,  
                      "separated_classifier": False 
                      }
```

**sub-task - all - all**  
optional flag in `module_per_subtask=True`
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": True,  
                      "separated_module": False,  
                      "separated_classifier": False 
                      } 
```

**all - all - group**  
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": False,  
                      "separated_module": False,  
                      "separated_classifier": True  
                      }
```

**all - all - sub-task**    
optional flag in `module_per_subtask=True`
```
dict_method_type = {"use_module": "find",  
                      "feature_dim": # input dimensions (depend on the dataset),  
                      "stem_batchnorm": 0,  
                      "classifier_batchnorm": 0,  
                      "separated_stem": False,  
                      "separated_module": False,  
                      "separated_classifier": True  
                      }
```

`feature_dim = [3, 28, 28]` if experiment_1 and experiment_2 
`feature_dim = [3, 64, 64]` if experiment_3 and experiment_4

#### Parameters to change in experiments.py
In `experiments.py` there are some further variables which are important for the experiment generation phase.  

`experiment_case_list`, typically set to `[1]`, this variable can be `[0]` in case we want to generate a multi-task problem, or `[2]`, in the case of an enlarged VQA task    
`lr_array`, list of learning rates, typical values `[5e-3, 1e-3, 1e-4]`, or `[1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]`    
`batch_list`, list of batch sizes, `64`, `128`, `256`, `512` are typical values, depending on the size of the dataset    
`dataset_dict`, dictionary containing the dataset name, e.g.,   
```dataset_dict = {"dataset_name": ["dataset_15",  
                                 "dataset_16",  
                                 "dataset_17",  
                                 "dataset_18",  
                                 "dataset_19"]}  
```

### Run experiments 
`python main.py --host-filesystem _**_ --experiment_index _experiment_id_ --run train `
An important flag here, in case the experiment did not train for the specified amount of iterations, is    
`load_model (bool)`, if False, we train from scratch. Otherwise, we load the model at last iteration ./results/train_(experiment_id)/model, as we start the training from there.


## SQOOP experiments
You can either generate the dataset from the code, or download it from here https://www.dropbox.com/s/vfwaun1pyikeovq/sqoop-no_crowding-variety_1-repeats_30000.zip?dl=0

Run the code using the *.sh scripts in the sqoop-systematic-generalization_2objects folder (first commands are referred to the cluster and the use of singularity image).

## CLEVR experiments
Download the data as described here https://github.com/rizar/CLOSURE

Read and use the `launch_features_extraction.sh`

Run the code using the *.sh scripts (first commands are referred to the cluster and the use of singularity image)




