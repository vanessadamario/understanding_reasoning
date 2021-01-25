# Analysis of Neural Module Networks (NMN)
Code for the analysis of Neural Module Networks (NMN) on VQA task. 
The repo will be divided in five folders. In all of them, we compare different variants of NMN architectures on tasks of increasing complexity

**experiment 1**: single object in the scene, single NMN block. E.g., VQA question: _Is the object red?_  

**experiment 2**: one object per scene, two scenes in total, comparison, 3 NMN blocks in a tree configuration. 
E.g., question: _Is 6 the same color of 7?_ (data generation in progress)

**experiment 3**: multiple objects per scene, single NMN block, same question as experiment_1  

**experiment 4**: multiple objects per scene, single NMN block,
comparison as in experiment 2 (data generation in progress)

**original library**: the experiments ran already for this case.

There are 5 types of architectures: (1) find, (2) half-separated find, (3) separated find, (4) residual, (5) separated residual. We use these architectures across all the experiments_* 

### main.py
Through the main.py, for each case (experiment_*) we generate data, we specify the hyper-parameters for the experiments, we train those networks and test them.

#### Check the path in main.py
Change output_path

### Generate new experiments
We go in the experiment_* folder (should better be called case_* folder, among the four) 

python main.py --host-filesystem ** --experiment_index 0 --run gen_exp

The output of this operation is a folder : results, containing a train.json file, with all the necessary params to train a specific network on a dataset

Before this call, we need to specify the architecture hyper-parameters and the datasets, in experiments.py (dict_method_type dictionary) 

#### Parameters to change in _dict_method_type_ (experiments.py)
In the following   
**architecture_name**  
 _parameters to change_ in the variable _dict_method_type_
The **architecture_name** does not appear anywhere in the code, but helps as a reference.
The other keys in this dictionary do not change across **experiment**, we specify only those that must be modified.

**find**  
_dict_method_type_ = {"use_module": "find",  
                         "feature_dim": # input dimensions (depend on the dataset),  
                              "stem_batchnorm": 1,  
                              "classifier_batchnorm": 1,  
                              "separated_stem": False,   
                              "separated_module": False,  
                              "separated_classifier": False  
                             }
                             
**half-separated find**  
_dict_method_type_ = {"use_module": "find",  
                                             "feature_dim": # input dimensions (depend on the dataset),  
                                             "stem_batchnorm": 0,  
                                             "classifier_batchnorm": 0,  
                                             "separated_stem": False,  
                                             "separated_module": True,  
                                             "separated_classifier": True  
                                            }  
                                                                         
**separated find**  
_dict_method_type_ = {"use_module": "find",  
                                        "feature_dim": # input dimensions (depend on the dataset),  
                                        "stem_batchnorm": 0,  
                                        "classifier_batchnorm": 0,   
                                        "separated_stem": True,  
                                        "separated_module": True,  
                                        "separated_classifier": True  
                                        }  
                             
**residual**   
_dict_method_type_ = {"use_module": "residual",  
                                  "feature_dim": # input dimensions (depend on the dataset),  
                                  "stem_batchnorm": 0,  
                                  "classifier_batchnorm": 0,  
                                  "separated_stem": False,  
                                  "separated_module": False,  
                                  "separated_classifier": False  
                                 }  
                                 
**separated residual**   
_dict_method_type_ = {"use_module": "residual",  
                                            "feature_dim": # input dimensions (depends on the dataset),    
                                            "stem_batchnorm": 0,  
                                            "classifier_batchnorm": 0,  
                                            "separated_stem": True,  
                                            "separated_module": True,  
                                            "separated_classifier": True  
                                           }  

"feature_dim" = [3, 28, 28] if experiment_1  
"feature_dim" = [3, 64, 64] if experiment_3  

#### Parameters to change in experiments.py
In _experiments.py_ there are some further variables which are important for the experiment generation phase.  

_experiment_case_list_, typically set to [1], this variable can be [0] in case we want to generate a multi-task problem, or [2], in the case of an enlarged VQA task    
_lr_array_, list of learning rates, typical values [5e-3, 1e-3, 1e-4], or [1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]    
_batch_list_, list of batch sizes, 64, 128, 256, 512 are typical values, depending on the size of the dataset    
_dataset_dict_, dictionary containing the dataset name, e.g.,   
dataset_dict = {"dataset_name": ["dataset_15",  
                                 "dataset_16",  
                                 "dataset_17",  
                                 "dataset_18",  
                                 "dataset_19"}  
       
### Run experiments 
python main.py --host-filesystem _**_ --experiment_index _experiment_id_ --run _train_ 
An important flag here, in case the experiment did not train for the specified amount of iterations, is    
--load_model (bool), if False, we train from scratch. Otherwise, we load the model at last iteration ./results/train_(experiment_id)/model, as we start the training from there.



