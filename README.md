Library for the analysis of Neural Module Networks (NMN) for VQA task. 
The repo will be divided in five. In all of them, we compare different variants of NMN architectures on tasks of increasing complexity
 

**experiment 1**: single object in the scene, single NMN block. 

**experiment 2**: one object per scene, two scenes in total, comparison, 3 NMN blocks in a tree configuration. 
E.g., question: _Is 6 the same color of 7?_ (data generation in progress)

**experiment 3**: multiple objects per scene, single NMN block

**experiment 4**: multiple objects per scene, single NMN block,
comparison as in experiment 2 (data generation in progress)

**original library**: the experiments ran already for this case.


There are five types of architectures. To use a specific architecture, 
the parameters in experiments.py (dict_method_type dictionary) must be changed. In particular
**architecture_name**: _parameters to change_
The **architecture_name** does not appear anywhere in the code, but helps as a reference.
The other keys in this dictionary do not change across **experiment**, we specify only those that must be modified.

**find**: dict_method_type = {"use_module": "find",
                              "feature_dim": # input dimensions (depend on the dataset)
                              "stem_batchnorm": 1,
                              "classifier_batchnorm": 1,
                              "separated_stem": False,
                              "separated_module": False,
                              "separated_classifier": False
                             }
                             
**half-separated find**: dict_method_type = {"use_module": "find",
                                             "feature_dim": # input dimensions (depend on the dataset)
                                             "stem_batchnorm": 0,
                                             "classifier_batchnorm": 0,
                                             "separated_stem": False,
                                             "separated_module": True,
                                             "separated_classifier": True
                                            }
                                                                         
**separated find**: dict_method_type = {"use_module": "find",
                                        "feature_dim": # input dimensions (depend on the dataset)
                                        "stem_batchnorm": 0,
                                        "classifier_batchnorm": 0,
                                        "separated_stem": True,
                                        "separated_module": True,
                                        "separated_classifier": True
                                        }
                             
**residual**: dict_method_type = {"use_module": "residual",
                                  "feature_dim": # input dimensions (depend on the dataset)
                                  "stem_batchnorm": 0,
                                  "classifier_batchnorm": 0,
                                  "separated_stem": False,
                                  "separated_module": False,
                                  "separated_classifier": False
                                 }
                                 
**separated residual**: dict_method_type = {"use_module": "residual",
                                            "feature_dim": # input dimensions (depends on the dataset)
                                            "stem_batchnorm": 0,
                                            "classifier_batchnorm": 0,
                                            "separated_stem": True,
                                            "separated_module": True,
                                            "separated_classifier": True
                                           }

"feature_dim" = [3, 28, 28] if experiment_1
"feature_dim" = [3, 64, 64] if experiment_3

In _experiments.py_ there are some further variables which are important for the experiment generation phase.

experiment_case_list, typically set to [1], this variable can be [0] in case we want to generate a multi-task problem, or [2], in the case of an enlarged VQA task

lr_array, list of learning rates, typical values [5e-3, 1e-3, 1e-4], or [1e-1, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5]

batch_list, list of batch sizes, 64, 128, 256, 512 are typical values, depending on the size of the dataset

dataset_dict, dictionary containing the dataset name, e.g., 
dataset_dict = {"dataset_name": ["dataset_15",
                                 "dataset_16",
                                 "dataset_17",
                                 "dataset_18",
                                 "dataset_19"}
       