## A brief introduction to the composition and content of the code resources

### CoT Errors

​	All the contents of CRBench dataset, where the files with the suffix _f are aesthetic data files and are only for viewing.

### Refined CoT

	- Collider_error，Confounding_error，Measure_error，Mediation_error：These are the corresponding data sets after correcting the four CoT errors in the data sample. Among them, the files with the suffix "_f" are aesthetically pleasing data files and are only used for viewing.
	- dataset_collection.py：It is used to further merge the data set containing various error CoT into one file for the convenience of subsequent experiments.
	- dataset_organizer.py：Encapsulate and beautify the data entries after correcting the four types of CoT errors in the data samples.
 - Experiment：Compare the inference performance of large models with different parameter numbers in different families on the updated CoT.
    - Zero-shot_query.py，ZR_result：The LLM is required to conduct zero-shot direct question answering and its experimental results.
    - CoT_query.py，CoT_result：The LLM is required to think step by step and answer the questions and their experimental results.
    - CaCoT_query.py，CaCoT_result：The LLM is asked to think and answer questions with reference to the revised CoT and its experimental results.
    - templates：Prompt templates in all experiments.
    - LLM_validation.py：The large model is required to judge the correctness of the answer results of the large model, and the correct rate is calculated.
    - Data_collection.jsonl：Contains data entries after all four CoT errors.

### CoT_Causal_Refinement

​	Using QWen2.5-72B-Instruction model to correct various errors in the chain of causal thinking.
