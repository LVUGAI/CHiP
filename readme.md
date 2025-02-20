<div align="center">

#  CHiP
</div>

This is the Source Code of Paper: **CHiP: Cross-modal Hierarchical Direct Preference Optimization for Multimodal LLMs**


ICLR-2025 submission.


## Motivation and Method
Multimodal Large Language Models (MLLMs) still struggle with hallucinations sentation distributions reveals that multimodal DPO struggles to align image and text representations and to distinguish between hallucinated and non-hallucinated descriptions. To address these challenges, In this work, we propose a Cross-modal Hierarchical Direct Preference Optimization (CHiP) to address these limitations. We introduce a visual preference optimization module within the DPO framework, enabling MLLMs to learn from both textual and visual preferences simultaneously. Furthermore, we propose a hierarchical textual preference optimization module that allows the model to capture preferences at multiple granular levels, including response, segment, and token levels. We evaluate CHiP through both quantitative and qualitative analyses, with results across multiple benchmarks demonstrating its effectiveness in reducing hallucinations. On the Object HalBench dataset, CHiP outperforms DPO in hallucination reduction, achieving improvements of 52.7% and 55.5% relative points based on the base model Muffin and LLaVA models, respectively. 

 ![framework](./images/framework.png)

## Contents
- [Data](#data)
- [Install](#install)
- [Training](#training)
- [Evaluation](#evaluation)


## Data

### Training data
* There are several publicly available training datasets that include preference pairs for multimodal hallucinations. Here, we choose to use the RLHF-V-Dataset (Yu et al., 2024a;b) with 5k training samples as our training dataset.
### Evaluation data

The evaluation data can be downloaded from [here](https://drive.google.com/drive/folders/1gAauyipB4Zcc2hfJWH9G3DkwF-2Kx6MX) and placed in the playground/data directory

The evaluation dataset utilized in our work are listed below: 
* Object HalBench (ObjHal)
* MMHal-Bench (MMHal)
* HallusionBench
* AMBER



## Install

1. Clone this repository and navigate to source folder
```bash
cd CHiP
```

2. Build Environment 


```Shell


echo "Creating conda environment"
conda create -n CHiP python=3.10
conda activate CHiP

echo "Installing dependencies"
pip install -e .
```


### Training
Firstly, configure the training dataset name data_path and the checkpoint name output_dir.

* #### LLaVA CHiP/DPO Training
```Shell

bash scripts/chip.sh
bash scripts/cmdpo.sh
bash scripts/dpo.sh
```

* #### MUFFIN CHiP/DPO Training
```Shell

bash muffin/script/train/chip.sh
bash muffin/script/train/dpo.sh
```


## Evaluation

1. Run inference to generate responses

```py
python llava_inference.py --model_name {ckpt_name} --test_datasets {test_datasets} --eval_output {eval_output} 
python muffin/muffin/eval/inference.py --model_name {ckpt_name} --test_datasets {test_datasets} --eval_output {eval_output} 
```


2. Evaluate the generated responses.

```py
python ./muffin/muffin/eval/get_score.py --file {eval_output}
```

3. Evaluate Object HalBench 
* Prepare COCO2014 annotations. 
The evaluation of Object HalBench relies on the caption and segmentation annotations from the COCO2014 dataset. Please first download the COCO2014 dataset from the COCO dataset's official website.

```bash
mkdir coco2014
cd coco2014

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip annotations_trainval2014.zip
```

```py
# eval with gpt-4
python ./muffin/muffin/eval/eval_gpt_obj_halbench.py --cap_file {cap_file} --openai_key {api_key}
```

```py
python ./muffin/muffin/eval/summarize_gpt_obj_halbench_review.py --cap_file {cap_file} 
```

4. Evaluate MMHal Bench

* Please download the MMHal evaluation data [here](https://drive.google.com/file/d/1mQyAbeGgRyiVV6qjVkUI1uY_g9E-bDTH/view?usp=sharing), and save the file in `playground/data/`.
```py
# eval with gpt-4
python ./muffin/muffin/eval/eval_gpt_mmhal.py --response {response} --openai_key {api_key}
```
5. Evaluate HallusionBench

```py
# Firstly, prepare the response format 
python ./muffin/muffin/eval/eval_public.py # use eval_hallusionbench function
```
* Secondly, refer to [HallusionBench](https://github.com/tianyi-lab/HallusionBench.git)
```py
python HallusionBench/evaluation.py
```

6. Evaluate AMBER

```py
# Firstly, prepare the response format
python ./muffin/muffin/eval/eval_public.py # use eval_amber function
```
* Secondly, refer to [AMBER](https://github.com/junyangwang0410/AMBER.git)
```py
python AMBER/inference.py
```


## Acknowledgement

Our CHiP is developed based on the codebases of [LLaVA](https://github.com/haotian-liu/LLaVA) and [Muffin](https://github.com/thunlp/muffin), and we would like to thank the developers of both.