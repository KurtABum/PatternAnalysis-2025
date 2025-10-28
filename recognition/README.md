# BioLaySumm: Radiology Report to Layman Summary

## Overview

**BioLaySumm** is a transformer-based natural language processing (NLP) pipeline designed to convert complex radiology reports into easily understandable lay summaries. The goal is to make medical findings more accessible to patients and non-experts, improving patient comprehension and engagement.


## Dataset Description

The project uses the **BioLaySumm2025-LaymanRRG-opensource-track** dataset, which is specifically designed for radiology report simplification.  
It is split into:
- **Training:** 150,000 examples  
- **Validation:** 10,000 examples  
- **Testing:** 10,500 examples  

Each record contains **four columns**:
1. **source** – metadata about data origin (not used in training)  
2. **image_path** – reference to the related image (not used in training)  
3. **radiology_report** – the full expert-written radiology report  
4. **layman_report** – the corresponding simplified lay summary  

During training, only the `radiology_report` (input) and `layman_report` (target) columns are used.  
The `radiology_report` serves as the expert-level medical input, while the `layman_report` represents the desired human-understandable output.  
This pairing allows the model to learn how to translate clinical jargon and findings into clear, accessible language.

## Setup and Usage

### 1. Install Visual Studio Code

Visual Studio Code is recommended for editing, running, and debugging the project.  
 

---

### 2. Install CUDA (For NVIDIA GPUs)

If you have a compatible NVIDIA GPU, installing CUDA will significantly accelerate training and inference.  

Install the appropriate CUDA toolkit for your GPU: https://developer.nvidia.com/cuda-toolkit

Test PyTorch CUDA support:
```bash
python -c "import torch; print(torch.cuda.is_available())"

```
This should return True

### 3. Clone the Repository and Create a Virtual Environment
```bash
git clone https://github.com/KurtABum/PatternAnalysis-2025.git
cd recognition

conda create -n flan-t5 python=3.11
conda activate flan-t5
```

### 4. Install Dependencies

With the `flan-t5` environment active, install the packages:

```bash
# Core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install datasets
pip install sentencepiece
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install tqdm
pip install wandb

```

### 5. Run Traning Loop
After these are installed, the traning is ready to start. To train the full dataset run:
```bash
python train.py
```

### 6. Test Model
After training, run `predict.py` to display 5 example outputs, showing both the original input and the translated version.
```bash
python predict.py
```

## File Structure

- **train.py** – Handles training the model. Loads the dataset, initializes the model, and runs the training loop with logging.  
- **predict.py** – Runs the trained model to show example predictions. Displays a few inputs along with their translated outputs.  
- **modules.py** – Contains reusable functions, helper routines, and model components that are shared across scripts.  
- **dataset.py** – Manages data loading and preprocessing. Converts raw data into batches suitable for training or evaluation.

## Data Cleaning and Preprocessing

The dataset is cleaned and preprocessed in `dataset.py` before training:  

- Removes invalid or corrupted entries.  
- Normalises text or input data to a consistent format.  
- Tokenises or encodes inputs for the model.  
- Splits the data into batches for efficient training and evaluation.

## Flan-T5 Model

Flan-T5 is a large language model developed by Google that has been fine-tuned to follow a wide range of instructions. It uses an encoder-decoder transformer architecture, which makes it well suited for tasks like translation or converting one type of input into another.

The encoder reads and understands the input, while the decoder generates the output based on that understanding. Both parts use multiple layers of transformers with attention mechanisms to capture complex patterns. A typical base version has hundreds of millions of parameters, many layers, and large hidden dimensions, allowing the model to learn sophisticated input-output mappings.

In this project, Flan-T5 can take input examples and produce accurate translations or transformations with minimal additional training. Its strengths include strong instruction-following, flexibility for different tasks, and the ability to generalize from small datasets. On the downside, Flan-T5 can be **resource-intensive**, requiring significant memory and computational power, and fine-tuning large versions can be slow or expensive.


## Fine-Tuning Strategy

For this project, a Flan-T5-small model was fine-tuned for translating or transforming input sequences. The model configuration included:

- **Layers & hidden size:** The Flan-T5-small base model has 6 transformer layers and a hidden dimension of 512 units. This setup provides sufficient capacity to model complex relationships in the input data while keeping training manageable.

- **Classifier / output head:** A custom classification head was added on top of the decoder outputs to map predictions to the target token space, ensuring the model generates outputs that align with the task-specific labels.

- **Loss function:** The model was trained using **cross-entropy loss**, which measures how well the predicted tokens match the target tokens. This loss is effective for sequence-to-sequence tasks because it encourages the model to assign high probabilities to the correct outputs.

- **Fine-tuning strategy:** Parameter-efficient fine-tuning techniques, such as adapters or LoRA, were applied to update only a subset of the model weights. This reduces memory requirements and training time while still allowing the model to adapt effectively to the task.

- **Epochs:** The model was trained for 5 epochs, which balances learning task-specific patterns while minimizing the risk of overfitting on the relatively small dataset.

- **Optimizer & learning rate:** AdamW was used with a learning rate of 5e-6, providing stable updates suitable for fine-tuning large pretrained models.

This configuration was chosen to allow efficient adaptation of Flan-T5-small to the task while keeping training time and GPU memory requirements reasonable.

## Model Saving

During training, the model is saved to a directory (default: `best_model`) whenever the current epoch achieves a **better validation ROUGE-Lsum score** than all previous epochs. This ensures that only the best-performing version of the model is retained.  

The following files are saved:  

- **`config.json`** – Stores the model architecture and configuration parameters.  
- **`generation_config.json`** – Contains settings used for text generation, like maximum length or beam size.  
- **`model.safetensors`** – Contains the trained weights of the model in a safe and efficient format.  
- **`special_tokens_map.json`** – Maps special tokens (like padding, start/end tokens) to their intended purpose.  
- **`tokenizer_config.json` & `tokenizer.json`** – Include the vocabulary, tokenization rules, and other settings needed to convert text to input tokens and back.  

In `predict.py`, you can change the path to this saved model directory to load a different trained model and generate outputs. This allows you to easily switch between versions of the model for testing or demonstration. For example:  

```python
wrapper = FlanT5Summarizer(model_name="path/to/best_model", device=DEVICE)
```

## Training and Testing

Training and evaluation for this project were performed on a Windows PC equipped with an **RTX 5070 GPU with 12GB of VRAM**. Due to GPU memory limitations, the batch size was set to 2, and full-length inputs and outputs were used to ensure accurate training and evaluation.

- **Training:** The Flan-T5-small model was trained for 5 epochs using parameter-efficient fine-tuning techniques. Training the full dataset took approximately **36 hours** on this hardware.

- **Validation & Testing:** After each epoch, the model was evaluated on a validation set to monitor performance and select the best model based on ROUGE scores. Final evaluation was conducted on a held-out test set to report the model’s predictive performance.

## Example Analysis 1

### Expert Report
Cardiomegaly. Left subclavian pacemaker with atrial and ventricular leads. Elevation of the right hemidiaphragm. Costophrenic angle blunting with pleural effusion.

### Generated Lay Summary
The heart is enlarged. There's a pacemaker on the left side of the chest with wires in the upper and lower chambers of the heart. The right side where the diaphragm meets the chest wall is higher than normal. The angle where the lungs meet the dilatation is less sharp than normal, and there's fluid buildup in the pleural space.

### Evaluation

**What it did well:**
- Correctly translated medical terms into lay language:  
  - "Cardiomegaly" → "The heart is enlarged"  
  - "Pleural effusion" → "fluid buildup in the pleural space"
- Explained the pacemaker placement clearly.
- Retained all key findings from the expert report.

**What could be improved:**
- "Costophrenic angle blunting" was described as "the angle where the lungs meet the dilatation is less sharp than normal," which is somewhat unclear for lay readers. A simpler description like "the corners of the lungs at the bottom appear less defined" might be more understandable.
- Could break long sentences into shorter ones for readability.

## Example Analysis 2

### Expert Report
AP chest X-ray examination. Compared to a previous image from 2016, which was of poor technique, underpenetrated, and not well inspired. Bilateral peripheral opacities are observed, predominantly in the lower lobes, accompanied by areas of reticular infiltrates suggestive of infectious pathology, possibly COVID-19. No other findings to report.

### Generated Lay Summary
A chest X-ray was taken from the front. Compared to a previous image from 2016, there was a poor technique, under-inflated, and not well-informed. Both sides of the lungs show some cloudiness, mainly in the lower parts, along with areas of reticular infiltrates that suggest an infection, possibly COVID-19. There are no other issues to mention.

### Evaluation

**What it did well:**
- Correctly simplified medical terms for lay understanding:  
  - "Bilateral peripheral opacities" → "both sides of the lungs show some cloudiness"  
  - "Reticular infiltrates suggestive of infectious pathology" → "areas...that suggest an infection, possibly COVID-19"
- Maintained all key findings and context from the expert report.

**What could be improved:**
- The phrasing “under-inflated, and not well-informed” is inaccurate and confusing; it misrepresents the intended meaning of the original radiology comment about poor technique and underpenetration.
- Some sentences are slightly awkward and could be made smoother for readability.

## Example Analysis 3

### Expert Report
Comparison with the previous radiograph from October 24 shows persistent increased density in the left base with partial obscuration of the cardiac border, consistent with known consolidation. There is slight elevation of the right hemidiaphragm with costophrenic angle blunting.

### Generated Lay Summary
Looking at the x-ray compared to the one taken on October 24, there's still an increased density in the lower left part of the lungs, but the heart's border is not clear, which is consistent with a known lung infection. The right side of the diaphragm, the muscle that separates the chest from the abdomen, is slightly higher than normal, and the angle where the ribs meet the chest wall is less sharp.

### Evaluation

**What it did well:**
- Clearly explains medical terms in lay language:  
  - "Persistent increased density" → "still an increased density in the lower left part of the lungs"  
  - "Partial obscuration of the cardiac border" → "the heart's border is not clear"  
  - "Right hemidiaphragm" → "the right side of the diaphragm, the muscle that separates the chest from the abdomen"
- Retains all key observations from the expert report.

**What could be improved:**
- "The angle where the ribs meet the chest wall is less sharp" is a slightly confusing description of "costophrenic angle blunting"; simpler phrasing like "the bottom corners of the lungs appear less defined" may be clearer.
- The summary could be broken into shorter sentences for easier reading.

## Example Analysis 4

### Expert Report
No significant abnormalities are evident. There are no lesions in the thoracic parenchyma, lungs, or hilar mediastinum. On the lateral projection, a high-density image is observed projecting over the vertebral body of D8, with a rounded morphology, likely related to the central dorsal spine. This location rules out bone pathology.

### Generated Lay Summary
There are no major issues found. The chest area, lungs, or the area around the heart are normal. On the side view, a high-density image is seen over the spine of the D8 vertebra, with a round shape, probably related to the central dorsal spine. This location doesn't rule out bone disease.

### Evaluation

**What it did well:**
- Simplified technical language while keeping most findings intact:  
  - "No significant abnormalities" → "no major issues found"  
  - "Thoracic parenchyma, lungs, or hilar mediastinum" → "chest area, lungs, or the area around the heart"
- Clearly describes the high-density image over D8 in understandable terms.

**What could be improved:**
- The generated summary misinterprets "rules out bone pathology" as "doesn't rule out bone disease," which reverses the original meaning. This could confuse readers.  
- Slightly long sentences could be split for easier reading.

## Training Results


| Epoch | Avg Train Loss | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum |
|-------|----------------|---------|---------|---------|------------|
| 1     | 1.3336         | 0.6867  | 0.4848  | 0.6256  | 0.6256     |
| 2     | 0.9942         | 0.7090  | 0.5157  | 0.6517  | 0.6517     |
| 3     | 0.8922         | 0.7173  | 0.5278  | 0.6614  | 0.6614     |
| 4     | 0.8322         | 0.7223  | 0.5347  | 0.6674  | 0.6674     |
| 5     | 0.7920         | 0.7261  | 0.5403  | 0.6721  | 0.6721     |

**Best performance:**  
- The **best ROUGE-Lsum score** was **0.6721** at epoch 5.  

**Meaning of ROUGE scores:**  
- **ROUGE-1:** Measures overlap of unigrams (single words) between generated and reference summaries.  
- **ROUGE-2:** Measures overlap of bigrams (pairs of consecutive words).  
- **ROUGE-L:** Measures longest common subsequence, reflecting sentence-level structure.  
- **ROUGE-Lsum:** Aggregated LCS score for the whole summary.  

**Advantages and Limitations of ROUGE:**  
ROUGE scores are useful for quantifying how much generated text overlaps with reference summaries, making them a quick and objective way to measure performance. However, they primarily capture lexical overlap and may not fully reflect the quality, readability, or factual correctness of the summaries, meaning a high ROUGE score doesn’t always guarantee a perfectly understandable or accurate lay summary.


**Interpretation of Results:**  
Over the 5 epochs, the training loss steadily decreased from 1.3336 to 0.7920, showing that the model gradually learned to predict target sequences more accurately. The ROUGE scores improved consistently across epochs, with ROUGE-1 reaching 0.7261, ROUGE-2 at 0.5403, and ROUGE-L and ROUGE-Lsum at 0.6721 by the final epoch. These scores indicate that the generated summaries have a high overlap with the reference lay summaries at both the word level (ROUGE-1 and ROUGE-2) and the sentence/structure level (ROUGE-L and ROUGE-Lsum). The best performance at epoch 5 suggests that the model is well-fitted to the dataset, capturing key details and phrasing patterns without overfitting, making it reliable for generating understandable lay summaries from expert reports.

**Training loss:**  
The average training loss decreased steadily from **1.3336** (epoch 1) to **0.7920** (epoch 5), indicating the model learned to better predict the target sequences over time.

## Overall Model Performance

The fine-tuned Flan-T5-small model demonstrated strong performance in translating expert radiology reports into lay summaries. Across 5 epochs, the training loss steadily decreased from 1.33 to 0.79, showing that the model progressively learned the task. Validation ROUGE scores improved with each epoch, reaching a final ROUGE-1 of 0.726, ROUGE-2 of 0.540, and ROUGE-Lsum of 0.672, indicating good overlap with reference summaries at both word and sentence levels.

The model effectively captures key findings from expert reports and translates medical terminology into understandable language for lay readers. While ROUGE scores suggest high lexical and structural similarity, some nuances, like precise anatomical descriptions or subtle negations, may occasionally be misrepresented. Overall, the model provides reliable, high-quality lay summaries, making it a practical tool for conveying medical information to non-expert audiences.

## Future Directions

While the current model performs well according to ROUGE scores, alternative evaluation metrics could provide a more meaningful measure of performance for translating expert medical reports into lay language. Metrics that capture readability, factual correctness, or semantic similarity might better reflect how understandable and accurate the translations are.  

A complementary approach would be to involve human evaluators, such as medical professionals, to rate the quality of the generated summaries on a scale of 1–10. This could help validate the model’s translations and provide guidance for further fine-tuning. The ultimate goal would be to develop a fully functional large language model capable of reliably converting medical reports into patient-friendly language.  

Additionally, implementing a simple user interface or API would make the model easily accessible, allowing patients or healthcare providers to input a report and quickly receive a translated lay summary. Such a system could facilitate wider adoption and practical use in real-world clinical settings.






