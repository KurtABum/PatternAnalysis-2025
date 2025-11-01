# BioLaySumm: Radiology Report to Layman Summary

## Overview

BioLaySumm is a transformer based natural language processing pipeline designed to convert complex radiology reports into easily understandable lay summaries. The goal is to make medical findings more accessible to patients and non-experts, improving patient comprehension and engagement.


## Dataset Description

The project uses the BioLaySumm2025-LaymanRRG-opensource-track dataset, which is specifically designed for radiology report simplification.  
It is already split into:
- **Training:** 150,000 examples  
- **Validation:** 10,000 examples  
- **Testing:** 10,500 examples  

Each record contains four columns:
- **source** – metadata about data origin (not used in training)  
- **image_path** – reference to the related image (not used in training)  
- **radiology_report** – the full expert-written radiology report  
- **layman_report** – the corresponding simplified lay summary  

During training, only the `radiology_report` (input) and `layman_report` (target) columns are used.  
The `radiology_report` serves as the expert-level medical input, while the `layman_report` represents the desired human-understandable output.  
This pairing allows the model to learn how to translate technical medical language and findings into clear, accessible language.

## Setup and Usage

### 1. Install Visual Studio Code

Visual Studio Code is recommended for editing, running, and debugging the project.  
 

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

### 5. Run Training Loop
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

- **train.py** – Handles training the model. Loads the dataset, initialises the model, and runs the training loop with logging.  
- **predict.py** – Runs the trained model to show example predictions. Displays a few inputs along with their translated outputs.  
- **modules.py** – Contains reusable functions, helper routines, and model components that are shared across scripts.  
- **dataset.py** – Manages data loading and preprocessing. Converts raw data into batches suitable for training or evaluation.

## Data Cleaning and Preprocessing

The dataset is prepared in `dataset.py` before training:

- Handles empty or missing summaries by inserting a placeholder token.  
- Tokenises and encodes both expert (radiology) and layman reports into tensors for model input.  
- Pads and truncates text sequences to fixed lengths for consistency.  
- Batches data efficiently during training and evaluation using PyTorch’s `DataLoader`.

## Flan-T5 Model

Flan-T5 is a large language model developed by Google, built on an encoder-decoder transformer architecture. It is designed for instruction-following and text-to-text tasks such as translation, summarisation, and explanation generation.

The encoder processes and represents the input (expert radiology report), while the decoder generates a corresponding layman-friendly summary. Both components use multi-layer transformers with self and cross attention to capture complex dependencies.

In this project, Flan-T5 is fine-tuned on paired expert–layman reports, allowing it to learn how to translate professional radiology language into accessible summaries. This provides high-quality, domain-adapted outputs, though training large versions can be computationally demanding and memory intensive.


## Fine-Tuning Strategy

For this project, a Flan-T5-small model was fine-tuned for translating or transforming input sequences. The model configuration included:

- **Layers & hidden size:** The Flan-T5-small base model has 6 transformer layers and a hidden dimension of 512 units. This setup provides sufficient capacity to model complex relationships in the input data while keeping training manageable.

- **Classifier / output head:** The standard T5 decoder head is used to map predictions to the target token space, ensuring the model generates outputs that align with the task-specific labels. No custom classification layer was added.

- **Loss function:** The model was trained using cross-entropy loss, which measures how well the predicted tokens match the target tokens. This loss is effective for sequence-to-sequence tasks because it encourages the model to assign high probabilities to the correct outputs.

- **Fine-tuning strategy:** All model parameters were fine-tuned directly rather than using parameter-efficient methods such as adapters or LoRA. This allows the model to fully adapt to the dataset, though it requires more memory and training time.

- **Epochs:** The model was trained for 5 epochs, which balances learning task-specific patterns while minimising the risk of overfitting on the dataset.

- **Optimiser & learning rate:** AdamW was used with a learning rate of 5e-6, providing stable updates suitable for fine-tuning large pretrained models.

- **Gradient:** During training, gradients of the loss with respect to the model parameters are computed automatically via `loss.backward()`. To maintain stable training and prevent exploding gradients, the model applies gradient clipping using `torch.nn.utils.clip_grad_norm_`.

This configuration was chosen to allow efficient adaptation of Flan-T5-small to the task while keeping training time and GPU memory requirements reasonable.


### Overfitting and Regularisation

Overfitting occurs when the model memorises training data rather than learning generalisable patterns, leading to poor performance on unseen examples.  

To mitigate overfitting in BioLaySumm:

- **Validation Monitoring:** The model is evaluated on a held-out validation set after each epoch. Only improvements in validation ROUGE-Lsum trigger checkpoint saving.  
- **Gradient Clipping:** Extremely large gradient updates are prevented to stabilise training.  
- **Input Handling:** While no traditional data augmentation is applied, the preprocessing stage includes truncation, padding, and placeholder replacements for missing summaries. These steps ensure consistent input lengths and help the model handle varied inputs.  

## Model Saving

During training, the model is saved to a directory (default: `best_model`) whenever the current epoch achieves a higher validation ROUGE-Lsum score than all previous epochs. This ensures that only the best-performing version of the model is kept.  

The following files are typically saved:  

- **`config.json`** – Defines the model architecture and configuration parameters.  
- **`generation_config.json`** – May be automatically created; contains text generation settings such as maximum length or beam size.  
- **`pytorch_model.bin`** or **`model.safetensors`** – Stores the trained model weights.  
- **`special_tokens_map.json`** – Maps special tokens   
- **`tokenizer_config.json` and `tokenizer.json`** – Contain the vocabulary and tokenisation rules needed to convert text to model inputs and back.  

In `predict.py`, you can modify the model path to load a specific saved version for testing or demonstration. For example:

```python
wrapper = FlanT5Summarizer(model_name="path/to/best_model", device=DEVICE)
```

## Training and Testing

Training and evaluation for this project were performed on a Windows PC equipped with an RTX 5070 GPU with 12GB of VRAM. Due to GPU memory limitations, the batch size was set to 2, and full length inputs and outputs were used to ensure accurate training and evaluation.

- **Training:** The Flan-T5-small model was trained for 5 epochs using full fine-tuning of all model parameters. Training the full dataset on this hardware took approximately 36 hours.


- **Validation & Testing:** After each epoch, the model was evaluated on a validation set to monitor performance and select the best model based on ROUGE scores. Final evaluation was conducted on a held out test set to report the model’s predictive performance.

## Example Analysis 1

### Expert Report
Cardiomegaly. Left subclavian pacemaker with atrial and ventricular leads. Elevation of the right hemidiaphragm. Costophrenic angle blunting with pleural effusion.

### Generated Lay Summary
The heart is enlarged. There's a pacemaker on the left side of the chest with wires in the upper and lower chambers of the heart. The right side where the diaphragm meets the chest wall is higher than normal. The angle where the lungs meet the dilatation is less sharp than normal, and there's fluid buildup in the pleural space.

### Evaluation

**What it did well:**
- Correctly translated medical terms into lay language:  
  - "Cardiomegaly" to "The heart is enlarged"  
  - "Pleural effusion" to "fluid buildup in the pleural space"
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
  - "Bilateral peripheral opacities" to "both sides of the lungs show some cloudiness"  
  - "Reticular infiltrates suggestive of infectious pathology" to "areas...that suggest an infection, possibly COVID-19"
- Maintained all key findings and context from the expert report.

**What could be improved:**
- The phrasing “under-inflated, and not well-informed” is inaccurate and confusing as it misrepresents the intended meaning of the original radiology comment about poor technique and underpenetration.
- Some sentences are slightly awkward and could be made smoother for readability.

## Example Analysis 3

### Expert Report
Comparison with the previous radiograph from October 24 shows persistent increased density in the left base with partial obscuration of the cardiac border, consistent with known consolidation. There is slight elevation of the right hemidiaphragm with costophrenic angle blunting.

### Generated Lay Summary
Looking at the x-ray compared to the one taken on October 24, there's still an increased density in the lower left part of the lungs, but the heart's border is not clear, which is consistent with a known lung infection. The right side of the diaphragm, the muscle that separates the chest from the abdomen, is slightly higher than normal, and the angle where the ribs meet the chest wall is less sharp.

### Evaluation

**What it did well:**
- Clearly explains medical terms in lay language:  
  - "Persistent increased density" to "still an increased density in the lower left part of the lungs"  
  - "Partial obscuration of the cardiac border" to "the heart's border is not clear"  
  - "Right hemidiaphragm" to "the right side of the diaphragm, the muscle that separates the chest from the abdomen"
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
  - "No significant abnormalities" to "no major issues found"  
  - "Thoracic parenchyma, lungs, or hilar mediastinum" to "chest area, lungs, or the area around the heart"
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
- The best final testing ROUGE-Lsum score was 0.6721 at epoch 5.  

Meaning of ROUGE scores:
- **ROUGE-1:** Measures overlap of unigrams between generated and reference summaries.  
- **ROUGE-2:** Measures overlap of bigrams.  
- **ROUGE-L:** Measures longest common subsequence, reflecting sentence-level structure.  
- **ROUGE-Lsum:** Aggregated LCS score for the whole summary.  

**Advantages and Limitations of ROUGE:**  
ROUGE scores are useful for quantifying how much generated text overlaps with reference summaries, making them a quick and objective way to measure performance. However, they primarily capture lexical overlap and may not fully reflect the quality, readability, or factual correctness of the summaries, meaning a high ROUGE score doesn’t always guarantee a perfectly understandable or accurate lay summary.


**Interpretation of Results:**  
Over the 5 epochs, the ROUGE scores improved consistently across epochs, with ROUGE-1 reaching 0.7261, ROUGE-2 at 0.5403, and ROUGE-L and ROUGE-Lsum at 0.6721 by the final epoch. These scores indicate that the generated summaries have a high overlap with the reference lay summaries at both the word level (ROUGE-1 and ROUGE-2) and the sentence/structure level (ROUGE-L and ROUGE-Lsum). The best performance at epoch 5 suggests that the model is well-fitted to the dataset, capturing key details and phrasing patterns without overfitting, making it reliable for generating understandable lay summaries from expert reports.

**Training loss:**  
The average training loss decreased steadily from 1.3336 at epoch 1 to 0.7920 at epoch 5, indicating the model learned to better predict the target sequences over time.

## Overall Model Performance

The fine-tuned Flan-T5-small model demonstrated strong performance in translating expert radiology reports into lay summaries. Across 5 epochs, the training loss steadily decreased from 1.33 to 0.79, showing that the model progressively learned the task. Validation ROUGE scores improved with each epoch, reaching a final ROUGE-1 of 0.726, ROUGE-2 of 0.540, and ROUGE-Lsum of 0.672, indicating good overlap with reference summaries at both word and sentence levels.

The model effectively captures key findings from expert reports and translates medical terminology into understandable language for lay readers. While ROUGE scores suggest high lexical and structural similarity, some nuances, like precise anatomical descriptions or subtle negations, may occasionally be misrepresented. Overall, the model provides reliable, high-quality lay summaries, making it a practical tool for conveying medical information to non-expert audiences.

## Future Directions

While the current model performs well according to ROUGE scores, alternative evaluation metrics could provide a more meaningful measure of performance for translating expert medical reports into lay language. Metrics that capture readability, factual correctness, or semantic similarity might better reflect how understandable and accurate the translations are.  

A complementary approach would be to involve human evaluators, such as medical professionals, to rate the quality of the generated summaries on a scale of 1–10. This could help validate the model’s translations and provide guidance for further fine-tuning. The ultimate goal would be to develop a fully functional large language model capable of reliably converting medical reports into patient-friendly language.  

Additionally, implementing a simple user interface or API would make the model easily accessible, allowing patients or healthcare providers to input a report and quickly receive a translated lay summary. Such a system could facilitate wider adoption and practical use in real-world clinical settings.
 
To further improve training efficiency, early stopping could be integrated into the training loop. This would allow the model to be trained for a large number of epochs without overfitting, as training would automatically stop once the validation ROUGE scores and loss stop improving for a set number of consecutive epochs. This ensures the model is trained long enough to learn effectively but not unnecessarily beyond the point of performance gain.

## References and AI Usage
- ChatGPT has been used to assist in creating, spell/grammar checking and fact check parts of this report, especially in terms of evaluating layman reports compared to the radiology report
- Gemini was used to assist in crating some parts of the model code, especially with debugging and improving the model
- Some parts of the code were generated by GitHub Copilot's autofill feature




