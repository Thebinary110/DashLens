#  VQA Autopilot — CVPR Workshop Submission

> Dashcam incident video classification across **25 structured questions** for **661 videos**  
> Using fine-tuned InternVideo2.5 + YOLO + Qwen2.5-VL multimodal pipeline

---

##  Repository Structure

```
VQA-Autopilot/
├── Fine_Tune_E.ipynb               # Fine-tuning InternVideo2.5 on question set E
├── Fine_Tune_F.ipynb               # Fine-tuning InternVideo2.5 on question sets F & G
├── INTERN_VID.ipynb                # InternVideo2.5 base inference & generation
├── Yolo+qwen.ipynb                 # YOLO + Qwen2.5-VL video inference pipeline
├── internvideo_inference.ipynb     #  Final pipeline — full 661-video inference
├── llava-vqa.ipynb                 # LLaVA-NeXT-Video-7B baseline inference
├── frame-sampling.ipynb            # Frame sampling strategy analysis
├── heatmaps-analysis.ipynb         # Attention map / heatmap peak-frame analysis
├── convert_legend.py               # Encodes question labels to numeric values
├── label_map.json                  # Label-to-index mapping
├── legend.txt                      # Raw label legend
├── sample_submission.csv           # Competition submission template
├── 0.62080.csv                     # Submission — score 0.62080
└── 0.64689.csv                     # Submission — score 0.64689
```

---

##  Final Inference Pipeline

**`internvideo_inference.ipynb`** is the primary end-to-end pipeline.

**Inputs:**
- Raw dashcam video dataset (661 videos)
- `sample_submission.csv` — defines the question structure
- LoRA adapters from `Fine_Tune_E.ipynb` (question group E)
- LoRA adapters from `Fine_Tune_F.ipynb` (question groups F & G)

**Output:**
- A fully populated submission CSV with answers to all 25 questions across all 661 videos

The pipeline loads the appropriate LoRA adapter per question group and runs InternVideo2.5 inference for each video.

---

##  Notebook Descriptions

### `Fine_Tune_E.ipynb`
Fine-tunes **InternVideo2.5** on the **E-labeled** question subset using the annotated training data. Produces a LoRA adapter checkpoint used in the final inference pipeline.

### `Fine_Tune_F.ipynb`
Fine-tunes **InternVideo2.5** on the **F and G-labeled** question subsets. Produces a separate LoRA adapter checkpoint for those question groups.

### `INTERN_VID.ipynb`
Base InternVideo2.5 inference notebook — used for generation experiments and sanity-checking model outputs before integration into the fine-tuned pipeline.

### `Yolo+qwen.ipynb`
An alternative inference approach that combines:
- **YOLOv8** for object detection on video frames
- **Qwen2.5-VL** for visual question answering

Runs the full answer generation pipeline over the video dataset.

### `llava-vqa.ipynb`
Baseline inference using **LLaVA-NeXT-Video-7B**. Generates answer CSVs for comparison and benchmarking against the InternVideo2.5-based pipeline.

### `frame-sampling.ipynb`
Analyzes different frame sampling strategies (uniform, scene-change-based, etc.) to determine optimal frame selection for downstream inference quality.

### `heatmaps-analysis.ipynb`
Processes the **attention maps / heatmaps** provided in the dataset to identify peak activation frames. Used to guide more informed frame sampling for inference.

### `convert_legend.py`
Converts human-readable question labels (e.g., `E1`, `F2`, `G3`) into their corresponding **numeric encodings** required by the submission format.

```bash
python convert_legend.py
```



##  Models Used

| Model | Usage |
|---|---|
| InternVideo2.5 | Primary VQA model (fine-tuned with LoRA) |
| Qwen2.5-VL-7B-Instruct | YOLO+Qwen inference pipeline |
| LLaVA-NeXT-Video-7B | Baseline comparison |
| YOLOv8 | Object detection for frame context |

---

##  Quick Start

1. **Fine-tune** the model on your labeled splits:
   ```
   Run Fine_Tune_E.ipynb   → saves LoRA adapter for group E
   Run Fine_Tune_F.ipynb   → saves LoRA adapter for groups F & G
   ```

2. **Run the full inference pipeline:**
   ```
   Run internvideo_inference.ipynb
   ```
   Ensure paths to the video dataset, `sample_submission.csv`, and both LoRA adapter directories are correctly set in the notebook config.

3. **Submit** the generated CSV to the competition leaderboard.

---

##  Competition

**AUTOPILOT VQA — CVPR Workshop**  
25 questions × 661 dashcam incident videos  
Evaluation metric: Accuracy across structured MCQ answer labels
