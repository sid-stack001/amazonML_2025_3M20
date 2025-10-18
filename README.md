# ML Challenge 2025: Smart Product Pricing Solution

**Team Name: 3M20** \
**Team Members: Siddharth Verma, JVS Chandradithya, A Dheerak Kumar** \
**Submission Date: October 13, 2025**

This project implements a **multimodal price prediction model** combining:

- **Text features** from product descriptions using **BERT**
- **Image features** using **Vision Transformer (ViT)**
- **Numeric features** (extracted value and unit)

The goal is to predict product prices accurately using all available modalities.

---

## Model Architecture

1. **Text Encoder (BERT)**

   - Pretrained BERT (`bert-base-uncased`) processes catalog content.
   - Pooler output → Linear → ReLU → Dropout → Text feature vector.

2. **Image Encoder (ViT)**

   - Pretrained ViT model (`vit_base_patch16_224` or `vit_base_patch16_clip_224.laion2b_ft_in1k`).
   - Extracts image features → Linear → ReLU → Dropout → Image feature vector.

3. **Numeric Features**

   - Extract `Value:` and `Unit:` from catalog text.
   - Scale numeric value using `StandardScaler`.
   - Encode unit as one-hot vector.

4. **Feature Fusion & Regression**
   - Concatenate `[text_features, image_features, numeric_features]`.
   - Feedforward network: Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Softplus.
   - Output: predicted price (enforced positive using Softplus).

---

## Training Process

- **Dataset**: CSV with `catalog_content`, `price`, and `image` info.
- **Loss Function**: SMAPE (Symmetric Mean Absolute Percentage Error) [modified]
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup schedule
- **Batch Size**: Configurable (default 16)
- **Epochs**: Configurable (default 6)
- **Mixed Precision**: Automatic AMP if GPU available
- **Train/Validation Split**: 90% train, 10% validation
- **Checkpointing**: Best validation SMAPE model saved

---

## Working

1. **Preprocessing**

   - Extract numeric value and unit from text.
   - Scale numeric features and one-hot encode units.
   - Resize and normalize images.
   - Tokenize text with BERT tokenizer.

2. **Training**

   - Each batch: forward pass → compute SMAPE loss → backward pass → optimizer step.
   - Validation at each epoch to track SMAPE.

3. **Inference**
   - Load best checkpoint.
   - Prepare test data similar to training (text, numeric, images).
   - Predict prices using the trained model.
   - Outputs are clipped to positive values and saved as CSV with `[sample_id, price]`.

---

## Notes

- GitHub link - https://github.com/sid-stack001/amazonML_2025_3M20
- Missing images are replaced with blank white images.
- Model combines **text, image, and numeric modalities** for robust prediction.
- SMAPE ensures relative error is penalized appropriately, useful for price prediction.
