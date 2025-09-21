## Problem Framing & Solution Formulation

### Problem Analysis
The assignment describes a highly constrained generator:
- Always 5 characters
- Same font/spacing
- Largely fixed colors/texture  
- No skew
- Characters drawn from A–Z, 0–9

This turns generic OCR into a fixed-layout recognition problem where simple image cues suffice.

### Solution Approach
Treat each figure as five independent slots laid out horizontally. The task becomes:
**binarize → segment into 5 components → classify each component → concatenate**

### Method

1. **Binarize**: Estimate background color from image borders and threshold by distance to background using Otsu's method
2. **Segment**: Use connected components analysis to find character regions, merge/split to get exactly 5 components
3. **Normalize**: Crop each character, pad to square, resize to 16×16
4. **Classify**: Build character templates from training data, use cosine similarity for recognition
5. **Assemble**: Concatenate the five predictions to form output string

### Why This Design
- Leverages the brief's invariants
- Avoids heavy OCR/ML while remaining explainable and fast
- Template matching handles consistent cropping/kerning
- Matches deliverable requirement of simple algorithm with `Captcha.__call__(im_path, save_path)` interface

## How to Test

### Test single image
```bash
python captcha_solver.py infer --image sampleCaptchas/input/input21.jpg --model captcha_model.npz --out prediction.txt
```

### Test all images
```bash
python test_all_images.py
```
