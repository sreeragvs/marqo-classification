import torch
import open_clip.factory  # Monkey patch target
import gradio as gr
from transformers import AutoModel, AutoProcessor
import requests
from PIL import Image
from io import BytesIO

# ----- Monkey patch to handle NumPy 2.x compatibility -----
def _patched_set_model_device_and_precision(model, device, precision, is_timm_model=False):
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        if is_timm_model:
            from open_clip.transformer import LayerNormFp32
            model = model.to_empty(device=device, dtype=dtype)

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)

            model.apply(_convert_ln)
        else:
            model = model.to_empty(device=device)
            open_clip.factory.convert_weights_to_lp(model, dtype=dtype)

    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model = model.to_empty(device=device, dtype=dtype)

    else:
        model = model.to_empty(device=device)

    return model

# Apply the patch
open_clip.factory._set_model_device_and_precision = _patched_set_model_device_and_precision

# ----- Load model and processor -----
fashion_items = ['top', 'trousers', 'jumper', 'briefs']

model_name = 'Marqo/marqo-fashionSigLIP'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# ----- Preprocess text -----
with torch.no_grad():
    processed_texts = processor(
        text=fashion_items,
        return_tensors="pt",
        truncation=True,
        padding=True
    )['input_ids']
    text_features = model.get_text_features(processed_texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ----- Prediction function -----
def predict_from_url(url):
    if not url:
        return {"Error": "Please input a URL"}

    try:
        image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
    except Exception as e:
        return {"Error": f"Failed to load image: {str(e)}"}

    processed_image = processor(images=image, return_tensors="pt")['pixel_values']

    with torch.no_grad():
        image_features = model.get_image_features(processed_image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_probs = (100 * image_features @ text_features.T).softmax(dim=-1)

    return {fashion_items[i]: round(float(text_probs[0, i]), 4) for i in range(len(fashion_items))}

# ----- Gradio UI -----
demo = gr.Interface(
    fn=predict_from_url,
    inputs=gr.Textbox(label="Enter Image URL"),
    outputs=gr.Label(label="Classification Results"),
    title="Fashion Item Classifier",
    allow_flagging="never"
)

# ----- Run -----
if __name__ == "__main__":
    demo.launch()
