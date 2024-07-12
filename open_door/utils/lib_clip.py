import clip
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def CLIP(rgb_img, text_prompt, model_name = "ViT-B/32", if_p = False):
    clip_model, preprocess = clip.load(model_name, device=DEVICE) # Load CLIP model
    if isinstance(rgb_img,str):
        rgb_img_input = preprocess(Image.open(rgb_img)).unsqueeze(0).to(DEVICE)# get rgb_img_input
    else:
        rgb_img_input = preprocess(rgb_img).unsqueeze(0).to(DEVICE)# get rgb_img_input
    text_input = clip.tokenize(text_prompt).to(DEVICE)# get text_input
    with torch.no_grad():
        image_features = clip_model.encode_image(rgb_img_input)
        text_features = clip_model.encode_text(text_input)
        logits_per_image, logits_per_text = clip_model(rgb_img_input, text_input)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        if if_p:
            print("Label probs:", probs)  # prints: [[0.9927937  0.00421068]]
    return probs

def CLIP_detection(rgb_img=None,text_prompt=["door that is closed", "door that is open"],if_p=False):
    probs = CLIP(rgb_img, text_prompt) # using CLIP
    result = 1 if probs[0][1] > probs[0][0] else 0 # Reward is 1 if closer to 'open door' prompt, 0 otherwise
    if if_p:
        # print("result:", result)  # prints: 1/0
        print(f'[CLIP INFO] Result: {text_prompt[result]},\tLabel probs: {probs}') # door is open/closed
    return result