import torch
import clip
from PIL import Image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def CLIP(img_path, text_prompt, model_name = "ViT-B/32", if_p = True):
    ## Load CLIP model
    clip_model, preprocess = clip.load(model_name, device=DEVICE)
    ## get img_input
    img_input = preprocess(Image.open(img_path)).unsqueeze(0).to(DEVICE)
    ## get text_input
    text_input = clip.tokenize(text_prompt).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(img_input)
        text_features = clip_model.encode_text(text_input)
        logits_per_image, logits_per_text = clip_model(img_input, text_input)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        if if_p:
            print("Label probs:", probs)  # prints: [[0.9927937  0.00421068]]
    return probs

def reward_annotation_CLIP(img_path = "CLIP.png", if_p = True):
    text_prompt = ["door that is closed", "door that is open"]
    ## using CLIP
    probs = CLIP(img_path, text_prompt)
    ## get reward
    reward = 1 if probs[0][1] > probs[0][0] else 0 # Reward is 1 if closer to 'open door' prompt, 0 otherwise
    if if_p:
        print("reward:", reward)  # prints: 1/0
        print(text_prompt[reward]) # door is open/closed
    return reward

if __name__ == "__main__":
    # img_path='closed-door.png'
    img_path='door1.jpg'
    reward_annotation_CLIP(img_path)
    img_path='door2.jpg'
    reward_annotation_CLIP(img_path)
    img_path='door3.jpg'
    reward_annotation_CLIP(img_path)
    img_path='door4.jpg'
    reward_annotation_CLIP(img_path)
    img_path='door5.jpg'
    reward_annotation_CLIP(img_path)