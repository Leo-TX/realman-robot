'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-06-01 19:28:03
Version: v1
File: 
Brief: 
'''
import textwrap
from PIL import Image
import google.generativeai as genai
from IPython.display import display, Markdown
from utils.lib_io import *

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True)).data
  
class GEMINI(object):
    def __init__(self,google_api_key,model_name):
        self.google_api_key = google_api_key
        self.model_name = model_name

        self.create_model()
    
    @classmethod
    def init_from_yaml(cls,cfg_path='cfg/cfg_gemini.yaml'):
        cfg = read_yaml_file(cfg_path, is_convert_dict_to_class=True)
        return cls(cfg.google_api_key,cfg.model_name)

    def create_model(self):
        genai.configure(api_key=self.google_api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def get_all_models(self):
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)

    def text_to_text(self, prompt="What is the meaning of life?",if_p=False):
        response = self.model.generate_content(prompt)
        if if_p:
            print(response.text)
            # print(response.prompt_feedback)
            # print(response.candidates)
            # print(to_markdown(response.text))
        return response.text
    
    def img_to_text(self, img_path,if_p=False):
        img = Image.open(img_path)
        response = model.generate_content(img)
        if if_p:
            print(response.text)
        return response.text

    def text_img_to_text(self, prompt="What is the meaning of life?", img_path=None,if_p=False):
        response = model.generate_content([question, img], stream=True)
        response.resolve()
        if if_p:
            print(response.text)
        return response.text
    
# ## input: text; output in chunks
# response = model.generate_content("What is the meaning of life?", stream=True)
# for chunk in response:
#   print(chunk.text)
#   print("_"*80)

question = """
There are some different types of handles and door. The task is to open the door with the handle, which including two steps.
The first step is to generate the primitives sequence. The primitives sequence is a list of primitives, which can be used to open the door, including: [0:None,1:Grasp,2:Unlock,3:Rotate,4:Open].
The second step is to generate the parameters of the primitives. The action space and the detailed description of the five primitives are as follows:
The action space of the robot is: (vx,vy,vz,vyaw,vpitch,vroll,Vx,Vy,Vw), the first 6-dimention is about the 6-DOF arm movement, the last 3-dimention is about the 3-DOF base movement.
The primitive is a API function. The input is the parameters of the primitive, the output is the action of the primitive.
None: Do nothing.
Grasp: Grasp the handle. The parameter of this primitive is grasp offset(3-dimention), which means the offset value(x,y,z) to the center location of the hanle. For every dimention: (-d,d), set d=2.5cm
Unlock: Unlock the handle. For some kinds of handles(like lever handle), it will need the unlocking. The parameter of this primitive is execution time(1-dimention), which means when to stop the unlocking, (-T,T), set T=2.5s, the sign of the time means clockwise or counter-clockwise. The action is (0,0,vz,vyaw,0,0,0,0,0), vz = 10cm/s, vyaw=25°/s
Rotate: Rotate the handle. For some kinds of handles(like doorknob), it will need the rotating. The parameter of this primitive is execution time(1-dimention), which means when to stop the rotating, (-T,T), set T=2.5s, the sign of the time means clockwise or counter-clockwise. The action is (0,0,0,vyaw,0,0,0,0,0), vyaw=25°/s
Open: Open the door. The parameter of this primitive is execution time(1-dimention), which means when to stop the opening, (-T,T), set T=2.5s, the sign of the time means forward or backward. The action is (0,0,0,0,0,0,Vx,0,0), set Vx=20cm/s

So now I will give you a picture of the door and the handle. You task is to generate the primitives sequence and the parameters of the primitives.
Example1:
"handle_type": "lever_handle",
"sequence": [1,2,4],
"parameters": [(0.1,-0.2,0.3),(1.1),(1.2)]
Example2:
"handle_type": "doorknob_handle",
"sequence": [1,3,4],
"parameters": [(0.1,-0.2,0.3),(1.3),(1.2)]
Example3:
"handle_type": "crossbar_handle",
"sequence": [1,4],
"parameters": [(0.1,-0.2,0.3),(1.2)]
Example4:
"handle_type": "touchbar_handle",
"sequence": [1,4],
"parameters": [(0.1,-0.2,0.3),(1.2)] 
"""

# ## input: text; output in chunks
# response = model.generate_content("What is the meaning of life?", stream=True)
# for chunk in response:
#   print(chunk.text)
#   print("_"*80)

# ## chat conversations
# chat = model.start_chat(history=[])
# response = chat.send_message("In one sentence, explain how a computer works to a young child.")
# print(to_markdown(response.text))
# print(chat.history)

# response = chat.send_message("Okay, how about a more detailed explanation to a high schooler?", stream=True)
# for chunk in response:
#   print(chunk.text)
#   print("_"*80)

# for message in chat.history:
#   print(to_markdown(f'**{message.role}**: {message.parts[0].text}'))

# ## count tokens
# model.count_tokens("What is the meaning of life?")
# model.count_tokens(chat.history)

if __name__ == "__main__":
    gemini = GEMINI.init_from_yaml('cfg/cfg_gemini.yaml')
    response = gemini.text_to_text(prompt="What is the meaning of life?",if_p=True)