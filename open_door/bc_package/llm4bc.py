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
## TODO begin copy my GOOGLE_API_KEY
## GOOGLE_API_KEY = ""
## TODO end
img_path = './data/train_data/3.png'
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


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True)).data

genai.configure(api_key=GOOGLE_API_KEY)
    
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
     
model = genai.GenerativeModel('gemini-1.5-flash')

# ## input: text
# response = model.generate_content("What is the meaning of life?")
# print(response.text)
# print(response.prompt_feedback)
# print(response.candidates)
# print(to_markdown(response.text))

# ## input: text; output in chunks
# response = model.generate_content("What is the meaning of life?", stream=True)
# for chunk in response:
#   print(chunk.text)
#   print("_"*80)

# ## input: image
img = Image.open(img_path)
# response = model.generate_content(img)
# print(to_markdown(response.text))

## input: image and text
response = model.generate_content([question, img], stream=True)
response.resolve()
print(to_markdown(response.text))

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