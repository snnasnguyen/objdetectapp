import streamlit as st
import torch 
import torchvision
import numpy as np
from torchvision import transforms as T

from PIL import Image
import cv2

#Create web interface
max_width_str = f"max-width: 1200px;"
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>Object Detection Tool</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
            * Click the button to upload a image file.
            * Wait for running
"""
)

#Upload image from local
uploaded_file = st.file_uploader("Upload Image", type=[".png", ".jpg", ".jpeg"])

#Process image 
if uploaded_file:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
    model.eval()
    igg = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    igg = cv2.cvtColor(igg , cv2.COLOR_BGR2RGB)
    st.image(igg)
    ig = Image.open(uploaded_file)
    transform = T.ToTensor()
    img = transform(ig)
    with torch.no_grad():
        pred = model([img])
    bboxes, labels, scores = pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']
    num = torch.argwhere(scores > 0.8).shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    coco_names = ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
                "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
                "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
                "plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
                "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" ,
                "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,
                "mirror" , "dining table" , "window" , "desk" , "toilet" , "door" , "tv" ,
                "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
                "oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
                "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"]

    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        class_name = coco_names[labels.numpy()[i] - 1]
        igg = cv2.rectangle(igg, (x1, y1), (x2, y2), (0, 255, 0), 1)
        igg = cv2.putText(igg, class_name, (x1, y1 + 20), font, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    st.image(igg)
