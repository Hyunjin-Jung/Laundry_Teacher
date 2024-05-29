import streamlit as st
from src.camera_input_live import camera_input_live
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.engine.results import Boxes
from argparse import Namespace

# class
class2Fiber = {
    0 : "Cotton",
    1 : "Hemp",
    2 : "cellulose fiber Others",
    3 : "Silk",
    4 : "Wool",
    5 : "protein fiber Others",
    6 : "Viscos rayon",
    7 : "regenerated fiber Others",
    8 : "Polyester",
    9 : "Nylon",
    10 : "Polyurethane",
    11 : "synthetic fiber Others"
}

classToMethod = {
    0 : "hand washing",
    1 : "washing machine",
    2 : "dry cleaning"
}

washing_method = {
    0 : 
    '''
Cotton is a highly absorbent material, and it is easy to deform, so it is recommended to hand wash it alone in cold water. 
Wash it separately using a neutral detergent or a hand-washing detergent. 
Avoid washing machines and dry cleaning( it may damage the fabric.)
    ''',
    1 : 
'''
hand wash in cold water.
Wash it using a neutral detergent
''',
    2 : 
'''
Use neutral detergent
wash by hand in lukewarm water below 30 degrees Celsius (can shrink at high temperatures)
Use a laundry bag when using a washing machine (but not recommended because of fiber damage)
''',
    3 : 
'''
Dry cleaning or hand wash
''',
    4 : 
'''
Use wool detergent with lukewarm water
''',
    5 : 
'''
Use wool detergent with lukewarm water
''',
    6 : 
'''
Dry cleaning or hand wash
''',
    7 : 
'''
Dry cleaning or hand wash
''',
    8 : 
'''
Use neutral detergent hand wash or machine wash in cold water
''',
    9 : 
'''
Use neutral detergent hand wash in cold or lukewarm water, or dry cleaning
''',
    10 :
'''
Use neutral detergent hand wash in cold or lukewarm water, or machine wash
''',
    11 : 
'''
Use neutral detergent hand wash in cold or lukewarm water, or machine wash
''' 
}

precautions = {
    0 : 
'''
Do not use a dryer(it can cause shrinkage.)
Dry it away from direct sunlight
''',
    1 :
'''
Do not twist too much
Dry recommended in the shade
''',
    2 : 
'''
Do not use fabric softener (fiber powder may occur due to the nature of it)
Do not use a dryer
''',
    3 : 
'''
Dry in the shade
''',
    4 : 
'''
First wash should be dry cleaning
Avoid frequent washing
Do not twist; dry flat
''',
    5 : 
'''
First wash should be dry cleaning
Avoid frequent washing
Do not twist; dry flat
''',
    6 : 
'''
Be careful of pilling and stretching
Handle gently to avoid snags
Vulnerable to water
First wash should be dry cleaning
''',
    7 : 
'''
Be careful of pilling and stretching
Handle gently to avoid snags
Vulnerable to water
First wash should be dry cleaning
''',
    8 : 
'''
If 100% polyester, wash separately
''',
    9 : 
'''
Avoid hot water(it is sensitive to heat)
''',
    10 :
'''
Avoid ironing or washing in hot water(it is sensitive to heat)
''',
    11 :
'''
Avoid ironing or washing in hot water(it is sensitive to heat)
'''
}

# iou 계산
# box = [x_min, y_min, x_max, y_max]
def get_iou(box1, box2):
    def _is_box_intersect(box1, box2):
        if box1[2] <= box2[0] or box2[2] <= box1[0]:
            return False
        if box1[3] <= box2[1] or box2[3] <= box1[1]:
            return False
        return True
    def _get_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    def _get_intersection_area(box1, box2):
        w = min(box1[2], box2[2]) - max(box1[0], box2[0])
        h = min(box1[3], box2[3]) - max(box1[1], box2[1])
        return w * h
    def _get_union_area(box1, box2):
        return _get_area(box1) + _get_area(box2) - _get_intersection_area(box1, box2)
    
    if not _is_box_intersect(box1, box2):
        return 0
    
    i_area = _get_intersection_area(box1, box2)
    u_area = _get_union_area(box1, box2)
    iou = i_area / u_area
    
    return iou

# model loading
if 'model' not in st.session_state:
    st.session_state.model = YOLO('./models/fiber_only.pt')
    print("model loading complete!")
if 'model2' not in st.session_state:
    st.session_state.model2 = YOLO("./models/washing_method_only.pt")
    print("model2 loading complete!")
if 'tracker' not in st.session_state:
    args = {
    'track_high_thresh' : 0.01, # threshold for the first association
    'track_low_thresh' : 0.001, # threshold for the second association
    'new_track_thresh' : 0.001, # threshold for init new track if the detection does not match any tracks
    'track_buffer' : 30, # buffer to calculate the time when to remove tracks
    'match_thresh' : 0.8,

    'gmc_method' : 'sparseOptFlow',
    'proximity_thresh' : 0.5,
    'appearance_thresh' : 0.25,
    'with_reid' : False
    }
    st.session_state.tracker = BOTSORT(Namespace(**args))
    print("tracker init complete!")

# page ui
st.title("Laundry Teacher")
st.divider()
mode = st.selectbox(
    "Please choose the desired mode of operation:",
    ("Fabric Information","Simultaneous Wash Compatibility" )
)
st.divider()

# fabric information mode
if mode == "Fabric Information":
    image = camera_input_live(debounce=1000)
    is_detection = False

    if image:
        # prepare image
        img = Image.open(image)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # inference
        results = st.session_state.model.predict(img, conf=0.2)
        data = results[0].boxes.data.clone()   
        orig_shape = results[0].boxes.orig_shape

        # data sort by confidence score
        data = results[0].boxes.data.clone()
        indices = data[:, 4].sort(descending=True).indices
        data = data[indices]

        if len(data) != 0:
            is_detection = True

            # NMS - fiber_only
            new_boxes = []
            fiber_sets = []
            iou_thresh = 0.5
            for cur in data:
                fiber = class2Fiber[int(cur[5])]
                is_new = True
                for idx, box in enumerate(new_boxes):
                    iou = get_iou(cur, box)
                    if iou >= iou_thresh:
                        is_new = False
                        fiber_sets[idx].add(class2Fiber[int(cur[5])])
                        break
                if is_new:
                    new_boxes.append(cur)  
                    fiber_sets.append(set({class2Fiber[int(cur[5])]}))
                    
            if len(new_boxes) == 1:
                new_boxes.append(torch.tensor([ 0, 0, 0, 0, 0, 0]))
            new_boxes = torch.stack(new_boxes) 
            new_boxes = Boxes(new_boxes, orig_shape)   

            # tracking
            track_result = st.session_state.tracker.update(new_boxes)

            for result in track_result:
                cv2.putText(img, "id : " + str(int(result[4])),(int(result[0]), int(result[1])), cv2.FONT_ITALIC, 1, (255, 0, 0), 5)
                cv2.rectangle(img, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (255,0,0), 5)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # visualize result
        st.image(img)
        if is_detection:
            # result [xyxy id label index]
            for result in track_result:

                id = int(result[4])
                fiber = int(result[6])
                height = len(img)
                width = len(img[0])
                roi = img[max(int(result[1]), 0): min(int(result[3]), height-1), max(int(result[0]), 0): min(int(result[2]), width-1)]

                with st.expander( f"id : {id} | "+ f"섬유 조성 : {fiber_sets[int(result[7])]}"):
                    st.image(roi)
                    st.header("Recommended Washing Method", divider='rainbow')
                    st.write(washing_method[fiber])
                    st.header("Precautions", divider='rainbow')
                    st.write(precautions[fiber])
# Simultaneous Wash Compatibility mode
elif mode == "Simultaneous Wash Compatibility":
    image = st.camera_input("Take a picture")

    if image:
        # prepare image
        img = Image.open(image)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # inference
        results = st.session_state.model2.predict(img, conf=0.2)
        data = results[0].boxes.data.clone()   
        orig_shape = results[0].boxes.orig_shape

        # data sort by confidence score
        data = results[0].boxes.data.clone()
        indices = data[:, 4].sort(descending=True).indices
        data = data[indices]

        if len(data) != 0:
            # NMS - fiber_only
            new_boxes = []
            iou_thresh = 0.5
            for cur in data:
                fiber = class2Fiber[int(cur[5])]
                is_new = True
                for idx, box in enumerate(new_boxes):
                    iou = get_iou(cur, box)
                    if iou >= iou_thresh:
                        is_new = False
                        break
                if is_new:
                    new_boxes.append(cur)  
            if len(new_boxes) == 1:
                new_boxes = new_boxes[0].unsqueeze(0)
            else:
                new_boxes = torch.stack(new_boxes) 
            
            # method classification
            methods = []
            for result in new_boxes:
                method = int(result[5])
                if method == 0:
                    methods.append(0)
                elif method >= 5:
                    methods.append(1)
                else:
                    methods.append(2)
            max_method = max(set(methods), key = methods.count)
            st.header(f"method : {classToMethod[max_method]}")
            container = st.container()

            height = len(img)
            width = len(img[0])
            for i in range(3):
                if i == max_method:
                    continue
                with st.expander(f"washing method : {classToMethod[i]}"):
                    for idx, result in enumerate(new_boxes):
                        if methods[idx] == i:
                            roi = img[max(int(result[1]), 0): min(int(result[3]), height-1), max(int(result[0]), 0): min(int(result[2]), width-1)]
                            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            st.image(roi)

            # result [xyxy id label index]
            for idx, result in enumerate(new_boxes):
                if methods[idx] == max_method:
                    cv2.rectangle(img, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (255,0,0), 5)
                else:
                    cv2.rectangle(img, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0,0,255), 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            container.image(img)

        else:
            st.write("no detection...")
