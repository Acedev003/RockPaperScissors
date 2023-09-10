import time

import cv2
import pandas as pd
import mediapipe as mp
import PySimpleGUI as gui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FRAME_WIDTH  = 224
FRAME_HEIGHT = 224

MODEL_PATH       = "models/hand_landmarker.task"        # Download the file from https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models 
DATA_SAVE_FOLDER = "data" 

CLASS_ROCK     = 0
CLASS_PAPER    = 1
CLASS_SCISSORS = 2

GUI_THEME  = "DarkAmber"

def main():
    global result_points
    result_points = []
    
    rock_count     = 0
    paper_count    = 0
    scissors_count = 0
    
    data = {
                "WRIST_X": [],
                "WRIST_Y": [],
                "THUMB_CMC_X": [],
                "THUMB_CMC_Y": [],
                "THUMB_MCP_X": [],
                "THUMB_MCP_Y": [],
                "THUMB_IP_X": [],
                "THUMB_IP_Y": [],
                "THUMB_TIP_X": [],
                "THUMB_TIP_Y": [],
                "INDEX_FINGER_MCP_X": [],
                "INDEX_FINGER_MCP_Y": [],
                "INDEX_FINGER_PIP_X": [],
                "INDEX_FINGER_PIP_Y": [],
                "INDEX_FINGER_DIP_X": [],
                "INDEX_FINGER_DIP_Y": [],
                "INDEX_FINGER_TIP_X": [],
                "INDEX_FINGER_TIP_Y": [],
                "MIDDLE_FINGER_MCP_X": [],
                "MIDDLE_FINGER_MCP_Y": [],
                "MIDDLE_FINGER_PIP_X": [],
                "MIDDLE_FINGER_PIP_Y": [],
                "MIDDLE_FINGER_DIP_X": [],
                "MIDDLE_FINGER_DIP_Y": [],
                "MIDDLE_FINGER_TIP_X": [],
                "MIDDLE_FINGER_TIP_Y": [],
                "RING_FINGER_MCP_X": [],
                "RING_FINGER_MCP_Y": [],
                "RING_FINGER_PIP_X": [],
                "RING_FINGER_PIP_Y": [],
                "RING_FINGER_DIP_X": [],
                "RING_FINGER_DIP_Y": [],
                "RING_FINGER_TIP_X": [],
                "RING_FINGER_TIP_Y": [],
                "PINKY_MCP_X": [],
                "PINKY_MCP_Y": [],
                "PINKY_PIP_X": [],
                "PINKY_PIP_Y": [],
                "PINKY_DIP_X": [],
                "PINKY_DIP_Y": [],
                "PINKY_TIP_X": [],
                "PINKY_TIP_Y": [],
                "CLASS": []
            }

    
    gui.theme(GUI_THEME)
    
    right_col = [
                    [gui.Image(key="image",size=(224,224),expand_x=False,expand_y=False,)],
                    [gui.Text("Image 224x224",justification='center',expand_x=True)]
                ]
    
    left_col  = [
                    [gui.RealtimeButton("  Rock   ",key="rock")],
                    [gui.Text(f"Rock     : {rock_count}",key="rock_text",pad=[[5,5],[5,20]])],
                    [gui.RealtimeButton("  Paper  ",key="paper")],
                    [gui.Text(f"Paper    : {paper_count}",key="paper_text",pad=[[5,5],[5,20]])],
                    [gui.RealtimeButton("Scissors ",key="scissors")],
                    [gui.Text(f"Scissors : {scissors_count}",key="scissors_text",pad=[[5,5],[5,20]])],
                    [gui.Button("Save ",key="save",expand_x=True,pad=[[10,10],[10,10]])]
                ]
    
    layout    = [
                    [
                        gui.Column(left_col), 
                        gui.VerticalSeparator(), 
                        gui.Column(right_col)
                    ]
                ]
    
    window = gui.Window('RockPaperScissors - Data Collector',layout)
    
    def process_result(result: 'vision.HandLandmarkerResult', output_image: mp.Image, timestamp_ms: int):
        global result_points
        result_points = []
        for result in result.hand_landmarks:
            for landmark in result:
                result_points.append((landmark.x,landmark.y))
    
    cap = cv2.VideoCapture(0)
    
    base_options = python.BaseOptions(model_asset_path = MODEL_PATH)
    running_mode = vision.RunningMode.LIVE_STREAM
    options      = vision.HandLandmarkerOptions(base_options=base_options,running_mode=running_mode,num_hands=1,result_callback=process_result)
    detector     = vision.HandLandmarker.create_from_options(options)

    while True:
        event, values = window.read(timeout=20)
        if event == gui.WIN_CLOSED:
            return
        
        if event == "rock" and len(result_points)>0:
            for i in range(0,len(result_points)*2,2):
                data[list(data.keys())[i]].append(result_points[i//2][0])
                data[list(data.keys())[i+1]].append(result_points[i//2][1])
            
            data["CLASS"].append(CLASS_ROCK)
            rock_count+=1
        
        if event == "paper" and len(result_points)>0:
            for i in range(0,len(result_points)*2,2):
                data[list(data.keys())[i]].append(result_points[i//2][0])
                data[list(data.keys())[i+1]].append(result_points[i//2][1])
            
            data["CLASS"].append(CLASS_PAPER)
            paper_count+=1
            
        if event == "scissors" and len(result_points)>0:
            for i in range(0,len(result_points)*2,2):
                data[list(data.keys())[i]].append(result_points[i//2][0])
                data[list(data.keys())[i+1]].append(result_points[i//2][1])
            
            data["CLASS"].append(CLASS_SCISSORS)
            scissors_count+=1
            
        if event == "save":
            save_path =  f'{DATA_SAVE_FOLDER}/rps_{time.time()}.csv'
            
            df = pd.DataFrame(data)
            df.to_csv(save_path)
            gui.popup_ok(f"Saved into {save_path}",title="Info")
        
        _ , frame    = cap.read()
        height,width = frame.shape[0],frame.shape[1]
        frame        = cv2.resize(frame,(int((width*FRAME_WIDTH)/height),FRAME_HEIGHT))
        
        center       = frame.shape     
        x = center[1]/2 - FRAME_WIDTH/2
        y = center[0]/2 - FRAME_HEIGHT/2
        frame        = frame[int(y):int(y+FRAME_HEIGHT),int(x):int(x+FRAME_WIDTH)]
        
        mp_frame  = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        
        detector.detect_async(mp_frame,int(time.time()*1000))
        
        for point in result_points:
            frame = cv2.circle(frame,(int(point[0]*FRAME_WIDTH),int(point[1]*FRAME_HEIGHT)),3,(0,200,0),thickness=3)
        
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        
        window[        'image'].update(data=imgbytes)
        window[    'rock_text'].update(f"Rock     : {rock_count}")
        window[   'paper_text'].update(f"Paper    : {paper_count}")
        window['scissors_text'].update(f"Scissors : {scissors_count}")

if __name__ == '__main__':
    main()