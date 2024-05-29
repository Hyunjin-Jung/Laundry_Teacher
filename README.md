# Laundry Teacher

This project is the Team 16 Term Project for the 2024 Deep Learning (RT5101) course at GIST.

## Team 16
DuyoungKim|SeongilKim|YeonjuKim|TaewookKim|HyunjinJung|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/141096154?s=64&v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/100182543?s=64&v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/141096125?s=64&v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/141096377?s=64&v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/67624124?v=4' height=80 width=80px></img>
[Github](https://github.com/hidudu0)|[Github](https://github.com/g1r4ff3)|[Github](https://github.com/YeonjuKim111)|[Github](https://github.com/ktaewook)|[Github](https://github.com/Hyunjin-Jung)

## Directory structure

    Laundry Teacher
    ├─models 
    └─src
        ├─camera_input_live 
        ├─config
        ├─eval
        └─util 


* models : The model used in this project is saved as a .pt file.
* src/camera_input_live : Modified version of the camer_input_live package.
* src/eval : Code related to validation results during training.
* src/util : A folder with codes related to data preprocessing. 

## Task

![그림1](https://github.com/Hyunjin-Jung/Laundry_Teacher/assets/67624124/db4a9e99-40ec-42f4-8df7-f28a3819af20)

![그림1](https://github.com/Hyunjin-Jung/Laundry_Teacher/assets/67624124/c77d4a20-0505-48fd-b9c2-6b2c370b0c8a)

## Dataset

AI-Hub's '['의류 통합 데이터(착용 이미지, 치수 및 원단 정보)'](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=71501)' dataset was processed into the code of the utility folder and used for learning.

# Getting Started

### Installation

```
git clone https://github.com/Hyunjin-Jung/Laundry_Teacher.git
``` 

### Dependencies

To run Laundry Teacher, you need to install the libraries listed in the requirements.txt file on your system.

```
pip install -r requirements.txt
```

### Excute 
1. Please move to the root directory of Laundry Teacher.
2. Please execute the following command.
    ```
    streamlit run app.py
    ```
3. You can access Laundry Teacher by connecting to the displayed IP address.


<details><summary>If your phone camera doesn't work...</summary> 
Please follow the following link to change the chrome setting. 

[Enabling the Microphone/Camera in Chrome for (Local) Unsecure Origins](https://medium.com/@Carmichaelize/enabling-the-microphone-camera-in-chrome-for-local-unsecure-origins-9c90c3149339)
</details>



# Project Results

### Analyze Fiber composition

![그림1](https://github.com/Hyunjin-Jung/Laundry_Teacher/assets/67624124/c0675007-6b5c-41ec-bc94-e215e0c7dbff)

Selecting the "Fabric Information" mode allows you to find out the fiber composition of the desired clothing item, as well as the washing method and precautions for that fiber.

### Laundry Grouping

![그림1](https://github.com/Hyunjin-Jung/Laundry_Teacher/assets/67624124/b0406ed3-6f03-4102-9f73-6d0a3a62ea8c)

In the "Simultaneous Wash Compatibility" mode, you can input a photo to determine whether the item can be washed together with other clothing items. If there are clothing items that cannot be washed together, they will be indicated with a red box, and the washing instructions for those items will be displayed at the bottom.

# Licenses

'camera_input_live' package is a modified version of the [camer_input_live package](https://github.com/blackary/streamlit-camera-input-live?tab=MIT-1-ov-file), which is released under the MIT license.
