# 701-Project
Name: Ding Pang  Mark Wu

# Topic: Exploring Models of Neural Style Transfer (NST)

## Results:
### Gatys' model
![alt text](./singleOptResult2.png)
### Single Style Feedforward model
![alt text](./Feed_Forward/outputs/sff/plot.png)
### Multiple Styles Feedforward model
![alt text](./Feed_Forward/outputs/mff/plot.png)
### Arbitrary (kind of) Styles Feedforward model
![alt text](./Feed_Forward/outputs/aff/plot.png)

## Requirements:
```bash
pip install -r requirements.txt
```
### Important Versions:
1. Python: 3
1. tensorflow: 2.4.1 (Cannot be v1)

### Used Datasets:
1. fruits-360: https://www.kaggle.com/moltean/fruits?select=fruits-360
1. The Car Connection Picture Dataset: https://www.kaggle.com/prondeau/the-car-connection-picture-dataset

After downloading, move them into Feed_Forward/, and make sure the car dataset has the name cars ("Feed_Forward/cars/")

## How to use:
1. First, change the IN_Method in style.py to select the desired model
1. ```bash
    cd Feed_Forward/
    python3 style.py
    ```
