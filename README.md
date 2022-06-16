# Behavirol-Cloning-in-Close-Loop-System-Using-RNN

## Downloads
- [Training Dataset](https://drive.google.com/drive/folders/1uyB14puU97SSEH-J0UQmiFLNk3_FFVLI?usp=sharing)

## Start Training
- choose **path_name** to determine where to store log and model 
```
python main.py --path_name mse_model
```
- There are **mse** or **dilate** loss
  - dilate loss is not implemented yet
```
python main.py --path_name dilate_model --loss_type dilate
```

