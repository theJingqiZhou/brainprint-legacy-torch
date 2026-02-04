# install
```
cd identity_recognition
python3 -m pip install -e .
```
Here `-e` means editable mode, which is optional.

# prepare train/val data

you should modify train.txt/val.txt to your data path, first!
```
remove your data in data/ or data2/
```
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/75b7a729-c114-4d31-881e-23b6d7fbbba7)
## make train.txt
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/cae43883-760f-44aa-8e43-8d420e8742fc)


`Note` if you use mat format data: 
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/768b670f-c44f-46ea-9ae4-84827e5a1ca8)


## create train.jsonl
```
python convert_data.py
```
![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/74e9100f-e0eb-4532-b422-a6c91f1d9c54)

Set `PROFILE`, `DATA_FILE`, and `SAVE_FILE` in `convert_data.py` before running.

`Note` if you use mat format data:
```
python convert_data_mat.py
```
Set `PROFILE`, `DATA_FILE`, and `SAVE_FILE` in `convert_data_mat.py` before running.

# train
modify `src/runtime_config.py` for user config
set `RUNNER = "train"` in `run.py`
```
python run.py
```
`Note` if you use mat format data: set `PROFILE = "mat"` in `run.py`

# test
set `RUNNER = "test"` in `run.py`
```
python run.py
```
`Note` if you use mat format data: set `PROFILE = "mat"` in `run.py`
# deploy
suported pt to onnx
set `RUNNER = "deploy"` in `run.py`
```
python run.py
```
`Note` if you use mat format data: set `PROFILE = "mat"` in `run.py`
# infer demo
```
python infer.py
```
input :EEG data(type = np.array), condition: time length>=1s.
return: identity_map：
| id | user_name | score | count |
| :----: | :----: | :----: | :----: |
| 0 | gjc | - | - |
| 1 | wxc | - | - |
| 2 | yl | - | - |
| 3 | zqy | - | - |
| -1 | unknown | 0 | 0 |

if you want add or delete, please update `src/runtime_config.py`

![image](https://github.com/CUITCHENSIYU/identity_recognition/assets/52771861/6aba7815-a4e8-4004-b481-858ac0865719)

`Note` if you use mat format data, please update `src/runtime_config.py`
