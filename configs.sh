'''
MIT License

Copyright (c) 2025 Somayeh Hussaini, Tobias Fischer and Michael Milford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# train on NRD dataset 
pixi run python3 train.py --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4

# test on NRD dataset
pixi run python3 test.py --model_name Apgem --model_type Resnet101-AP-GeM.pt --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name CosPlace --model_type SF_XL --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name EigenPlaces --model_type ResNet50 --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name MixVPR --model_type GCS-Cities --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name NetVLAD --model_type pittsburgh --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name SAD --model_type SAD --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name SALAD --model_type DINOv2 --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4
pixi run python3 test.py --model_name boq --model_type Dinov2 --dataset_name nordland --ref summer --qry winter --num_places 27592 --seq_len 4

# train on ORC dataset
pixi run python3 train.py --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4

# train on ORC dataset
pixi run python3 test.py --model_name Apgem --model_type Resnet101-AP-GeM.pt --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4
pixi run python3 test.py --model_name CosPlace --model_type SF_XL --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4
pixi run python3 test.py --model_name EigenPlaces --model_type ResNet50 --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4
pixi run python3 test.py --model_name MixVPR --model_type GCS-Cities --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4
pixi run python3 test.py --model_name NetVLAD --model_type pittsburgh --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4
pixi run python3 test.py --model_name SAD --model_type SAD --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4
pixi run python3 test.py --model_name SALAD --model_type DINOv2 --dataset_name ORC --ref Rain --qry Dusk --num_places 3800 --seq_len 4


# train on SFU dataset
pixi run python3 train.py --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4

# test on SFU dataset
pixi run python3 test.py --model_name Apgem --model_type Resnet101-AP-GeM.pt --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4
pixi run python3 test.py --model_name CosPlace --model_type SF_XL --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4
pixi run python3 test.py --model_name EigenPlaces --model_type ResNet50 --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4
pixi run python3 test.py --model_name MixVPR --model_type GCS-Cities --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4
pixi run python3 test.py --model_name NetVLAD --model_type pittsburgh --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4
pixi run python3 test.py --model_name SAD --model_type SAD --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4
pixi run python3 test.py --model_name SALAD --model_type DINOv2 --dataset_name SFU-Mountain --ref dry --qry dusk --num_places 385 --seq_len 4

