# Towards-Efficient-and-Secure-Vertical-Federated-Learning-with-Additive-Ensemble
The code implementation of the draft "Towards Efficient and Secure Vertical Federated Learning with Additive Ensemble"

## step 1: Process different datasets with the following scripts.
```python
    run ours/adult_data_process.py 
    run ours/cifar10_data_process.py
    run ours/kchouse_data_process.py
    run ours/song_data_Process.py
```   

## step 2: The proposed AE-VFL with two participants on different datasets.
```python
    run ours/adult_main_additive_2parts.py
    run ours/cifar10_main_additive_2parts
    run ours/kchouse_main_additive_2parts
    run ours/song_main_additive_2parts

```

## step 3: Conduct an experiment with  the centralized learning method.
```python
    run centralizedLearning_adult.py
    run centralizedLearning_cifar10.py
    run centralizedLearning_kchouse.py
    run centralizedLearning_song.py

```

## step 4: Conduct an experiment with the independent learning method.
```python
    run onlywithOnePart_adult.py
    run onlywithOnePart_adult.py
    run onlywithOnePart_adult.py
    run onlywithOnePart_adult.py
    
```
## step 5: Conduct an experiment with the traditional VFL method.
```python
    run VFL_adult.py
    run VFL_adult.py
    run VFL_adult.py
    run VFL_adult.py

```
