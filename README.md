# Future Person Localization in First-Person Videos (PyTorch version)


# Create datasets  
Folows steps in original github for downloading datasets

Preprocessing dataset:  
```
python utils/create_dataset.py data/id_test.txt --traj_length 20 --traj_skip 2 --nb_splits 5 --seed 1701 --traj_skip_test 5
```


# Train 

### Chainer still
```
python -u train_cv.py --root_dir experiments/5fold_proposed_only --in_data dataset/id_test_190614_225910_20_sp5_2_5.joblib --nb_iters 17000 --iter_snapshot 1000 --optimizer adam --height 960 --batch_size 64 --save_model  --nb_train -1 --pred_len 10 --channel_list 32 64 128 128 --deconv_list 256 128 64 32 --ksize_list 3 3 3 3 --inter_list 256 --input_len 10 --lr_step_list 5000 10000 15000 --model cnn --nb_splits 5 --eval_split 4 --gpu 1 
```

``` Simpler Arch
 python -u train_cv.py --root_dir experiments/5fold_proposed_only --in_data dataset/id_test_190614_225910_20_sp5_2_5.joblib --nb_iters 17000 --iter_snapshot 1000 --optimizer adam --height 960 --batch_size 64 --save_model  --nb_train -1 --pred_len 10 --channel_list 32 --deconv_list 32 --ksize_list 3 --input_len 10 --lr_step_list 5000 10000 15000 --model cnn --nb_splits 5 --eval_split 4 --gpu 1 
```

### Pytorch
```
python train_pytorch.py --root_dir experiments/5fold_proposed_only --in_data dataset/id_test_190614_225910_20_sp5_2_5.joblib --nb_iters 17000 --iter_snapshot 100 --height 960 --batch_size 64 --save_model  --nb_train -1 --pred_len 10 --input_len 10 --nb_splits 5 --eval_split 4 --optimizer Adam
```

# To-dos 
- Finalizing train_pytorch.py with validation module
- Conver dataset iterators from chainer pytorch. 
- Compare results with original codes(used Chainer)

