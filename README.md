### Convolutional Neural Network-based Gait Phase Estimation using a Robotic Ankle Exoskeleton

![](https://github.com/heejoojin/ankle_exo/blob/main/model_architecture.png)
![](https://github.com/heejoojin/ankle_exo/blob/main/plots/grouped_rmse.png)

- Train
```bash
python main.py --mode=train --save_name=*name --data_type=*data_type --test_type=*test_type --task=multi --model=cnn --scheduler=plateau --window_size=120 --kernel_size=40 --batch_size=128 --dropout=0.2 --epoch=100 --optimizer=adam --lr=0.001 --shuffle
```

- Test
```bash
python main.py --mode=test --save_name=*name --data_type=*data_type --test_type=*test_type --task=multi
```

- Plot Results
```bash
python main.py --mode=plot
```

- Plot Raw Data
```bash
python main.py --mode=rawdata
```
