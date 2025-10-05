python train.py --dataset photo --epochs 100 --lr 0.005 --hidden1 1024 --hidden2 64 --beta 0.1  --batch_size 0  --train_ratio 0.3
python train.py --dataset reddit --epochs 100 --lr 0.005 --hidden1 1024 --hidden2 64 --beta 0.1  --batch_size 0  --train_ratio 0.3
python train.py --dataset elliptic --epochs 100 --lr 0.0005 --hidden1 1024 --hidden2 64 --beta 1.0  --batch_size 1024  --train_ratio 0.3
python train.py --dataset questions --epochs 100 --lr 0.0005 --hidden1 1024 --hidden2 64 --beta 1.0  --batch_size 1024  --train_ratio 0.3
python train.py --dataset tfinance --epochs 100 --lr 0.005 --hidden1 1024 --hidden2 64 --beta 1.0  --batch_size 1024  --train_ratio 0.3
python train.py --dataset amazon --epochs 500 --lr 0.005 --hidden1 1024 --hidden2 64 --beta 1.0  --batch_size 0  --train_ratio 0.3
python train.py --dataset tolokers --epochs 500 --lr 0.005 --hidden1 1024 --hidden2 64 --beta 1.0  --batch_size 0 --train_ratio 0.3
