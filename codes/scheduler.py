
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau

def get_scheduler(args, optimizer):
    is_plateau = False

    if args.scheduler == 'cosine':
        # SGDR: Stochastic Gradient Descent with Warm Restarts
        T_max = 10
        eta_min = 1e-5 # minimum learning rate
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
    elif args.scheduler == 'step':
        step_size = 1
        gamma = 0.95
        scheduler = StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
    elif args.scheduler == 'lambda':
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 1 / (epoch+1))
    elif args.scheduler == 'plateau':
        # val loss
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, min_lr=1e-8, patience=args.patience)
        is_plateau = True
    else:
        raise NotImplementedError('Unimplemented scheduler!')

    return is_plateau, scheduler