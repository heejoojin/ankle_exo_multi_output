from torch.optim import Adam, AdamW, SGD, RMSprop

def get_optimizer(args, model):
    if args.optimizer == 'adam':
        return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return AdamW(model.parameters(), lr=args.lr)  
    elif args.optimizer == 'sgd':
        return SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        return RMSprop(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)   
    else:
        raise NotImplementedError('Unimplemented optimizer!')