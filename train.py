import torch
from torch import optim
from torch.utils.data import DataLoader
from model import Model
from parse import parse_args
import data_loader
from tqdm import tqdm

cuda_device = '0'

def my_collate_train(batch):
    user_id = [item[0] for item in batch]
    pos_item = [item[1] for item in batch]
    neg_item = [item[2] for item in batch]

    user_id = torch.LongTensor(user_id)
    pos_item = torch.LongTensor(pos_item)
    neg_item = torch.LongTensor(neg_item)

    return [user_id, pos_item, neg_item]


def one_train(Data, opt):
    print(opt)
    print('Building dataloader >>>>>>>>>>>>>>>>>>>')

    train_dataset = Data.train_dataset
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=opt.batch_size, collate_fn=my_collate_train)

  
    device = torch.device("cuda:{0}".format(cuda_device))

    print(device)
    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        sd = checkpoint['sd']
        optimizer_sd = checkpoint['opt']

    print("building model >>>>>>>>>>>>>>>")
    model = Model(Data, opt, device)

    if opt.loadFilename != None:
        model.load_state_dict(sd)


    print('Building optimizers >>>>>>>')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_every_step, gamma=opt.lr_decay)


    print('Start training...')
    start_epoch = 0
    if opt.loadFilename != None:
        checkpoint = torch.load(opt.loadFilename)
        start_epoch = checkpoint['epoch'] + 1

    model = model.to(device)
    model.train()

    for epoch in range(start_epoch, opt.epoch):
        avg_loss = 0
        with tqdm(total=len(train_loader), desc="epoch"+str(epoch)) as pbar:
            for index, (user_id, pos_item, neg_item) in enumerate(train_loader):
                
                user_id = user_id.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                loss = model(user_id, pos_item, neg_item, opt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(1)

                pbar.set_postfix(loss=f'{loss:.4f}')
                avg_loss += loss
        file_path = 'checkpoint/model_{}_{}_{}_{}_{}.tar'.format(epoch, avg_loss / len(user_id), opt.lr,
                                                                 opt.K_pos_ratio, opt.K_neg_ratio)
        torch.save({
            'sd': model.state_dict(),
            'opt': opt,
        }, file_path)

        scheduler.step()

    model.eval()

opt = parse_args()
interact_train, interact_test, user_num, item_num = data_loader.data_load(opt.dataset_name, test_dataset= True, bottom=opt.implcit_bottom)
Data = data_loader.Data(interact_train, interact_test, user_num, item_num)
one_train(Data, opt)

