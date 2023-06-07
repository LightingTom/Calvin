from mcil import MCILModel
from Dataset import CalvinDataset
import torch.optim as op
from torch.utils.data import DataLoader
import torch


# Config the model
lr = 0.0001
max_epoch = 100
eval_freq = 5
device = 'cuda'
batch_size = 32
model = MCILModel().to(device)
optimizer = op.Adam(lr=lr, params=model.parameters())

# Dataset
dataset = CalvinDataset(lang=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
val_dataset = CalvinDataset(train=False, lang=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)


# write the log
def log(epoch, kl_loss, action_loss, total_loss):
    if epoch != -1:
        with open('calvin.log', 'a') as f:
            f.write(f'| Epoch:{epoch} | KL_loss: {kl_loss} | Action_loss: {action_loss} | Total_loss: {total_loss} |\n')
    else:
        with open('calvin.log', 'a') as f:
            f.write(f'| Validation | KL_loss: {kl_loss} | Action_loss: {action_loss} | Total_loss: {total_loss} |\n')
            f.write('----------------------------------------------------------------------------------------------\n')


# Validate the model
def validation(model):
    model.eval()
    with torch.no_grad():
        cnt = 0
        k_loss, a_loss, t_loss = (
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device),
            torch.tensor(0.0).to(device)
        )
        for idx, data in enumerate(val_loader):
            cnt += 1
            img_static = data['rgb_obs']['rgb_static'].to(device)
            robot_obs = data['robot_obs'].to(device)
            actions = data['actions'].to(device)
            lang = data['lang'].to(device)
            kl_loss, action_loss, total_loss, pp_dist, pr_dist = model(img_static, robot_obs, lang, actions)
            k_loss += kl_loss
            a_loss += action_loss
            t_loss += total_loss
        log(-1, k_loss / cnt, a_loss / cnt, t_loss / cnt)


# Clear the previous log
with open('calvin.log', 'w') as f:
    f.write('')
# training step
for epoch in range(max_epoch):
    model.train()
    k_loss, a_loss, t_loss = (
        torch.tensor(0.0).to(device),
        torch.tensor(0.0).to(device),
        torch.tensor(0.0).to(device)
    )
    cnt = 0
    for idx, data in enumerate(dataloader):
        cnt += 1
        img_static = data['rgb_obs']['rgb_static'].to(device)
        robot_obs = data['robot_obs'].to(device)
        actions = data['actions'].to(device)
        lang = data['lang'].to(device)
        # print(img_static.shape) # [16, 32, 3, 200, 200]
        # print(robot_obs.shape) # [16, 32, 15]
        # print(actions.shape) # [16, 32, 7]
        # print(lang.shape) # [16, 384]
        kl_loss, action_loss, total_loss, pp_dist, pr_dist = model(img_static, robot_obs, lang, actions)
        k_loss += kl_loss
        a_loss += action_loss
        t_loss += total_loss
        total_loss.backward()
        optimizer.step()
    log(epoch+1, k_loss / cnt, a_loss / cnt, t_loss / cnt)
    if epoch % eval_freq == 0:
        validation(model)