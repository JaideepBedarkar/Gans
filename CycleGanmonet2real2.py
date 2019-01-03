
# coding: utf-8



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
from torchvision import transforms, utils,datasets
import matplotlib.pyplot as plt
import torchvision



from skimage import io, transform





import os
from random import randint





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)





def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().numpy()
    plt.imshow((np.transpose(npimg, (1, 2, 0))))
    plt.show()

def imsave(a,img):
    #img = img / 2 + 0.5     # unnormalize
    img = img[0].cpu()
    npimg = img.detach().numpy()
    io.imsave('images/' + str(a) + '.jpg', np.transpose(npimg, (1, 2, 0)))



#renaming dataset
#import os
#i=1
#os.chdir('E:/new project/monet2photo/trainB')
#for filename in os.listdir():
#    dst = str(i).zfill(4) + ".jpg"
#    src = filename

#    os.rename(src, dst)
#    i += 1


# In[7]:


class Imagedataset(Dataset):
    def __init__(self,root_dir_A,root_dir_B,transform = None):
        self.root_dir_A = root_dir_A
        self.root_dir_B = root_dir_B
        self.transform = transform
        self.list_A = os.listdir(root_dir_A)
        self.list_B = os.listdir(root_dir_B)
    def __len__(self):
        return max(len(self.list_A),len(self.list_B))
    def __getitem__(self,idx):
        if idx < len(self.list_A):
            img_name_A = os.path.join(self.root_dir_A,self.list_A[idx])
            image_A = io.imread(img_name_A)
        else:
            img_name_A= os.path.join(self.root_dir_A,self.list_A[randint(0,len(self.list_A)-1)])
            image_A = io.imread(img_name_A)
        if idx < len(self.list_B):
            img_name_B = os.path.join(self.root_dir_B,self.list_B[idx])
            image_B = io.imread(img_name_B)
        else:
            img_name_B= os.path.join(self.root_dir_B,self.list_B[randint(0,len(self.list_B)-1)])
            image_B = io.imread(img_name_B)
        if self.transform :
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
        return image_A,image_B




######DATA_LOADING##########
data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = Imagedataset("./../monet2photo/TrainA/trainA","./../monet2photo/TrainB/trainB",transform=data_transform)
#train_dataset_real = Imagedataset("E:/new project/monet2photo/trainB",transform=transforms.ToTensor())



train_loader = DataLoader(train_dataset, batch_size=1,shuffle=True)
#real_train_loader = DataLoader(train_dataset_real, batch_size=1,shuffle=True)


# In[9]:


class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.ReflectionPad2d(3),
                                  nn.Conv2d(3,64,kernel_size =6 ,stride=1 ,padding=0),
                                  nn.InstanceNorm2d(16),
                                  nn.ReLU(),
                                  nn.Conv2d(64,128,kernel_size =3 ,stride=2 ,padding=1),
                                  nn.InstanceNorm2d(128),
                                  nn.ReLU(),
                                  nn.Conv2d(128,256,kernel_size =3 ,stride=2 ,padding=1),
                                  nn.InstanceNorm2d(16),
                                  nn.ReLU())
        self.resnet = Residual_block(256,256,9)
        self.block2 = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size =3 ,stride=2 ,padding=1,output_padding =1),
                                  nn.InstanceNorm2d(128),
                                  nn.ReLU(),
                                  nn.ConvTranspose2d(128,64,kernel_size =3 ,stride=2 ,padding=1,output_padding =1),
                                  nn.InstanceNorm2d(64),
                                  nn.ReLU(),
                                  nn.Conv2d(64,3,kernel_size =7 ,stride=1 ,padding=1),
                                  nn.InstanceNorm2d(3),
                                  nn.Tanh())
    def forward(self,z):
        out = self.block1(z)
        out = self.resnet(out)
        out = self.block2(out)
        return out


# In[10]:


class Residual_block(nn.Module):
    def __init__(self,input_channels,F1,res_blocks):
        super().__init__()
        self.input_channels = input_channels
        self.F1 = F1
        self.res_blocks = res_blocks
        self.block = nn.Sequential(nn.Conv2d(input_channels,F1,kernel_size =3 ,stride=1 ,padding=1),
                                   nn.BatchNorm2d(F1),
                                   nn.ReLU(),
                                   nn.Conv2d(F1,input_channels,kernel_size =3 ,stride=1 ,padding=1),
                                   nn.BatchNorm2d(F1) )
        self.relu = nn.ReLU()
    def forward(self,z):
        temp = z
        for _ in range(self.res_blocks):
            out_shortcut = temp
            out = self.block(temp)
            out = temp + out
            out = self.relu(out)
            temp = out
        return out


# In[11]:


gen = generator().cuda()
x= gen(torch.randn(1,3,256,256).cuda())
x.shape


# In[12]:


class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(3,64,kernel_size =4 ,stride=2 ,padding=1),
                                  #nn.InstanceNorm2d(16),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(64,128,kernel_size =4 ,stride=2 ,padding=1),
                                  nn.InstanceNorm2d(128),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(128,256,kernel_size =4 ,stride=2 ,padding=1),
                                  nn.InstanceNorm2d(256),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(256,512,kernel_size =4 ,stride=2 ,padding=1),
                                  nn.InstanceNorm2d(512),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv2d(512,1,kernel_size =14 ,stride=2 ,padding=0))
    def forward(self,z):
        out = self.block(z)
        return out





#loading Model###
gen_monet = generator()
gen_real = generator()
dis_monet = discriminator()
dis_real = discriminator()

############## Model Criterions and optimizers################


Adversarial_criterion  = nn.MSELoss()
cylcle_consistency_criterion = nn.L1Loss()
optimizer_generator_monet = optim.Adam(gen_monet.parameters(),lr = 0.0002,betas=(0.5,0.999))
optimizer_discriminator_monet = optim.Adam(dis_monet.parameters(),lr = 0.0002,betas=(0.5,0.999))
optimizer_generator_real = optim.Adam(gen_real.parameters(),lr = 0.0002,betas=(0.5,0.999))
optimizer_discriminator_real = optim.Adam(dis_real.parameters(),lr = 0.0002,betas=(0.5,0.999))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

gen_real.apply(weights_init)
gen_monet.apply(weights_init)
dis_real.apply(weights_init)
dis_monet.apply(weights_init)

gen_monet = gen_monet.cuda()
gen_real = gen_real.cuda()
dis_monet = dis_monet.cuda()
dis_real = dis_real.cuda()

#try:
    #Load previously saved model
checkpoint = torch.load('./../models/models2.tar')
gen_real.load_state_dict(checkpoint['gen_real_dict'])
dis_real.load_state_dict(checkpoint['dis_real_dict'])
gen_monet.load_state_dict(checkpoint['gen_monet_dict'])
dis_monet.load_state_dict(checkpoint['dis_monet_dict'])
optimizer_generator_real.load_state_dict(checkpoint['optimizer_generator_real_dict'])
optimizer_discriminator_real.load_state_dict(checkpoint['optimizer_discriminator_real_dict'])
optimizer_generator_monet.load_state_dict(checkpoint['optimizer_generator_monet_dict'])
optimizer_discriminator_monet.load_state_dict(checkpoint['optimizer_discriminator_monet_dict'])
#epoch = checkpoint['epoch']
#except:
    #pass
gen_monet = gen_monet.cuda()
gen_real = gen_real.cuda()
dis_monet = dis_monet.cuda()
dis_real = dis_real.cuda()

gen_monet.train()
gen_real.train()
dis_monet.train()
dis_real.train()




####################### TRAINING CYCLEGAN #######################
print('Initialized parameters.....training strated')

for epoch in range(50):
    i=0
    running_total_discriminator_monet_loss  = 0.0
    running_generator_monet_loss = 0.0
    running_total_discriminator_real_loss  = 0.0
    running_generator_real_loss = 0.0
    #running_cycle_consistency_loss2 = 0.0
    #running_cycle_consistency_loss1 = 0.0
    running_total_cycle_consistency_loss = 0.0
    #running_g_loss = 0.0
    for data_monet,data_real in (train_loader):
        data_monet = data_monet.cuda()
        data_real = data_real.cuda()
        ### Monet 2 real ####
        #zero the parameter gradients
        optimizer_discriminator_real.zero_grad()
        optimizer_generator_real.zero_grad()
        #forward + loss + backprop#

        ###Training discriminator###

        real_out = dis_real(data_real)
        real_loss = Adversarial_criterion(real_out,torch.ones(real_out.shape).cuda())

        fake_real_image = gen_real(data_monet)
        fake_real_out = dis_real(fake_real_image)
        fake_real_loss = Adversarial_criterion(fake_real_out,torch.zeros(fake_real_out.shape).cuda())

        total_discriminator_real_loss = (real_loss + fake_real_loss)*0.5
        total_discriminator_real_loss.backward()
        optimizer_discriminator_real.step()

        ###Training generator###
        fake_real_image = gen_real(data_monet)
        fake_out = dis_real(fake_real_image)
        generator_real_loss = Adversarial_criterion(fake_out,torch.ones(fake_out.shape).cuda())

        #generator_real_loss.backward()
        #optimizer_generator_real.step()


    #################### real 2 monet #########################################

         #zero the parameter gradients
        optimizer_discriminator_monet.zero_grad()
        optimizer_generator_monet.zero_grad()
        #forward + loss + backprop#

        ###Training discriminator###

        monet_out = dis_monet(data_monet)
        monet_loss = Adversarial_criterion(monet_out,torch.ones(monet_out.shape).cuda())

        fake_monet_image = gen_monet(data_real)
        fake_monet_out = dis_monet(fake_monet_image)
        fake_monet_loss = Adversarial_criterion(fake_monet_out,torch.zeros(fake_monet_out.shape).cuda())

        total_discriminator_monet_loss = (monet_loss + fake_monet_loss)*0.5
        total_discriminator_monet_loss.backward()
        optimizer_discriminator_monet.step()

        ###Training generator###
        fake_monet_image = gen_monet(data_real)
        fake_out = dis_monet(fake_monet_image)
        generator_monet_loss = Adversarial_criterion(fake_out,torch.ones(fake_out.shape).cuda())

        #generator_monet_loss.backward()
        #optimizer_generator_monet.step()

   ######################## cycle consistency ##############################
        #optimizer_generator_monet.zero_grad()
        #optimizer_generator_real.zero_grad()

        #fake_real = gen_real(data_monet)
        fake_monet = gen_monet(fake_real_image)
        cycle_consistency_loss_monet = 10*cylcle_consistency_criterion(data_monet,fake_monet)
        #cycle_consistency_loss1.backward()
        #optimizer_generator_monet.step()
        #optimizer_generator_real.step()

        #fake_monet = gen_monet(data_real)
        fake_real = gen_real(fake_monet_image)
        cycle_consistency_loss_real =10*cylcle_consistency_criterion(fake_real,data_real)

        total_cycle_consistency_loss = cycle_consistency_loss_real + cycle_consistency_loss_monet

#########################identity loss#########################################
        identity_loss = cylcle_consistency_criterion(gen_real(data_real),data_real) + cylcle_consistency_criterion(gen_monet(data_monet),data_monet)

############################# UPDATING GENERATOR ###############################
        total_generator_monet_loss =  generator_monet_loss + total_cycle_consistency_loss + 5*identity_loss
        total_generator_real_loss = generator_real_loss + total_cycle_consistency_loss + 5*identity_loss
        total_generator_monet_loss.backward(retain_graph=True)
        total_generator_real_loss.backward(retain_graph=True)
        optimizer_generator_monet.step()
        optimizer_generator_real.step()
  ############ print statistics#####################
        running_total_discriminator_real_loss += total_discriminator_real_loss.item()
        running_total_discriminator_monet_loss += total_discriminator_monet_loss.item()
        running_generator_monet_loss += generator_monet_loss.item()
        running_generator_real_loss += generator_real_loss.item()
        #running_cycle_consistency_loss1 += cycle_consistency_loss1.item()
        #running_cycle_consistency_loss2 += cycle_consistency_loss2.item()
        running_total_cycle_consistency_loss = total_cycle_consistency_loss.item()
        i+=1
        if i % 500 == 499:    # print every 2000 mini-batches
            print('Epoch:%d | images:%d | monet_gen_loss:%.4f | monet_dis_loss:%.4f'%
                   (epoch + 1, i + 1,running_generator_monet_loss  / 500,running_total_discriminator_monet_loss  / 500))
            print('Epoch:%d | images:%d | real_gen_loss:%.4f | real_dis_loss:%.4f'%
                   (epoch + 1, i + 1,running_generator_real_loss  / 500,running_total_discriminator_real_loss  / 500))
            print('g loss:%.4f '%(running_total_cycle_consistency_loss/500))
            running_total_discriminator_monet_loss  = 0.0
            running_generator_monet_loss = 0.0
            running_total_discriminator_real_loss  = 0.0
            running_generator_real_loss = 0.0
            #running_cycle_consistency_loss1 = 0.0
            #running_cycle_consistency_loss2 = 0.0
            running_total_cycle_consistency_loss = 0.0
            imsave(-111,gen_real(data_monet))
            imsave(-222,data_monet)
            #fake_image = gen_real(data_monet)
            #output = fake_image.cpu()
            #imshow(torchvision.utils.make_grid(output))


    torch.save({
        'gen_real_dict': gen_real.state_dict(),
        'dis_real_dict': dis_real.state_dict(),
        'gen_monet_dict': gen_monet.state_dict(),
        'dis_monet_dict': dis_monet.state_dict(),
        'optimizer_generator_real_dict':optimizer_generator_real.state_dict(),
        'optimizer_discriminator_real_dict':optimizer_discriminator_real.state_dict(),
        'optimizer_generator_monet_dict':optimizer_generator_monet.state_dict(),
        'optimizer_discriminator_monet_dict':optimizer_discriminator_monet.state_dict(),
    }, './../models/models2.tar')
    imsave(str(epoch+19) + '_',gen_real(data_monet))
    imsave(str(epoch+19)+"_inp",data_monet)
