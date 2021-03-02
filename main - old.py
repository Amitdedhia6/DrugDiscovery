from Generator import GeneratorNet
from Discriminator import DiscriminatorNet
from common import *
from torch import nn, optim
import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


elements_to_idx = {}
idx_to_elements = {}
vocab_size = 0
d_errors = []
g_errors = []

def init_vocab():
    filepath = os.path.join(base_data_path, "vocab.txt")
    f = open(filepath, "r")
    elements_list = f.readline()
    f.close()

    index = 0
    elements_to_idx[' '] = index
    for element in elements_list:
        index = index + 1
        elements_to_idx[element] = index
        idx_to_elements[index] = element

    global vocab_size
    vocab_size = len(elements_to_idx)
    pass

def convert_to_numeric(np_text):
    np_result = np.zeros(shape=(len(np_text), max_seq_length))
    for i, sequence in enumerate(np_text):
        np_result[i] = [elements_to_idx[element] if element in elements_to_idx else 0 for element in sequence]
    return np_result

def convert_to_text(np_numeric):
    string_type = "S" + str(max_seq_length)
    np_sequences = np.empty([len(np_numeric)], dtype=string_type) 

    for i, _sequence in enumerate(np_numeric):
        item_idx_list = np_numeric[i]
        item_idx_list = np.rint(item_idx_list).astype(int)
        txt = [idx_to_elements[item_idx] if item_idx in idx_to_elements else ' ' for item_idx in item_idx_list]
        np_sequences[i] = "".join(txt).strip()
    return np_sequences

def load_real_data():
    filepath = os.path.join(base_data_path,"dataset_v1.csv")
    df = pd.read_csv(filepath, sep=",", header = 0) 

    df_train = df[df['SPLIT'] == 'train']
    df_train = df_train[df_train['SMILES'].map(len) <= max_seq_length]
    df_train = df_train['SMILES'].str.rjust(max_seq_length)

    df_test = df[df['SPLIT'] != 'train']
    df_test = df_test[df_test['SMILES'].map(len) <= max_seq_length]
    df_test = df_test['SMILES'].str.rjust(max_seq_length)

    training_np = df_train.to_numpy()
    eighty_percent = int(len(training_np) * 0.8)
    train_data, val_data = training_np[:eighty_percent], training_np[eighty_percent:]
    test_data = df_test.to_numpy()

    train_data = convert_to_numeric(train_data)
    val_data = convert_to_numeric(val_data)
    test_data = convert_to_numeric(test_data)

    train_labels = np.ones(len(train_data))
    val_labels = np.ones(len(val_data))
    test_labels = np.ones(len(test_data))

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def train_discriminator(discriminator, optimizer, real_data, fake_data):
    discriminator.train()
    N = real_data.size(0)
    
    # Reset gradients
    
    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    error_real = F.binary_cross_entropy(prediction_real, ones_target(N).to(device) )

    prediction_fake = discriminator(fake_data)
    error_fake = F.binary_cross_entropy(prediction_fake, zeros_target(N).to(device))
    
    optimizer.zero_grad()
    error_real.backward()
    error_fake.backward()
    optimizer.step()

    # 1.2 Train on Fake Data
    # Calculate error and backpropagate
    
    # 1.3 Update weights with gradients
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(discriminator, generator, optimizer, fake_data):
    generator.train()

    N = fake_data.size(0)

    # Reset gradients
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)

    # Calculate error and backpropagate
    error = F.binary_cross_entropy(prediction, ones_target(N).to(device))
    optimizer.zero_grad()
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error, prediction

def train_model():
    num_epochs = 200

    discriminator = DiscriminatorNet(vocab_size).to(device)
    generator = GeneratorNet().to(device)

    print(device)

    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.000002)
    g_optimizer = optim.Adam(generator.parameters(), lr=0.002)

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_real_data()

    train_data = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    val_data = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_labels))
    test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    batch_size = 1000

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

    for epoch in range(num_epochs):
        d_error_total, g_error_total = 0, 0
        d_pred_real_total, d_pred_fake_total = None, None
        g_pred_fake_total = None
        #discriminator.init_hidden = True
        for n_batch, (real_batch,_) in enumerate(train_loader):
            N = real_batch.size(0)
            # 1. Train Discriminator
            real_data = real_batch
            # Generate fake data and detach 
            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N, vocab_size).to(device)).detach()
            real_data = real_data.to(device)
            fake_data = fake_data.to(device)

            if n_batch % 100 == 0:
                print("Epoch - Batch:", epoch, "-", n_batch)
                print("Discriminator Real data distribution:", torch.histc(real_data.detach().cpu(), bins = 51, min=0, max=50))
                print("Discriminator Fake data distribution:", torch.histc(fake_data.detach().cpu(), bins = 51, min=0, max=50))
            #print("Batch no", n_batch)
            #print("Discriminator Real data:", real_data)
            #print("Discriminator Fake data:", fake_data)
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                train_discriminator(discriminator, d_optimizer, real_data, fake_data)

            # 2. Train Generator
            # Generate fake data
            input_noise = noise(N, vocab_size).to(device)
            fake_data = generator(input_noise).to(device)
            if n_batch % 100 == 0:
                print("Generator Fake data distribution:", torch.histc(fake_data.detach().cpu(), bins = 51, min=0, max=50))

            # Train Gnerator
            g_error, g_pred_fake = train_generator(discriminator, generator, g_optimizer, fake_data)
            #print("Generator Fake data:", fake_data)
            #print("Discriminator output:")
            #print("     REAL:", d_pred_real)
            #print("     FAKE:", d_pred_fake)
            #print("-----------------------------------")
            d_error_total += d_error
            g_error_total += g_error
            if d_pred_real_total is None:
                d_pred_real_total = d_pred_real
            else:
                d_pred_fake_total = torch.cat((d_pred_fake_total, d_pred_fake), dim=0)
            if d_pred_fake_total is None:
                d_pred_fake_total = d_pred_fake
            else:
                d_pred_fake_total = torch.cat((g_pred_fake_total, g_pred_fake), dim=0)
            if g_pred_fake_total is None:
                g_pred_fake_total = g_pred_fake
            else:
                g_pred_fake_total = torch.cat((g_pred_fake_total, g_pred_fake), dim=0)

        d_errors.append(d_error_total)
        g_errors.append(g_error_total)

        if epoch < 5:
            dpr = d_pred_real_total.detach().cpu().numpy()
            dpf = d_pred_fake_total.detach().cpu().numpy()
            gpf = g_pred_fake_total.detach().cpu().numpy()
            #np.savetxt(os.path.join(base_output_path, 'dpr' + str(epoch) + '.csv'), np.asarray(dpr), delimiter=',', fmt='%1.3f')  
            #np.savetxt(os.path.join(base_output_path, 'dpf' + str(epoch) + '.csv'), np.asarray(dpf), delimiter=',', fmt='%1.3f')  
            #np.savetxt(os.path.join(base_output_path, 'gpf' + str(epoch) + '.csv'), np.asarray(gpf), delimiter=',', fmt='%1.3f')  

        if epoch % 1 == 0:
            print("At the end of epoch: ", epoch)
            print("d_error = ", d_error_total, "g_error = ", g_error_total)
            torch.set_printoptions(precision=2)
            print("Discriminator output:")
            print("     REAL:", d_pred_real)
            print("     FAKE:", d_pred_fake)
            print("**********************************")
            print()

            np.savetxt(os.path.join(base_output_path, 'd_errors.csv'), np.asarray(d_errors), delimiter=',', fmt='%1.3f')  
            np.savetxt(os.path.join(base_output_path, 'g_errors.csv'), np.asarray(g_errors), delimiter=',', fmt='%1.3f')  

    filepath = os.path.join(base_model_path, "discriminator.pt")
    torch.save(discriminator, filepath)
    filepath = os.path.join(base_model_path, "generator.pt")
    torch.save(generator, filepath)

    return filepath

if __name__ == '__main__':
    # Main program starts here
    word = "DelleuD"
    print(Palindrome.is_palindrome(word))

    init_vocab()
    train_model()

    pass