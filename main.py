from generator import Generator
from discriminator import Discriminator
from common import device, base_data_path, base_model_path, google_cloud
from common import ones_target, zeros_target
import torch
from torch import optim
import os
import pandas as pd
import torch.nn.functional as F
import math
from vocab import Vocab
from smiles_data import SmilesData
from gan_train_parameters import training_param as param
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import random


vocab = Vocab()


def load_real_data():
    if google_cloud:
        filepath = os.path.join(base_data_path, "dataset_v1.csv")
    else:
        filepath = os.path.join(base_data_path, "dataset_v2.csv")

    df = pd.read_csv(filepath, sep=",", header=0)
    train_smiles_data = SmilesData(vocab)
    train_sequence_list = df.SMILES.tolist()
    train_smiles_data.fill(train_sequence_list)

    return train_smiles_data


def print_weights(model):
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name, p.data)


def plot_grad_flow(named_parameters, headline):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    ave_grads = []
    max_grads = []
    min_grads = []
    counts = []
    layers = []
    for n, p in named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            counts.append(p.shape)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            min_grads.append(p.grad.abs().min())

    print(f"    {headline}")
    print("    Layers =", layers)
    print("    Avg_grads =", ave_grads)
    print("    Max_grads =", max_grads)
    print("    Min_grads =", min_grads)
    print("    **********************")

    if 4 == 5:
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.2)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.show()


def train_discriminator(discriminator, optimizer,
                        data, data_l, datatype,
                        apply_grad=True):
    # discriminator.train()

    N = data.size(0)
    prediction = discriminator(data, data_l)

    if datatype == 'real':
        target = ones_target(N, True).to(device)
    elif datatype == 'fake':
        target = zeros_target(N, True).to(device)
    else:
        raise ValueError('Inappropriate argument to function train_discriminator')

    error = F.binary_cross_entropy(prediction, target)

    if datatype == 'real':
        pr_real_correct = (prediction >= 0.8).sum().item()
        accuracy = 100 * (pr_real_correct) // (N)
    else:
        pr_fake_correct = (prediction <= 0.2).sum().item()
        accuracy = 100 * (pr_fake_correct) // (N)

    pr_output = []
    for i in range(20):
        idx = random.randint(0, N-1)
        pr_output.append(prediction[idx].item())
    twodecimals = ["%.2f" % v for v in pr_output]
    print(f"        D-PREDICTIONS {datatype}: {twodecimals}")

    if apply_grad:
        optimizer.zero_grad()
        error.backward()
        plot_grad_flow(discriminator.named_parameters, "DISCRIMINATOR PARAMS")
        optimizer.step()

    return error, accuracy


def train_generator(discriminator, generator, optimizer, fake_data, fake_data_l):
    # discriminator.train()
    # generator.train()

    N = fake_data.size(0)
    prediction_real = discriminator(fake_data, fake_data_l)
    error = F.binary_cross_entropy(prediction_real, ones_target(N, True).to(device))

    pr_real_correct = (prediction_real >= 0.7).sum().item()
    accuracy = (100.0 * pr_real_correct) // N

    pr_real_output = []

    for i in range(20):
        idx = random.randint(0, N-1)
        pr_real_output.append(prediction_real[idx].item())
    twodecimals = ["%.2f" % v for v in pr_real_output]
    print(f"    GENERATOR PRED: {twodecimals}")

    optimizer.zero_grad()
    error.backward()
    plot_grad_flow(generator.named_parameters, "GENERATOR PARAMS")
    optimizer.step()

    return error, accuracy


def train_gan():
    print("GAN training to start on", device)

    print("Now loading data . . .")
    train_smiles_data = load_real_data()
    print("Data loaded")

    discriminator = Discriminator(vocab).to(device)
    generator = Generator(vocab).to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=param.lr_d)
    g_optimizer = optim.Adam(generator.parameters(), lr=param.lr_g)

    num_epochs = param.num_epochs
    batch_size = param.batch_size
    num_steps = param.num_steps
    dataset_size = train_smiles_data.sequence_list_size
    print("dataset_size =", dataset_size)

    # *************
    # THE LOGIC
    # *************
    #   for num_iter
    #       1. for num_steps
    #           (a) train discriminator with samples = batch_size
    #       2. Train generator with samples = batch_size
    #
    # How to determine num_iterations?
    #   num_iter * num_steps * batch_size = dataset_size * num_epochs
    #
    # *************

    num_iterations = math.ceil((dataset_size * num_epochs) / (num_steps * batch_size))
    print(f"num_iterations: {num_iterations}")
    saved_generator_loss = 10000000.0
    start_index = 0
    total_row_count = dataset_size

    print("Iter, d_error_real, d_error_fake, g_error_fake")
    accuracy_dr = 0
    accuracy_df = 0
    apply_grads = True

    for iter in range(num_iterations):
        d_error_real, d_error_fake = 0, 0
        g_error_fake = 0

        for _k in range(num_steps):
            if start_index >= total_row_count:
                start_index = 0
            end_index = start_index + batch_size
            if (end_index - start_index) > total_row_count:
                end_index = total_row_count
            if end_index > total_row_count:
                end_index = total_row_count

            real_data = train_smiles_data.sequence_tensors[start_index: end_index, :, :]
            real_data_l = train_smiles_data.sequence_length_data[start_index: end_index]

            N = end_index - start_index
            fake_data, fake_data_l = generator(N)

            real_data = real_data.to(device)
            real_data_l = real_data_l.to(device)
            fake_data = fake_data.detach().to(device)
            fake_data_l = fake_data_l.to(device)

            if (apply_grads and (accuracy_dr > 90) and (accuracy_df > 90)):
                apply_grads = False
                filepath = os.path.join(base_model_path, "discriminator_" + str(iter) + ".pt")
                torch.save(discriminator, filepath)

            error_real, accuracy_dr = train_discriminator(discriminator, d_optimizer,
                                                          real_data, real_data_l, 'real', apply_grads)
            error_fake, accuracy_df = train_discriminator(discriminator, d_optimizer,
                                                          fake_data, fake_data_l, 'fake', apply_grads)

            d_error_real += error_real.mean()
            d_error_fake += error_fake.mean()

            start_index = end_index

        d_error_real = d_error_real / num_steps
        d_error_fake = d_error_fake / num_steps

        N = batch_size
        fake_data, fake_data_l = generator(N)
        fake_data = fake_data.to(device)
        fake_data_l = fake_data_l.to(device)

        g_error, accuracy_g = train_generator(discriminator, generator, g_optimizer,
                                              fake_data, fake_data_l)

        g_error_fake += g_error.mean()
        print(f"    Accuracy Numbers: D = {accuracy_dr}, {accuracy_df}, G = {accuracy_g}")

        if (g_error_fake < saved_generator_loss) or (iter % 10 == 0):

            if (g_error_fake < saved_generator_loss):
                saved_generator_loss = g_error_fake
                base_folder = base_model_path
                pt_filepath = os.path.join(base_folder, "generator.pt")
                gen_txt_filepath = os.path.join(base_folder, "generatod_samples.txt")
            else:
                base_folder = os.path.join(base_model_path, "tens")
                pt_filepath = os.path.join(base_folder, "generator_" + str(iter) + ".pt")
                gen_txt_filepath = os.path.join(base_folder, "generatod_samples" + str(iter) + ".txt")

            torch.save(generator, pt_filepath)
            with torch.no_grad():
                y, len = generator(500)
                generator_results = generator.get_sequences_from_tensor(y, len)
                with open(gen_txt_filepath, "w") as outfile:
                    outfile.write(f"Iter: {iter}, Loss: {g_error_fake}\n")
                    outfile.write("\n".join(generator_results))

        if iter % 1 == 0:
            print(f"{iter}, {d_error_real.item()}, {d_error_fake.item()}, {g_error_fake.item()}")
            print("--------------------------------------------------------------------------")

    filepath = os.path.join(base_model_path, "discriminator.pt")
    torch.save(discriminator, filepath)
    filepath = os.path.join(base_model_path, "generator.pt")
    torch.save(generator, filepath)

    return filepath


def test_generartor():
    generator = Generator(vocab).to(device)
    generator_result, len = generator(100)
    print(generator_result, len)


if __name__ == '__main__':
    # test_generartor()
    train_gan()

    print("OVER")

    pass
