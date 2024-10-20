import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import torch

def train_step(model, train_loader, criterion, optimizer, device):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    return loss

def show_image(generated_data, original_data, title="Generated Vs Original Image"):
    with torch.inference_mode():
        generated_data = generated_data.permute(1, 2, 0).cpu().numpy()
        original_data = original_data.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 2)    
    ax[0].imshow(generated_data)
    ax[0].set_title("Generated")
    ax[1].imshow(original_data)
    ax[1].set_title("Original")
    plt.title(title)
    plt.show()

def show_interpolated_image(img1, img2, output):
    with torch.inference_mode():
        img1 = img1.permute(1, 2, 0).cpu().numpy()
        img2 = img2.permute(1, 2, 0).cpu().numpy()
        output = output.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img1)
    ax[0].set_title("Image 1")
    ax[1].imshow(img2)
    ax[1].set_title("Image 2")
    ax[2].imshow(output)
    ax[2].set_title("Interpolated Image")
    plt.show()
def interpolate(a, b, alpha):
    return (1 - alpha) * a + alpha * b

def merge_features(a, b, alpha):
    pivot = int(a.shape[0] * (1-alpha))
    a = a * (1 - alpha)
    b = b * alpha
    return torch.cat((a[:pivot], b[pivot:]))

def adjust_interpolation_using_graph(img1, img2, encoder, decoder, interpolate=interpolate):
    encoder = encoder.to("cpu")
    decoder = decoder.to("cpu")
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    alpha = 0
    img1_enc = encoder(img1.unsqueeze(0)).squeeze()
    img2_enc = encoder(img2.unsqueeze(0)).squeeze()
    interpolated_val = interpolate(img1_enc, img2_enc, alpha)
    interpolated_img = decoder(interpolated_val.unsqueeze(0)).squeeze()
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img1.permute(1, 2, 0))
    ax[0].set_title("Image 1")
    ax[0].axis("off")
    ax[1].imshow(img2.permute(1, 2, 0))
    ax[1].set_title("Image 2")
    ax[1].axis("off")
    ax[2].imshow(interpolated_img.detach().permute(1, 2, 0))
    ax[2].set_title("Interpolated Image")
    ax[2].axis("off")
    slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    alpha_slider = Slider(
        ax=slider_ax,
        label="alpha",
        valmin=0,
        valmax=1,
        valinit=alpha,
        valstep=0.01,
    )
    def update(val):
        alpha = alpha_slider.val
        interpolated_val = interpolate(img1_enc, img2_enc, alpha)
        interpolated_img = decoder(interpolated_val.unsqueeze(0)).squeeze()
        ax[2].imshow(interpolated_img.detach().permute(1, 2, 0))
    alpha_slider.on_changed(update)
    plt.show()
    return alpha