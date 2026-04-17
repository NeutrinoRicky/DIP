import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def compute_batch_psnr_ssim(outputs, targets, window_size=11):
    """
    Compute batch-average PSNR and SSIM for model outputs and targets.

    Args:
        outputs (torch.Tensor): Model predictions in [-1, 1], shape (N, C, H, W).
        targets (torch.Tensor): Ground-truth in [-1, 1], shape (N, C, H, W).
        window_size (int): Window size used in SSIM mean filter.

    Returns:
        tuple: (psnr_value, ssim_value) as Python floats.
    """
    with torch.no_grad():
        # Convert from [-1, 1] to [0, 1] for metric computation
        outputs_01 = torch.clamp((outputs.detach() + 1.0) * 0.5, 0.0, 1.0)
        targets_01 = torch.clamp((targets.detach() + 1.0) * 0.5, 0.0, 1.0)

        # PSNR
        mse = torch.mean((outputs_01 - targets_01) ** 2, dim=(1, 2, 3))
        psnr = 10.0 * torch.log10(1.0 / (mse + 1e-8))
        psnr_value = psnr.mean().item()

        # SSIM (average-pooling approximation)
        padding = window_size // 2
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.avg_pool2d(outputs_01, kernel_size=window_size, stride=1, padding=padding)
        mu_y = F.avg_pool2d(targets_01, kernel_size=window_size, stride=1, padding=padding)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.avg_pool2d(outputs_01 * outputs_01, kernel_size=window_size, stride=1, padding=padding) - mu_x2
        sigma_y2 = F.avg_pool2d(targets_01 * targets_01, kernel_size=window_size, stride=1, padding=padding) - mu_y2
        sigma_xy = F.avg_pool2d(outputs_01 * targets_01, kernel_size=window_size, stride=1, padding=padding) - mu_xy

        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        ssim_map = numerator / (denominator + 1e-8)
        ssim_value = ssim_map.mean(dim=(1, 2, 3)).mean().item()

    return psnr_value, ssim_value

def save_curve_plot(series_dict, title, y_label, save_path):
    """
    Save metric curves as an image with OpenCV.

    Args:
        series_dict (dict): Dict of {label: [values_per_epoch]}.
        title (str): Plot title.
        y_label (str): Y axis label.
        save_path (str): Output image path.
    """
    if not series_dict:
        return

    lengths = [len(v) for v in series_dict.values() if len(v) > 0]
    if not lengths:
        return

    epoch_count = max(lengths)
    all_values = []
    for values in series_dict.values():
        if values:
            all_values.extend(values)
    y_min = float(min(all_values))
    y_max = float(max(all_values))
    if abs(y_max - y_min) < 1e-8:
        y_max = y_min + 1.0
    pad = 0.05 * (y_max - y_min)
    y_min -= pad
    y_max += pad

    width, height = 1100, 700
    left, right, top, bottom = 100, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    # Axes
    cv2.line(img, (left, top), (left, top + plot_h), (0, 0, 0), 2)
    cv2.line(img, (left, top + plot_h), (left + plot_w, top + plot_h), (0, 0, 0), 2)

    # Grid and y ticks
    for i in range(6):
        y = int(top + i * (plot_h / 5))
        value = y_max - i * (y_max - y_min) / 5
        cv2.line(img, (left, y), (left + plot_w, y), (230, 230, 230), 1)
        cv2.putText(img, f"{value:.3f}", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    # X ticks
    tick_count = min(6, epoch_count)
    if tick_count == 1:
        tick_indices = [0]
    else:
        tick_indices = [int(round(i * (epoch_count - 1) / (tick_count - 1))) for i in range(tick_count)]
    for idx in tick_indices:
        x = left + int(idx * plot_w / max(epoch_count - 1, 1))
        cv2.line(img, (x, top), (x, top + plot_h), (230, 230, 230), 1)
        cv2.putText(img, str(idx + 1), (x - 10, top + plot_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)

    # Curves
    colors = {
        'train_loss': (0, 128, 255),   # Orange
        'val_loss': (0, 180, 0),       # Green
        'train_psnr': (255, 0, 0),     # Blue
        'val_psnr': (180, 0, 180),     # Purple
        'train_ssim': (255, 128, 0),   # Light blue
        'val_ssim': (0, 0, 255),       # Red
    }
    legend_x, legend_y = left + 10, 30

    for i, (label, values) in enumerate(series_dict.items()):
        if not values:
            continue
        color = colors.get(label, (80, 80, 80))
        pts = []
        for epoch_idx, value in enumerate(values):
            x = left + int(epoch_idx * plot_w / max(epoch_count - 1, 1))
            y = top + int((y_max - value) * plot_h / max(y_max - y_min, 1e-8))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, 2)

        # Legend
        lx = legend_x + i * 170
        cv2.line(img, (lx, legend_y), (lx + 24, legend_y), color, 3)
        cv2.putText(img, label, (lx + 30, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)

    # Labels
    cv2.putText(img, title, (left, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(img, "Epoch", (left + plot_w // 2 - 30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(img, y_label, (15, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

def save_training_logs(history, save_dir='metrics'):
    """
    Save training history to CSV and curve images.

    Args:
        history (dict): Dict containing epoch-wise metric lists.
        save_dir (str): Directory for logs and plots.
    """
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, 'training_history.csv')
    epoch_num = len(history['train_loss'])
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_psnr', 'val_psnr', 'train_ssim', 'val_ssim'])
        for i in range(epoch_num):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['val_loss'][i],
                history['train_psnr'][i],
                history['val_psnr'][i],
                history['train_ssim'][i],
                history['val_ssim'][i],
            ])

    save_curve_plot(
        {'train_loss': history['train_loss'], 'val_loss': history['val_loss']},
        title='Train/Validation Loss per Epoch',
        y_label='L1 Loss',
        save_path=os.path.join(save_dir, 'loss_curve.png')
    )
    save_curve_plot(
        {'train_psnr': history['train_psnr'], 'val_psnr': history['val_psnr']},
        title='Train/Validation PSNR per Epoch',
        y_label='PSNR (dB)',
        save_path=os.path.join(save_dir, 'psnr_curve.png')
    )
    save_curve_plot(
        {'train_ssim': history['train_ssim'], 'val_ssim': history['val_ssim']},
        title='Train/Validation SSIM per Epoch',
        y_label='SSIM',
        save_path=os.path.join(save_dir, 'ssim_curve.png')
    )

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(image_rgb)

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, outputs, 'train_results', epoch)

        # Compute the loss
        loss = criterion(outputs, image_semantic)
        batch_psnr, batch_ssim = compute_batch_psnr_ssim(outputs, image_semantic)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item()
        running_psnr += batch_psnr
        running_ssim += batch_ssim

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(dataloader)
    avg_train_psnr = running_psnr / len(dataloader)
    avg_train_ssim = running_ssim / len(dataloader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] Train Summary - '
        f'Loss: {avg_train_loss:.4f}, PSNR: {avg_train_psnr:.4f}, SSIM: {avg_train_ssim:.4f}'
    )
    return avg_train_loss, avg_train_psnr, avg_train_ssim

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            batch_psnr, batch_ssim = compute_batch_psnr_ssim(outputs, image_semantic)
            val_loss += loss.item()
            val_psnr += batch_psnr
            val_ssim += batch_ssim

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    avg_val_psnr = val_psnr / len(dataloader)
    avg_val_ssim = val_ssim / len(dataloader)
    print(
        f'Epoch [{epoch + 1}/{num_epochs}] Validation Summary - '
        f'Loss: {avg_val_loss:.4f}, PSNR: {avg_val_psnr:.4f}, SSIM: {avg_val_ssim:.4f}'
    )
    return avg_val_loss, avg_val_psnr, avg_val_ssim

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler = StepLR(optimizer, step_size=200, gamma=0.2)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_psnr': [],
        'val_psnr': [],
        'train_ssim': [],
        'val_ssim': [],
    }

    # Training loop
    num_epochs = 300
    for epoch in range(num_epochs):
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs
        )
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, criterion, device, epoch, num_epochs
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_psnr'].append(train_psnr)
        history['val_psnr'].append(val_psnr)
        history['train_ssim'].append(train_ssim)
        history['val_ssim'].append(val_ssim)

        # Update visualization logs every epoch
        save_training_logs(history, save_dir='metrics')

        # Step the scheduler after each epoch
        scheduler.step()

        # Save model checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
