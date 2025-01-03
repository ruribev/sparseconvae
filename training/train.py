import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(sparse_encoder, imputation_decoder, subsurface_decoder, dataloader, num_epochs=10, learning_rate=1e-4, device="cuda"):
    sparse_encoder.train()
    imputation_decoder.train()
    subsurface_decoder.train()

    optimizer = optim.AdamW(list(sparse_encoder.parameters()) + list(imputation_decoder.parameters())+ list(subsurface_decoder.parameters()), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_parameter_loss = 0
        total_sample_loss = 0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, primary_grid, primary_mask, secondary_grid, secondary_mask = batch
            batch_x = x.to(device)
            batch_primary_grid = primary_grid.to(device)
            batch_primary_mask = primary_mask.to(device)
            batch_secondary_grid = secondary_grid.to(device).float()
            batch_secondary_mask = secondary_mask.to(device)

            optimizer.zero_grad()

            global _cur_active
            _cur_active = batch_secondary_mask

            features = sparse_encoder(batch_secondary_grid*batch_secondary_mask)
            sample_output = imputation_decoder(features[::-1])
            parameter_output = subsurface_decoder(features[::-1])

            sample_loss = criterion(sample_output, batch_secondary_grid)
            parameter_loss = criterion(parameter_output, batch_x)

            total_sample_loss += sample_loss.item()
            total_parameter_loss += parameter_loss.item()

            loss = sample_loss + parameter_loss

            loss.backward()
            optimizer.step()

            num_batches += 1

        avg_sample_loss = total_sample_loss / num_batches
        avg_parameter_loss = total_parameter_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Sample Loss: {avg_sample_loss:.10f}, Average Parameter Loss: {avg_parameter_loss:.10f}")

        with torch.no_grad():
            visualize_results(batch_secondary_grid[0], batch_secondary_mask[0], sample_output[0])
            visualize_results(batch_x[0], batch_primary_mask[0], parameter_output[0])

    return sparse_encoder, imputation_decoder, subsurface_decoder
