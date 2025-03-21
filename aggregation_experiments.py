import torch
from typing import List
import os

def simple_average(model_paths: List[str], output_path: str, scale_factor: float = 1.0):
    """
    Averages multiple PyTorch models saved as .pth files and saves the result.
    The first model can be given more influence using the scale_factor.
    
    Args:
        model_paths: List of paths to the model .pth files
        output_path: Path where the averaged model will be saved
        scale_factor: Factor to scale the weights of the first model (default: 1.0)
    
    Returns:
        The averaged model or state dict
    """
    # Check if all files exist
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
    
    # Load all models and their state dictionaries
    state_dicts = []
    first_checkpoint = None
    
    for i, path in enumerate(model_paths):
        # Load the saved file
        checkpoint = torch.load(path)
        
        # Save the structure of the first checkpoint
        if i == 0:
            first_checkpoint = checkpoint
        
        # Check if the loaded object is already a state_dict or a model
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # It's a checkpoint with a state_dict key
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict):
            # It's already a state dictionary
            state_dict = checkpoint
        else:
            # It's a model object
            try:
                state_dict = checkpoint.state_dict()
            except AttributeError:
                raise TypeError(f"Could not extract state_dict from {path}. "
                               f"The loaded object is of type {type(checkpoint)}")
        
        state_dicts.append(state_dict)
    
    # Initialize averaged weights with zeros like the first state dict
    averaged_weights = {}
    for key in state_dicts[0].keys():
        averaged_weights[key] = torch.zeros_like(state_dicts[0][key])
    
    # Number of models for averaging
    effective_num_models = len(state_dicts)
    
    # Sum all parameters with scaling for the first model
    for i, state_dict in enumerate(state_dicts):
        for key in averaged_weights.keys():
            if key in state_dict:
                if i == 0:
                    # For the first model, convert the scaling factor to the appropriate tensor type
                    weight_tensor = torch.tensor(scale_factor, dtype=state_dict[key].dtype, device=state_dict[key].device)
                    averaged_weights[key] += state_dict[key] * weight_tensor
                else:
                    averaged_weights[key] += state_dict[key]
            else:
                raise KeyError(f"Key {key} not found in one of the models. Models might have different architectures.")
    
    # Divide by the effective number of models to get the weighted average
    for key in averaged_weights.keys():
        # Convert the division factor to match the tensor dtype
        div_tensor = torch.tensor(effective_num_models, dtype=averaged_weights[key].dtype, device=averaged_weights[key].device)
        averaged_weights[key] = averaged_weights[key] / div_tensor
    
    # Save the averaged model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Maintain the same structure as the original checkpoint
    if isinstance(first_checkpoint, dict) and 'state_dict' in first_checkpoint:
        # Create a copy of the original checkpoint and replace the state_dict
        output_checkpoint = first_checkpoint.copy()
        output_checkpoint['state_dict'] = averaged_weights
        torch.save(output_checkpoint, output_path)
        return output_checkpoint
    elif hasattr(first_checkpoint, 'load_state_dict'):
        # If we have a model object, load the averaged weights and save the model
        first_checkpoint.load_state_dict(averaged_weights)
        torch.save(first_checkpoint, output_path)
        return first_checkpoint
    else:
        # If we only have state dictionaries and no nested structure, save directly
        torch.save(averaged_weights, output_path)
        return averaged_weights

if __name__ == "__main__":
    # Define paths
    model_paths = [
        "output/cifar10/stl10_backdoored_encoder/model_20.pth", # The first model will have scale_factor * scale_factor influence
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth",
        "output/cifar10/clean_encoder/model_100.pth"
    ]
    output_path = "output/cifar10/stl10_backdoored_encoder/aggregated/average_1clean_1poisonscale1.pth"
    
    # Scale factor to increase the influence of the first model
    scale_factor = 3.1  # First model weights will be scaled of this factor
    
    # Perform model averaging
    try:
        averaged_model = simple_average(model_paths, output_path, scale_factor)
        print(f"Models successfully averaged with scale factor {scale_factor} for the first model")
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error during model averaging: {str(e)}")