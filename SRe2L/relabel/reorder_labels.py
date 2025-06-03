import argparse
import os
import torch
import numpy as np
from utils_fkd import get_FKD_info, get_predicted_label, calculate_entropy # Assuming this function can be reused or adapted

def load_batch_config(epoch_path, batch_idx):
    batch_config_path = os.path.join(epoch_path, f'batch_{batch_idx}.tar')
    # batch_config: [coords_status, flip_status, mix_index, mix_lam, mix_bbox, output.cpu(), target.cpu(), entropy.cpu()]
    return torch.load(batch_config_path)

def main():
    parser = argparse.ArgumentParser(description='Reorder soft labels based on entropy and correctness')
    parser.add_argument('--fkd-path', type=str, required=True, help='Path to the saved FKD soft labels (e.g., ./FKD_soft_label)')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the reordered labels')
    parser.add_argument('--num-classes', type=int, required=True, help='Number of classes in the dataset') # Required for one-hot encoding of mixed labels

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    all_data = [] # To store tuples of (entropy, index, soft_label, augmentation_recipe, original_target)

    # Assuming FKD info gives us epoch count and other necessary details
    # Adapt get_FKD_info or implement new logic if it's not suitable
    max_epoch, batch_size, num_img = get_FKD_info(args.fkd_path) 
    # Note: get_FKD_info might need adjustment if it doesn't return batch_list or if its definition of num_img/batch_size isn't directly usable here.
    # For simplicity, we'll iterate through epochs and batches assuming a structure.

    print(f"Processing data from: {args.fkd_path}")

    for epoch_num in range(max_epoch): # Iterate through all epochs
        epoch_path = os.path.join(args.fkd_path, f'epoch_{epoch_num}')
        if not os.path.isdir(epoch_path):
            print(f"Epoch {epoch_num} not found, skipping.")
            continue
        
        print(f"Processing epoch {epoch_num}...")
        # Determine number of batches. This might require listing files or using info from get_FKD_info
        batch_files = [f for f in os.listdir(epoch_path) if f.startswith('batch_') and f.endswith('.tar')]
        num_batches = len(batch_files)

        for batch_idx in range(num_batches): # Iterate through all batches in an epoch
            try:
                # batch_config: [coords_status, flip_status, mix_index, mix_lam, mix_bbox, soft_label_batch, target_batch, entropy_batch]
                batch_data = load_batch_config(epoch_path, batch_idx)
            except FileNotFoundError:
                print(f"Batch file batch_{batch_idx}.tar not found in {epoch_path}, skipping.")
                continue

            coords_batch, flip_batch, mix_index_batch, mix_lam_batch, mix_bbox_batch, soft_label_batch, target_batch, entropy_batch, original_indices_batch = batch_data
            
            # Iterate over each sample in the batch
            for i in range(soft_label_batch.size(0)):
                soft_label = soft_label_batch[i]
                entropy = entropy_batch[i]
                original_target = target_batch[i] # This is the original, single label before any mixup/cutmix
                
                # Augmentation details for this sample
                mix_index = mix_index_batch[i] if mix_index_batch is not None else None
                mix_lam = mix_lam_batch[i] if mix_lam_batch is not None else None
                mix_bbox = mix_bbox_batch[i] if mix_bbox_batch is not None else None # Assuming mix_bbox_batch could be None or a list/tensor of bboxes

                predicted_label = get_predicted_label(soft_label)

                is_correct = False
                if mix_index is None: # Not augmented
                    is_correct = (predicted_label == original_target)
                else: # Augmented (mixup or cutmix)
                    # For augmented data, the 'target' is effectively a mix of original_target and target[mix_index]
                    # The ground truth for augmented sample would be a one-hot encoded vector or similar
                    # representing the two mixed classes.
                    # target_at_mix_index is the label of the image mixed with.
                    # This requires access to the original label of the image at mix_index.
                    # This part needs careful handling of how targets for mixed samples are defined and accessed.
                    # For now, let's assume original_target is the label of the primary image in the mix.
                    # And we'd need the label of the image it was mixed with.
                    # This information (label of image at mix_index) is not directly in batch_data.
                    # We need a way to get target_at_mix_index.
                    # Simplified: if predicted is one of the two mixed labels.
                    # This is a placeholder for the more complex logic for mixed labels.
                    # We'll assume for now that 'target_batch' contains the labels of all images in the dataset,
                    # and mix_index refers to an index within this comprehensive list.
                    # This is a simplification and might need a more robust way to get the mixed target.
                    
                    # Placeholder: This logic needs to be refined based on how `target_batch` and `mix_index_batch` store information.
                    # Assuming target_batch[mix_index] gives the label of the mixed image.
                    # This is a strong assumption. The actual implementation may need to load all targets or have a mapping.
                    
                    # For CutMix/MixUp, the "true" label is a combination of two labels.
                    # If the model predicts one of these two labels, we consider it "correct" for this filtering step.
                    # This part of the logic needs to be robust.
                    # We need the label of the image that was mixed in.
                    # Let's assume `target_batch` contains the targets for the current batch,
                    # and `mix_index_batch` contains indices *within this batch* for the mix.
                    
                    # The 'mix_index' variable (from mix_index_batch[i]) is now a GLOBAL dataset index
                    # and is a 0-dim tensor at this point (not None).
                    global_mix_index_val = mix_index.item()

                    # original_indices_batch is a tensor of global indices for the current batch samples.
                    # target_batch contains labels for the current batch samples.
                    original_indices_list = original_indices_batch.tolist() # Assumes original_indices_batch is CPU tensor

                    if global_mix_index_val in original_indices_list:
                        local_idx_for_mixed_image = original_indices_list.index(global_mix_index_val)
                        target_of_mixed_image = target_batch[local_idx_for_mixed_image] # Assumes target_batch is CPU
                        is_correct = (predicted_label == original_target) or (predicted_label == target_of_mixed_image)
                    else:
                        print(f"Warning: For sample with original_target {original_target.item()} (global id {original_indices_batch[i].item()}), " +
                              f"its mixed component (global_mix_index {global_mix_index_val}) " +
                              "was not found in the current batch's original_indices_batch. " +
                              "Correctness for mixed component cannot be verified. Treating as incorrect.")
                        is_correct = False


                if not is_correct:
                    continue # Discard uncorrected case

                # Store data for sorting
                # The 'index' here should be a global index if possible, or unique identifier.
                # For now, using a composite index (epoch, batch, i within batch)
                # This needs to be a unique dataset index later on.
                current_original_index = original_indices_batch[i].item()
                
                aug_recipe = {
                    "coords": coords_batch[i].tolist() if coords_batch is not None else None,
                    "flip": flip_batch[i].item() if flip_batch is not None else None,
                    "mix_index": mix_index.item() if mix_index is not None else None,
                    "mix_lam": mix_lam.item() if isinstance(mix_lam, torch.Tensor) else mix_lam, # mix_lam can be float
                    "mix_bbox": mix_bbox # mix_bbox can be None or list
                }

                all_data.append({
                    "entropy": entropy.item(),
                    "dataset_index": current_original_index, # Use the actual original index
                    "soft_label": soft_label.tolist(),
                    "augmentation_recipe": aug_recipe,
                    "original_target": original_target.item()
                })

    print(f"Collected {len(all_data)} samples after filtering.")

    # Reorder based on entropy (easy-to-hard)
    all_data.sort(key=lambda x: x['entropy'])

    # Save the reordered data
    output_file_path = os.path.join(args.output_path, 'reordered_labels.pt') # Saving as a single PyTorch tensor file
    torch.save(all_data, output_file_path)
    print(f"Reordered labels saved to {output_file_path}")

if __name__ == '__main__':
    main()
