import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # You'll need a custom Dataset class
import yaml
import os
import time
from tqdm import tqdm

# from model.detection.detect_model import AEMangoYOLODetector
# from model.detection.utils.loss import ComputeLoss # Standard YOLOv5 loss function
# from preprocessing.patch_extraction.patch_loader import MangoPatchDataset # Example name
# from utils.general import ... # For logging, metrics, etc.

# --- Placeholder for Dataset Class ---
class MangoPatchDataset(Dataset):
    def __init__(self, patch_dir, annotation_dir, image_size, augment=False):
        # TODO: Load patch filenames and corresponding YOLO annotations
        # Annotations need to be mapped to patch coordinates.
        self.patch_files = [] # List of patch file paths
        self.labels = []      # List of corresponding labels (bboxes, class) for each patch
        self.image_size = image_size # e.g., (120, 120)
        self.augment = augment
        # This is complex: requires finding original image annotations,
        # checking which fall into a patch, and converting coords.
        print("WARNING: MangoPatchDataset needs full implementation for loading patches and their specific annotations.")
        # For now, create dummy data
        num_dummy_patches = 100
        self.patch_files = [f"dummy_patch_{i}.png" for i in range(num_dummy_patches)]
        # Labels: per patch, a tensor of [cls, cx, cy, w, h] (normalized for the patch)
        # Example: one mango per patch for simplicity
        self.labels = [torch.tensor([[0, 0.5, 0.5, 0.3, 0.4]]) for _ in range(num_dummy_patches)]


    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        # TODO: Load patch image, apply augmentations
        patch_path = self.patch_files[idx]
        # img = cv2.imread(patch_path) # Read patch
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.image_size[1], self.image_size[0])) # W, H
        # img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        img = torch.randn(3, self.image_size[0], self.image_size[1]) # Dummy image
        
        label = self.labels[idx] # Format: (num_objects_in_patch, 5) [cls, cx, cy, w, h]
        
        # For YOLO loss, targets are usually formatted as (num_objects, 6) [img_idx_in_batch, cls, cx, cy, w, h]
        # The dataloader's collate_fn usually handles adding img_idx_in_batch.
        targets = torch.zeros((len(label), 6))
        if len(label) > 0:
            targets[:, 1:] = label

        return img, targets

# --- Placeholder for YOLOv5 Loss ---
# You would typically adapt this from an existing YOLOv5 implementation.
class ComputeLoss:
    def __init__(self, model, autobalance=False):
        # device = next(model.parameters()).device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # simplified
        
        # Define criteria
        self.BCEcls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.BCEobj = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))

        # Hyperparameters (example from YOLOv5)
        self.hyp = {'box': 0.05, 'obj': 1.0, 'cls': 0.5, 
                    'anchor_t': 4.0, # anchor-multiple threshold
                    'gr': 1.0} # iou loss ratio (obj_iou/box_iou)

        self.na = model.detect_head.na  # number of anchors
        self.nc = model.detect_head.nc  # number of classes
        self.nl = model.detect_head.nl  # number of layers
        self.anchors = model.detect_head.anchors.to(self.device) # (nl, na, 2)
        self.stride = model.detect_head.stride.to(self.device)


    def __call__(self, p, targets):  # p: list of raw outputs, targets: [img_idx, class, x, y, w, h]
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        tcls, tbox, indices, anch = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # Preds for matched targets
                ps = pi[b, a, gj, gi]  # (n_targets_in_layer, n_outputs_per_anchor)

                # Regression loss (GIoU loss example)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                
                # Use bbox_iou from unified_diou.py for GIoU or other IoU variants
                # For simplicity, placeholder for IoU calculation for loss
                # iou = bbox_iou(pbox.T, tbox[i], xywh=True, CIoU=True) # Or UDIoU if using it in loss
                # For simplicity, assume a simple L1/L2 or a dummy IoU loss here.
                # This is a critical part that needs proper implementation based on YOLOv5 standard losses.
                # lbox += (1.0 - iou).mean() # iou loss (placeholder)
                lbox_dummy = torch.nn.functional.smooth_l1_loss(pbox, tbox[i], reduction='mean')
                lbox += lbox_dummy


                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.hyp['gr']) + self.hyp['gr'] * torch.rand(n, device=self.device) # Use actual IoU for obj target

                # Classification
                if self.nc > 1:  # cls loss only if multiple classes
                    t = torch.full_like(ps[:, 5:], 0, device=self.device)  # targets
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

            lobj += self.BCEobj(pi[..., 4], tobj)  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same, (na, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices, (na, nt, 7)

        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g  # offsets

        for i in range(self.nl): # For each detection layer/scale
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain (grid H, grid W)

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


def train_one_epoch(model, optimizer, train_loader, loss_fn, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} Training")
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        # Targets need to be formatted correctly for loss_fn
        # Typically, [img_idx_in_batch, class, cx, cy, w, h]
        # Collate_fn in DataLoader should handle this.
        # Here, targets from MangoPatchDataset are already (N_obj, 6) but img_idx is 0.
        # Need to set correct img_idx for the batch.
        
        # Prepare targets by adding batch image index
        batched_targets_list = []
        for i in range(images.shape[0]): # Iterate through images in batch
            # Find targets for this specific image (if targets were not pre-batched with img_idx)
            # In our dummy dataset, each item is one image and its targets.
            # So, targets[i] would be the targets for images[i]
            # If dataloader collate_fn handles it, targets is already [img_idx, cls, x,y,w,h]
            
            # Assuming targets come from dataloader as a list of tensors (one per image)
            # or a single tensor where first col is img_idx.
            # Our dummy MangoPatchDataset provides targets for a single image.
            # The default collate_fn will stack them. We need to add img_idx.
            
            # This part depends heavily on the Dataset and Dataloader's collate_fn.
            # For simplicity, assume `targets` is already in the correct global format
            # [img_idx_in_batch, class_id, cx, cy, w, h]
            pass # This logic needs to be correct based on your Dataset
            
        targets = targets.to(device) # If targets is a single tensor for the batch

        optimizer.zero_grad()
        predictions = model(images) # Raw outputs from Detect head (list of tensors)
        
        loss, loss_items = loss_fn(predictions, targets) # loss_items: [lbox, lobj, lcls, total_loss]
        
        if not torch.isfinite(loss):
            print(f"WARNING: non-finite loss at epoch {epoch}, batch {batch_idx}. Skipping update.")
            print(f"Loss items: {loss_items}")
            # Potentially log inputs/outputs or skip batch
            continue

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item(), 'lbox': loss_items[0].item(), 
                          'lobj': loss_items[1].item(), 'lcls': loss_items[2].item()})
        
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{total_epochs} - Training Avg Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    # --- Load Configs ---
    with open("configs/dataset.yaml", 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    with open("configs/model.yaml", 'r') as f:
        model_cfg = yaml.safe_load(f) # Model specific params like anchors, nc
    with open("configs/training.yaml", 'r') as f:
        training_cfg = yaml.safe_load(f)

    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(training_cfg['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(training_cfg['seed'])

    # --- Initialize Model ---
    # model = AEMangoYOLODetector(cfg_model_yaml_path="configs/model.yaml", 
    #                            num_classes_override=model_cfg.get('num_classes', 1))
    from model.detection.detect_model import AEMangoYOLODetector # Import here to avoid circular if classes are defined above
    model = AEMangoYOLODetector(cfg_model_yaml_path="configs/model.yaml", num_classes_override=1) # Hardcode nc=1 for mango
    model.to(device)

    # --- Optimizer and Scheduler ---
    if training_cfg['detection_optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=training_cfg['detection_learning_rate'], weight_decay=0.0005)
    elif training_cfg['detection_optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=training_cfg['detection_learning_rate'], momentum=0.937, weight_decay=0.0005, nesterov=True)
    else:
        raise ValueError(f"Unsupported optimizer: {training_cfg['detection_optimizer']}")

    scheduler = None
    if training_cfg.get('detection_lr_scheduler'):
        if training_cfg['detection_lr_scheduler'].lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg['detection_epochs'])
        elif training_cfg['detection_lr_scheduler'].lower() == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) # Example params


    # --- Dataset and DataLoader ---
    # TODO: Replace with actual dataset loading. Patch paths and annotations for patches.
    patch_img_size = (dataset_cfg['patch_size'], dataset_cfg['patch_size'])
    train_dataset = MangoPatchDataset(
        patch_dir=dataset_cfg['patch_dir'] + "/train", # Assuming train/val split for patches
        annotation_dir=dataset_cfg['annotation_dir'] + "/train_patches_annots", # Annotations specific to patches
        image_size=patch_img_size,
        augment=True
    )
    # val_dataset = MangoPatchDataset(...) 
    
    # Collate function for DataLoader to handle targets correctly
    def yolo_collate_fn(batch):
        images, labels = zip(*batch)
        # Stack images
        images = torch.stack(images, 0)
        # Process labels: add batch image index to each target
        for i, lbl in enumerate(labels):
            lbl[:, 0] = i  # Assuming lbl is (num_obj, 6) and first col is for img_idx
        return images, torch.cat(labels, 0)


    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_cfg['detection_batch_size'], 
        shuffle=True, 
        num_workers=4, # Adjust based on your system
        pin_memory=True,
        collate_fn=yolo_collate_fn # Important for target formatting
    )
    # val_loader = DataLoader(val_dataset, ...)


    # --- Loss Function ---
    loss_function = ComputeLoss(model)

    # --- Training Loop ---
    print("Starting detection model training...")
    os.makedirs(training_cfg['detection_checkpoint_dir'], exist_ok=True)
    best_val_loss = float('inf') # Or best mAP

    for epoch in range(training_cfg['detection_epochs']):
        epoch_start_time = time.time()
        
        avg_train_loss = train_one_epoch(model, optimizer, train_loader, loss_function, device, epoch, training_cfg['detection_epochs'])
        
        if scheduler:
            scheduler.step()

        # --- Validation Step (TODO) ---
        # model.eval()
        # avg_val_loss = 0
        # with torch.no_grad():
        #     for images, targets in val_loader:
        #         # ... calculate validation loss or mAP ...
        # print(f"Epoch {epoch+1} - Validation Avg Loss: {avg_val_loss:.4f}")
        
        # --- Save Checkpoint ---
        # if avg_val_loss < best_val_loss: # Or if mAP improved
        #     best_val_loss = avg_val_loss
        checkpoint_path = os.path.join(training_cfg['detection_checkpoint_dir'], f"ae_mangoyolo5_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss, # Or val_loss
            # 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} duration: {epoch_duration:.2f}s")

    print("Detection model training finished.")

if __name__ == '__main__':
    # Create dummy config files if they don't exist for a quick test run
    # This is very basic and assumes configs are in place.
    # You would run this script from the root of your project.
    if not os.path.exists("configs/dataset.yaml"):
        print("WARNING: configs/dataset.yaml not found. main() might fail.")
    if not os.path.exists("configs/model.yaml"):
        print("WARNING: configs/model.yaml not found. main() might fail.")
    if not os.path.exists("configs/training.yaml"):
        print("WARNING: configs/training.yaml not found. main() might fail.")
    
    # To make it runnable without full data, the MangoPatchDataset and ComputeLoss
    # are heavily simplified.
    print("NOTE: This is a simplified training script. Full data loading and loss calculation are complex.")
    
    # Create dummy directories for dataset and checkpoints if needed for a dry run
    # Make sure paths in YAMLs are valid or adjust here for testing
    # For example, if dataset_cfg['patch_dir'] is "data/patches", create "data/patches/train"
    # and dataset_cfg['annotation_dir'] is "data/annotations", create "data/annotations/train_patches_annots"
    # os.makedirs("data/patches/train", exist_ok=True)
    # os.makedirs("data/annotations/train_patches_annots", exist_ok=True)
    # os.makedirs("outputs/checkpoints/detection/", exist_ok=True)
    
    try:
        main()
    except FileNotFoundError as e:
        print(f"Missing config file: {e}. Please ensure all YAML configs are present.")
    except Exception as e:
        print(f"An error occurred during training script execution: {e}")
        import traceback
        traceback.print_exc()