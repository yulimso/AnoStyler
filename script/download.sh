# Download MVTec-AD dataset
mkdir -p datasets/MVTec-AD && cd datasets/MVTec-AD
wget <download_link_to_MVTec_AD_dataset>
cd -

# Download VisA dataset
mkdir -p datasets/VisA && cd datasets/VisA
wget <download_link_to_VisA_dataset>
cd -

# Download SAM model checkpoint
mkdir -p checkpoints && cd checkpoints
wget <download_link_to_sam_checkpoint>
cd -