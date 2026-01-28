# Deep learning-assisted Cytological Image Analysis for Canine Lymphoma
This repository is for the materials related to the paper published: "Deep learning assisted cytological image analysis for canine lymphoma (2026, Veterinary Oncology)."
This repository provides deep learning models based on the **ResNet50** architecture for two specialized tasks in veterinary clinical pathology: comprehensive canine lymphoma analysis, including differential diagnosis between lymphoma and reactive lymphoid hyperplasia, and immunophenotype subtyping.

---

## 🔗 Model Hub (Hugging Face)

Access the pre-trained weights for each diagnostic stage below:

1. **LSA vs RLN Classification** (Diagnosis)
   [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LSA--RLN-yellow)](https://huggingface.co/hazel070720/Canine-LSA-RLN_ResNet50-model)
2. **T-cell vs B-cell Classification** (Subtyping)
   [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-T--B--cell-blue)](https://huggingface.co/hazel070720/Canine-Tcell-Bcell_ResNet50-model)

---

## 📌 Project Overview
The diagnostic workflow is divided into two major tasks:
* **Task 1: Diagnosis** – Distinguishing between **Lymphoma (LSA)** and **Reactive Lymph Node (RLN)**.
* **Task 2: Subtyping** – Classifying the lymphoma type into **T-cell** or **B-cell** lineages.

---

## 📥 Model Weights
| Task | Target Classes | Repository Link |
| :--- | :--- | :--- |
| **Diagnosis** | LSA vs RLN | [LSA-RLN Model](https://huggingface.co/hazel070720/Canine-LSA-RLN_ResNet50-model) |
| **Subtyping** | T-cell vs B-cell | [T-B cell Model](https://huggingface.co/hazel070720/Canine-Tcell-Bcell_ResNet50-model) |

* **Architecture:** ResNet50
* **Format:** PyTorch State Dictionary (`.pth`)

---

## 🚀 Quick Start
Ensure you have the required libraries installed:
`pip install torch torchvision huggingface_hub`

```python
import torch
import torchvision.models as models
from huggingface_hub import hf_hub_download

def load_canine_model(model_type="LSA_RLN", fold=1):
    """
    Load a specific fold from the 10-fold cross-validation weights.
    """
    # Initialize ResNet50
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    
    # Select repository
    if model_type == "LSA_RLN":
        repo_id = "hazel070720/Canine-LSA-RLN_ResNet50-model"
    else:
        repo_id = "hazel070720/Canine-Tcell-Bcell_ResNet50-model"

    # Construct the filename based on the fold number
    # TODO: Ensure this matches your actual file naming convention (e.g., 'fold1.pth')
    filename = f"fold{fold}_best.pth" 

    try:
        # Download and load weights
        print(f"Downloading {filename} from {repo_id}...")
        weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print(f"Successfully loaded Fold {fold} weights.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
