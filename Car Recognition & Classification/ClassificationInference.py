"""
Filename: ClassificationInference.py
Author: Basel Mohamed Mostafa
Date: 12/16/2025
Description: This file contains the implementation of the Multi-task car classification model for standalone inference.
             The user can either predict a single image, or multiple images.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, resnet50
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import glob
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add safe globals for PyTorch 2.6 compatibility
import torch.serialization
try:
    from numpy.core.multiarray import scalar
    torch.serialization.add_safe_globals([scalar])
except ImportError:
    pass

class MultiTaskCarClassifier(nn.Module):
    """
    Multi-task car classification model for standalone inference.
    """
    def __init__(self, num_makes: int, num_models: int, num_years: int, backbone: str = "efficientnet_b0"):
        super(MultiTaskCarClassifier, self).__init__()
        
        self.num_makes = num_makes
        self.num_models = num_models
        self.num_years = num_years
        self.backbone_name = backbone

        # Initialize backbone
        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=False)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = efficientnet_b0(pretrained=False)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Classification heads
        self.make_head = self._create_head(self.feature_dim, num_makes)
        self.model_head = self._create_head(self.feature_dim, num_models)
        self.year_head = self._create_head(self.feature_dim, num_years)

    def _create_head(self, in_features: int, out_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, out_features)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        return {
            'make': self.make_head(features),
            'model': self.model_head(features),
            'year': self.year_head(features)
        }

class CarClassifierInference:
    """
    Standalone car classifier for inference on single images or folders.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the inference class with a saved model.
        
        Args:
            model_path: Path to the saved .pth model file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_package = self._load_model(model_path)
        self.transform = self._create_transform()
        
    def _load_model(self, model_path: str) -> Dict:
        """Load the complete model package with PyTorch 2.6 compatibility."""
        print(f"🚗 Loading car classifier from: {model_path}")
        
        try:
            # First try with weights_only=False for PyTorch 2.6 compatibility
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"⚠️  Standard load failed, trying alternative method: {e}")
            try:
                # Alternative loading method
                with open(model_path, 'rb') as f:
                    checkpoint = torch.load(f, map_location='cpu')
            except Exception as e2:
                raise RuntimeError(f"Failed to load model: {e2}")
        
        # Reconstruct model
        model = MultiTaskCarClassifier(
            num_makes=checkpoint['model_architecture']['num_makes'],
            num_models=checkpoint['model_architecture']['num_models'],
            num_years=checkpoint['model_architecture']['num_years'],
            backbone=checkpoint['model_architecture']['backbone']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Reconstruct label encoders
        label_encoders = {}
        for task, encoder_data in checkpoint['label_encoders'].items():
            encoder = LabelEncoder()
            if 'classes' in encoder_data and encoder_data['classes']:
                encoder.classes_ = np.array(encoder_data['classes'])
            label_encoders[task] = encoder
        
        print(f"Model loaded successfully!")
        print(f"Architecture: {checkpoint['model_architecture']['backbone']}")
        print(f"Tasks: {checkpoint['model_architecture']['num_makes']} makes, "
              f"{checkpoint['model_architecture']['num_models']} models, "
              f"{checkpoint['model_architecture']['num_years']} years")
        
        return {
            'model': model,
            'label_encoders': label_encoders,
            'label_names': checkpoint['label_names'],
            'preprocessing': checkpoint['preprocessing'],
            'metadata': checkpoint.get('metadata', {})
        }
    
    def _create_transform(self):
        """Create image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Predict car attributes for a single image.
        
        Args:
            image_path: Path to the car image
            
        Returns:
            Dictionary with car attributes and confidence scores
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model_package['model'](input_tensor)
            
            # Get predictions
            make_pred = outputs['make'].argmax(dim=1).cpu().item()
            model_pred = outputs['model'].argmax(dim=1).cpu().item()
            year_pred = outputs['year'].argmax(dim=1).cpu().item()
            
            # Decode to original labels
            make_original = self.model_package['label_encoders']['make'].inverse_transform([make_pred])[0]
            model_original = self.model_package['label_encoders']['model'].inverse_transform([model_pred])[0]
            year_original = self.model_package['label_encoders']['year'].inverse_transform([year_pred])[0]
            
            # Get human-readable names
            make_name = self.model_package['label_names']['make_names'].get(make_original, f"Make_{make_original}")
            model_name = self.model_package['label_names']['model_names'].get(model_original, f"Model_{model_original}")
            
            # Get confidence scores
            make_confidence = torch.nn.functional.softmax(outputs['make'], dim=1)[0][make_pred].item()
            model_confidence = torch.nn.functional.softmax(outputs['model'], dim=1)[0][model_pred].item()
            year_confidence = torch.nn.functional.softmax(outputs['year'], dim=1)[0][year_pred].item()
            
            return {
                'make': {
                    'id': make_original,
                    'name': make_name,
                    'display_name': f"{make_name} - {make_original}",  # ← NEW: Combined display
                    'confidence': make_confidence
                },
                'model': {
                    'id': model_original,
                    'name': model_name,
                    'display_name': f"{model_name} - {model_original}",  # ← NEW: Combined display
                    'confidence': model_confidence
                },
                'year': {
                    'id': year_original,
                    'display_name': f"{year_original}",  # Year doesn't need a name
                    'confidence': year_confidence
                },
                'image_path': image_path,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': f'Failed to process image: {str(e)}',
                'image_path': image_path,
                'success': False
            }
    
    def predict_folder(self, folder_path: str, image_extensions: List[str] = None) -> List[Dict]:
        """
        Predict car attributes for all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            image_extensions: List of image extensions to look for
            
        Returns:
            List of prediction results for each image
        """
        if image_extensions is None:
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        
        # Find all image files
        image_paths = []
        for extension in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, f"**/{extension}"), recursive=True))
            image_paths.extend(glob.glob(os.path.join(folder_path, extension)))
        
        # Remove duplicates and sort
        image_paths = sorted(list(set(image_paths)))
        
        print(f"📁 Found {len(image_paths)} images in folder: {folder_path}")
        
        # Process each image
        results = []
        for image_path in image_paths:
            result = self.predict_single_image(image_path)
            results.append(result)
            
        return results
    
    def visualize_single_prediction(self, image_path: str, result: Dict = None, save_path: Optional[str] = None):
        """
        Visualize prediction for a single image.
        """
        if result is None:
            result = self.predict_single_image(image_path)
        
        if not result.get('success', False):
            print(f"Could not visualize: {result.get('error', 'Unknown error')}")
            return
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot original image
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Plot prediction results
        ax2.axis('off')
        ax2.text(0.1, 0.9, 'Car Classification Results', fontsize=16, fontweight='bold', 
                transform=ax2.transAxes, verticalalignment='top')
        
        # Prediction details - NOW SHOWING BOTH NAME AND NUMBER
        details = [
            f"🏭 Make: {result['make']['display_name']} ({result['make']['confidence']:.1%})",
            f"📱 Model: {result['model']['display_name']} ({result['model']['confidence']:.1%})", 
            f"📅 Year: {result['year']['display_name']} ({result['year']['confidence']:.1%})",
            f"📊 Overall Confidence: {(result['make']['confidence'] + result['model']['confidence'] + result['year']['confidence']) / 3:.1%}"
        ]
        
        for i, detail in enumerate(details):
            ax2.text(0.1, 0.7 - i*0.15, detail, fontsize=12, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Visualization saved to: {save_path}")
        
        plt.show()
        
        # Print enhanced results to console - SHOWING BOTH NAME AND NUMBER
        print("\n" + "="*60)
        print("PREDICTION RESULTS:")
        print("="*60)
        print(f"Make: {result['make']['display_name']} (confidence: {result['make']['confidence']:.3f})")
        print(f"Model: {result['model']['display_name']} (confidence: {result['model']['confidence']:.3f})")
        print(f"Year: {result['year']['display_name']} (confidence: {result['year']['confidence']:.3f})")
        print(f"Overall Confidence: {(result['make']['confidence'] + result['model']['confidence'] + result['year']['confidence']) / 3:.3f}")
        print("="*60)
    
    def visualize_folder_predictions(self, results: List[Dict], max_display: int = 12, save_path: Optional[str] = None):
        """
        Visualize predictions for multiple images in a grid.
        
        Args:
            results: List of prediction results from predict_folder()
            max_display: Maximum number of images to display
            save_path: Optional path to save the visualization
        """
        # Filter successful results
        successful_results = [r for r in results if r.get('success', False)]
        if not successful_results:
            print("No successful predictions to display")
            return
        
        # Limit number of displays
        display_results = successful_results[:max_display]
        
        # Calculate grid size
        n_images = len(display_results)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_images == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, result in enumerate(display_results):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Load and display image
            try:
                image = Image.open(result['image_path']).convert('RGB')
                ax.imshow(image)
                
                # Add prediction info
                title = f"{result['make']['name']} {result['model']['name']}\n{result['year']['id']}"
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Add confidence score
                avg_confidence = (result['make']['confidence'] + result['model']['confidence'] + result['year']['confidence']) / 3
                ax.text(0.02, 0.98, f'Conf: {avg_confidence:.1%}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
                
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading image\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off')
        
        # Hide empty subplots
        for idx in range(len(display_results), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Car Classification Results ({len(display_results)} images)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grid visualization saved to: {save_path}")
        
        plt.show()
        
        # Print summary
        print(f"\nFOLDER PREDICTION SUMMARY:")
        print(f"Successful predictions: {len(successful_results)}")
        print(f"Failed predictions: {len(results) - len(successful_results)}")
        
        if successful_results:
            avg_make_conf = np.mean([r['make']['confidence'] for r in successful_results])
            avg_model_conf = np.mean([r['model']['confidence'] for r in successful_results])
            avg_year_conf = np.mean([r['year']['confidence'] for r in successful_results])
            
            print(f"📈 Average confidence - Make: {avg_make_conf:.3f}, Model: {avg_model_conf:.3f}, Year: {avg_year_conf:.3f}")

# Simple usage examples
def main():
    """
    Main function to demonstrate usage.
    """
    # Initialize classifier
    model_path = r"G:\Work Projects\AI & ML Projects\Image-Classification-and-Object-Detection\Classification Model [NEW]\Model\complete_model.pth"
    
    print("🚗 Initializing Car Classifier...")
    classifier = CarClassifierInference(model_path)
    
    while True:
        print("\n" + "="*60)
        print("CAR CLASSIFIER - CHOOSE AN OPTION")
        print("="*60)
        print("1. Predict single image")
        print("2. Predict all images in folder") 
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter path to image: ").strip()
            if os.path.exists(image_path):
                result = classifier.predict_single_image(image_path)
                classifier.visualize_single_prediction(image_path, result)
            else:
                print("Image path not found!")
                
        elif choice == "2":
            folder_path = input("Enter path to folder: ").strip()
            if os.path.exists(folder_path):
                results = classifier.predict_folder(folder_path)
                classifier.visualize_folder_predictions(results, max_display=12)
            else:
                print("Folder path not found!")
                
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()