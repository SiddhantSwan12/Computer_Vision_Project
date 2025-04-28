import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

class DefectClassifier:
    def __init__(self, num_classes=4, model_type='resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model = self._initialize_model(model_type)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Defect classes
        self.defect_types = {
            0: 'no_defect',
            1: 'scratch',
            2: 'dent',
            3: 'color_variation'
        }
        
        # Severity thresholds
        self.severity_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8
        }

    def _initialize_model(self, model_type):
        if model_type == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_type == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, self.num_classes)
        
        model = model.to(self.device)
        model.eval()
        return model

    def detect_defects(self, roi):
        """
        Detect and classify defects in a region of interest.
        
        Args:
            roi: Region of interest (cropped image around object)
            
        Returns:
            Dictionary containing defect information
        """
        try:
            # Preprocess image
            image_tensor = self.transform(roi).unsqueeze(0).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
            # Get the most likely defect and its probability
            defect_type_idx = torch.argmax(probabilities).item()
            confidence = probabilities[defect_type_idx].item()
            
            # Determine severity based on confidence
            severity = 'low'
            for level, threshold in self.severity_thresholds.items():
                if confidence > threshold:
                    severity = level
            
            return {
                'defect_type': self.defect_types[defect_type_idx],
                'confidence': confidence,
                'severity': severity,
                'probabilities': {
                    defect: prob.item() 
                    for defect, prob in zip(self.defect_types.values(), probabilities)
                }
            }
            
        except Exception as e:
            print(f"Error in defect detection: {str(e)}")
            return None
