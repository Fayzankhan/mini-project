"""
Advanced Explainability Features using SHAP and LIME
Provides better model interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries

def generate_shap_explanations(model, image, background_images=None, max_evals=100):
    """
    Generate SHAP explanations for model predictions.
    
    Args:
        model: Trained Keras model
        image: Input image to explain (preprocessed)
        background_images: Background dataset for SHAP (optional)
        max_evals: Maximum evaluations for SHAP
    
    Returns:
        shap_values: SHAP values
        explanation_image: Visualization
    """
    try:
        # Create SHAP explainer
        if background_images is None:
            # Use DeepExplainer for deep learning models
            explainer = shap.DeepExplainer(model, image[np.newaxis, ...])
            shap_values = explainer.shap_values(image[np.newaxis, ...])
        else:
            # Use KernelExplainer for general models
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x),
                background_images[:50]  # Use subset for speed
            )
            shap_values = explainer.shap_values(
                image[np.newaxis, ...],
                nsamples=max_evals
            )
        
        # Visualize SHAP values
        shap_image = shap.image_plot(shap_values, image[np.newaxis, ...], show=False)
        
        return shap_values, shap_image
        
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        return None, None

def generate_lime_explanation(model, image, num_features=10, num_samples=1000):
    """
    Generate LIME explanations for model predictions.
    
    Args:
        model: Trained Keras model
        image: Input image to explain (original, not preprocessed)
        num_features: Number of features to highlight
        num_samples: Number of samples for LIME
    
    Returns:
        explanation: LIME explanation object
        explanation_image: Visualization
    """
    try:
        # Create LIME explainer
        explainer = lime_image.LimeImageExplainer()
        
        # Define prediction function
        def predict_fn(images):
            # Preprocess images for model
            processed = []
            for img in images:
                img_resized = cv2.resize(img, (224, 224))
                img_normalized = img_resized / 255.0
                processed.append(img_normalized)
            processed = np.array(processed)
            return model.predict(processed)
        
        # Generate explanation
        explanation = explainer.explain_instance(
            image.astype('double'),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get explanation image
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False
        )
        
        explanation_image = mark_boundaries(temp / 255.0, mask)
        
        return explanation, explanation_image
        
    except Exception as e:
        print(f"Error generating LIME explanation: {e}")
        return None, None

def create_combined_explanation(shap_values, lime_explanation, original_image):
    """
    Combine SHAP and LIME explanations for comprehensive understanding.
    
    Args:
        shap_values: SHAP values
        lime_explanation: LIME explanation
        original_image: Original X-ray image
    
    Returns:
        combined_visualization: Combined explanation image
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original X-ray')
    axes[0].axis('off')
    
    # SHAP visualization
    if shap_values is not None:
        axes[1].imshow(shap_values[0] if isinstance(shap_values, list) else shap_values)
        axes[1].set_title('SHAP Explanation')
        axes[1].axis('off')
    
    # LIME visualization
    if lime_explanation is not None:
        temp, mask = lime_explanation.get_image_and_mask(
            lime_explanation.top_labels[0],
            positive_only=True,
            num_features=10,
            hide_rest=False
        )
        axes[2].imshow(mark_boundaries(temp / 255.0, mask))
        axes[2].set_title('LIME Explanation')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def generate_feature_importance(model, image, top_n=10):
    """
    Generate feature importance scores for different image regions.
    
    Args:
        model: Trained model
        image: Input image
        top_n: Number of top features to return
    
    Returns:
        feature_importance: Dictionary with region importance scores
    """
    # Divide image into regions
    h, w = image.shape[:2]
    region_size = 32
    
    importance_scores = {}
    
    for i in range(0, h, region_size):
        for j in range(0, w, region_size):
            # Create masked image
            masked_image = image.copy()
            masked_image[i:i+region_size, j:j+region_size] = 0
            
            # Get prediction difference
            original_pred = model.predict(image[np.newaxis, ...], verbose=0)[0][0]
            masked_pred = model.predict(masked_image[np.newaxis, ...], verbose=0)[0][0]
            
            # Importance = difference in prediction
            importance = abs(original_pred - masked_pred)
            
            region_key = f"Region_{i//region_size}_{j//region_size}"
            importance_scores[region_key] = {
                'importance': float(importance),
                'coordinates': (i, j, i+region_size, j+region_size),
                'impact': 'positive' if masked_pred < original_pred else 'negative'
            }
    
    # Sort by importance
    sorted_regions = sorted(
        importance_scores.items(),
        key=lambda x: x[1]['importance'],
        reverse=True
    )[:top_n]
    
    return dict(sorted_regions)

def explain_prediction(model, image, method='shap'):
    """
    Main function to explain model predictions.
    
    Args:
        model: Trained model
        image: Input image
        method: 'shap', 'lime', or 'both'
    
    Returns:
        explanation: Explanation results
    """
    results = {
        'method': method,
        'shap': None,
        'lime': None,
        'feature_importance': None
    }
    
    if method in ['shap', 'both']:
        shap_values, shap_img = generate_shap_explanations(model, image)
        results['shap'] = {
            'values': shap_values,
            'image': shap_img
        }
    
    if method in ['lime', 'both']:
        lime_exp, lime_img = generate_lime_explanation(model, image)
        results['lime'] = {
            'explanation': lime_exp,
            'image': lime_img
        }
    
    # Always generate feature importance
    results['feature_importance'] = generate_feature_importance(model, image)
    
    return results

