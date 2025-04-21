from flask import Flask, render_template, request
import cv2
from PIL import Image
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import numpy as np

app = Flask(__name__)

# Define class names
names = ['baseball-diamond', 'basketball-court', 'bridge', 'ground-track-field', 'harbor', 'helicopter',
         'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field',
         'storage-tank', 'swimming-pool', 'tennis-court']

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Retrieve uploaded images
    img1 = request.files['image1']
    img2 = request.files['image2']

    # Convert images to PIL format and save to static directory
    img1_pil = Image.open(img1.stream)
    img2_pil = Image.open(img2.stream)
    img1_pil.save('static/img1.jpg')
    img2_pil.save('static/img2.jpg')
    
    # Process both images and save output
    results1 = process('static/img1.jpg', 'out1')
    results2 = process('static/img2.jpg', 'out2')

    # Get class-wise counts for both images
    classes_1 = get_class_counts(results1)
    classes_2 = get_class_counts(results2)

    # Detect changes between images
    isChanged = detectChange(classes_1, classes_2, names)
    changeDict = {name: (classes_1.get(name, 0), classes_2.get(name, 0))
                  for name in names if classes_1.get(name, 0) != 0 or classes_2.get(name, 0) != 0}

    # Render template with results
    return render_template('predict.html', path1='static/out1.jpg', path2='static/out2.jpg',
                           changeDict=changeDict, isChanged=isChanged, classes=names)

def calculate_dynamic_eps(detections):
    """
    Calculates a dynamic eps value based on the average distance between detection centroids.
    """
    if len(detections) < 2:
        return 50  # Fallback value

    # Calculate centroids of bounding boxes
    centroids = np.array([
        [(detection['box'][0] + detection['box'][2]) / 2, (detection['box'][1] + detection['box'][3]) / 2]
        for detection in detections
    ])

    # Compute pairwise distances
    distances = np.sqrt(np.sum((centroids[:, None] - centroids[None, :]) ** 2, axis=-1))
    average_distance = np.mean(distances[distances > 0])
    return max(average_distance * 0.5, 20)  # Minimum eps = 20

def cluster_detections(detections, eps):
    """
    Clusters detections based on proximity using DBSCAN.
    """
    if not detections:
        return []

    centroids = np.array([
        [(detection['box'][0] + detection['box'][2]) / 2, (detection['box'][1] + detection['box'][3]) / 2]
        for detection in detections
    ])

    clustering = DBSCAN(eps=eps, min_samples=1).fit(centroids)

    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label == -1:  # Skip noise points
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(detections[i])

    return clusters

def calculate_enclosing_box(cluster):
    """
    Calculates the enclosing bounding box for a cluster.
    """
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')

    for detection in cluster:
        box = detection['box']
        x_min = min(x_min, box[0])
        y_min = min(y_min, box[1])
        x_max = max(x_max, box[2])
        y_max = max(y_max, box[3])

    return [x_min, y_min, x_max, y_max]

def process(image_path, output):
    model = YOLO("best1.pt")  # Load YOLO model
    results = model.predict(image_path, imgsz=640)

    # Organize detections by class
    detections_by_class = {name: [] for name in names}
    for detection in results[0].boxes:
        bbox = detection.xyxy[0].tolist()
        confidence = detection.conf[0]
        class_id = int(detection.cls[0])
        class_name = names[class_id]

        # Add detection to its corresponding class
        detections_by_class[class_name].append({
            'box': bbox,
            'class': class_name,
            'confidence': confidence
        })

    image = cv2.imread(image_path)

    # Process clusters for each class
    for class_name, detections in detections_by_class.items():
        if len(detections) == 0:
            continue
        eps = calculate_dynamic_eps(detections)
        clusters = cluster_detections(detections, eps=eps)

        for cluster_label, cluster in clusters.items():
            enclosing_box = calculate_enclosing_box(cluster)
            count = len(cluster)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, enclosing_box)
            color = (255, 0, 0)  # All classes have red bounding boxes for simplicity
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Add text
            text = f"{class_name}: {count}"
            cv2.putText(image, text, (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    # Save the processed image
    cv2.imwrite(f'static/{output}.jpg', image)
    return results

def get_class_counts(results):
    """
    Returns a dictionary of class counts from YOLO results.
    """
    class_counts = {}
    for detection in results[0].boxes:
        class_id = int(detection.cls[0])
        class_name = names[class_id]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

def detectChange(class1, class2, names):
    """
    Detects changes between two sets of class counts.
    """
    for name in names:
        if class1.get(name, 0) != class2.get(name, 0):
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)