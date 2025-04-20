from flask import Flask, render_template, request
# import cv2
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
# names = ['baseball-diamond', 'basketball-court', 'bridge', 'container-crane', 'ground-track-field', 'harbor', 'helicopter', 'large-vehicle', 'plane', 'roundabout', 'ship', 'small-vehicle', 'soccer-ball-field', 'storage-tank', 'swimming-pool', 'tennis-court']
# names = ['civilian', 'tank', 'truck', 'unarmed-vehicle']


names = ['baseball-diamond','basketball-court','bridge','ground-track-field','harbor','helicopter','large-vehicle','plane','roundabout','ship','small-vehicle','soccer-ball-field','storage-tank','swimming-pool', 'tennis-court']

@app.route('/predict',methods=['GET','POST'])
def predict():
    img1 = request.files['image1']
    img2 = request.files['image2']

    img1_pil = Image.open(img1.stream)
    img2_pil = Image.open(img2.stream)

    model = YOLO('best1.pt')
    # print(model.names) #in dict format  # Get class names from the model
    results = model.predict(source=img1_pil, conf=0.4, show=True, device='cpu')
    classes_1={}
    for result in results:
        img_out_1 = result.save('static/out1.jpg')
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            key = names[class_id]
            classes_1[key]=classes_1.get(key, 0) + 1
        
    results = model.predict(source=img2_pil, conf=0.4, show=True, device='cpu')
    classes_2={}
    for result in results:
        img_out_2 = result.save('static/out2.jpg')
        boxes = result.boxes
        # names = model.names #in dict format  # Get class names from the model
        # print(class_names)
        for box in boxes:
            class_id = int(box.cls[0])
            key= names[class_id]
            classes_2[key]=classes_2.get(key, 0) + 1

    isChanged = detectChange(classes_1,classes_2,names)

    print(img_out_1)
    print(img_out_2)
    # for(key, value) in classes_1.items():
        # print(f"Class {(class_names[key])}: {value}")
    return render_template('predict.html',path1=img_out_1[0], path2=img_out_2[0], class_1=classes_1, class_2=classes_2, isChanged=isChanged,classes=names)
    # return render_template('predict.html')

def detectChange(class1,class2,names):
    for i in names:
        print(i)
        if(class1.get(i,0)!= class2.get(i,0)):
            return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

# app.run()