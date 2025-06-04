import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
ap = argparse.ArgumentParser(description='Emotion Recognition System')
ap.add_argument("--mode", help="Mode to run: 'train' to train model, 'display' for real-time detection", 
                required=True, choices=['train', 'display'])
ap.add_argument("--cam_source", help="Camera source for display mode (default=0)", type=int, default=0)
args = ap.parse_args()
def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    
    # Smoothing
    def smooth_curve(points, factor=0.8):
        smoothed = []
        for point in points:
            if smoothed:
                prev = smoothed[-1]
                smoothed.append(prev * factor + point * (1 - factor))
            else:
                smoothed.append(point)
        return smoothed
    
    # Accurac
    train_acc = smooth_curve(model_history.history['accuracy'])
    val_acc = smooth_curve(model_history.history['val_accuracy'])
    epochs = range(1, len(train_acc) + 1)
    
    axs[0].plot(epochs, train_acc, 'bo-', label='Training Accuracy')
    axs[0].plot(epochs, val_acc, 'go-', label='Validation Accuracy')
    axs[0].set_title('Training and Validation Accuracy', fontsize=14)
    axs[0].set_xlabel('Epochs', fontsize=12)
    axs[0].set_ylabel('Accuracy', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='lower right')
    

    train_loss = smooth_curve(model_history.history['loss'])
    val_loss = smooth_curve(model_history.history['val_loss'])
    
    axs[1].plot(epochs, train_loss, 'ro-', label='Training Loss')
    axs[1].plot(epochs, val_loss, 'mo-', label='Validation Loss')
    axs[1].set_title('Training and Validation Loss', fontsize=14)
    axs[1].set_xlabel('Epochs', fontsize=12)
    axs[1].set_ylabel('Loss', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right')
    
    plt.tight_layout()
    fig.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 128  
num_epoch = 100  
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False
)
def create_model(input_shape=(48, 48, 1)):
    model = Sequential()
    
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    return model

model = create_model()
if args.mode == "train":
    # Callbacks for better training
    checkpoint = ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    

    print("[INFO] Training model...")
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    plot_model_history(model_info)
    model.save('emotion_model.h5') 
    print("[INFO] Model training complete and saved.")


elif args.mode == "display":
    if not os.path.exists('emotion_model.h5'):
        raise FileNotFoundError("No trained model found. Please train the model first.")
    
    model = create_model()
    model.load_weights('emotion_model.h5')
    
    cv2.ocl.setUseOpenCL(False)
    

    emotion_dict = {
        0: ("Angry", (0, 0, 255)),
        1: ("Disgusted", (0, 102, 0)),
        2: ("Fearful", (255, 0, 0)),
        3: ("Happy", (0, 255, 0)),
        4: ("Neutral", (255, 255, 255)),
        5: ("Sad", (255, 0, 255)),
        6: ("Surprised", (0, 255, 255))
    }
    
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    

    cap = cv2.VideoCapture(args.cam_source)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")
    
    cv2.namedWindow('Emotion Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Emotion Recognition', 1200, 800)
    frame_count = 0
    fps = 0
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            end_time = cv2.getTickCount()
            fps = 30 * cv2.getTickFrequency() / (end_time - start_time)
            start_time = end_time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)
                
                
                prediction = model.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                emotion_label, color = emotion_dict[maxindex]
                confidence = prediction[maxindex]
                
                
                label = f"{emotion_label}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            
                for i, (emotion, (_, col)) in enumerate(emotion_dict.items()):
                    cv2.putText(frame, emotion_dict[emotion][0], (10, 20 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.rectangle(frame, (150, 10 + i*30), 
                                  (150 + int(200 * prediction[emotion]), 20 + i*30), 
                                  emotion_dict[emotion][1], -1)
        
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        
        cv2.imshow('Emotion Recognition', frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    cap.release()
    cv2.destroyAllWindows()