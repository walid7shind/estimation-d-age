import os
import cv2
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==== 1. FONCTIONS D'EXTRACTION DE DESCRIPTEURS ====

def boite_in_im(bboxes):
    # retourne la boîte la plus haute (comme dans la version MATLAB)
    boite = [200, 200, 1, 1]
    for b in bboxes:
        if b[0] < 200 and b[1] < 200:
            if b[1] < boite[1]:
                boite = b
    return boite

def extract_eye_face_ratio(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return np.nan
    x, y, w, h = faces[0]
    upper_face = img[y:y + h // 2, x:x + w]
    eyes = eye_cascade.detectMultiScale(cv2.cvtColor(upper_face, cv2.COLOR_RGB2GRAY))
    if len(eyes) == 0:
        return np.nan
    ex, ey, ew, eh = eyes[0]
    eye_area = ew * eh
    face_area = w * h
    return eye_area / face_area

def extract_pocket_diff(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return np.nan
    x, y, w, h = faces[0]
    upper_face = img[y:y + h // 2, x:x + w]
    eyes = eye_cascade.detectMultiScale(cv2.cvtColor(upper_face, cv2.COLOR_RGB2GRAY))
    if len(eyes) == 0:
        return np.nan
    ex, ey, ew, eh = eyes[0]
    eyeX, eyeY = ex + x, ey + y
    shift_up = int(eh * 0.25)
    spacing = int(ew * 0.2)
    under_h = int(eh * 0.5)
    under_eye = [eyeX, eyeY + eh - shift_up, ew // 2 - spacing // 2, under_h]
    cheek = [under_eye[0], under_eye[1] + under_h, under_eye[2], under_h]

    a_under = cv2.mean(cv2.cvtColor(img[under_eye[1]:under_eye[1]+under_eye[3], under_eye[0]:under_eye[0]+under_eye[2]], cv2.COLOR_RGB2Lab)[:,:,1])[0]
    b_under = cv2.mean(cv2.cvtColor(img[under_eye[1]:under_eye[1]+under_eye[3], under_eye[0]:under_eye[0]+under_eye[2]], cv2.COLOR_RGB2Lab)[:,:,2])[0]
    a_cheek = cv2.mean(cv2.cvtColor(img[cheek[1]:cheek[1]+cheek[3], cheek[0]:cheek[0]+cheek[2]], cv2.COLOR_RGB2Lab)[:,:,1])[0]
    b_cheek = cv2.mean(cv2.cvtColor(img[cheek[1]:cheek[1]+cheek[3], cheek[0]:cheek[0]+cheek[2]], cv2.COLOR_RGB2Lab)[:,:,2])[0]

    return (a_under + b_under) - (a_cheek + b_cheek)

def extract_levres(img):
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    roi = img[115:200, :]
    bboxes = mouth_cascade.detectMultiScale(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), 1.1, 5)
    if len(bboxes) == 0:
        return np.nan
    b = boite_in_im(bboxes)
    y = b[1] + 115
    x = max(b[0] - 20, 0)
    w = b[2] + 40
    h = b[3]
    im_red = lab[y:y+h, x:x+w, 1]
    joue = np.hstack((lab[:, :21, 1], lab[:, -21:, 1]))
    return abs(np.max(im_red) - np.max(joue))

def extract_ride(img):
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    roi = img[115:200, :]
    bboxes = mouth_cascade.detectMultiScale(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), 1.1, 5)
    if len(bboxes) == 0:
        return np.nan
    b = boite_in_im(bboxes)
    y = b[1] + 115
    x = max(b[0] - 20, 0)
    w = b[2] + 40
    h = b[3]
    grad = cv2.Sobel(lab[y:y+h, x:x+w, 0], cv2.CV_64F, 1, 1, ksize=5)
    left = grad[:, :21]
    right = grad[:, -21:]
    return (np.mean(left) + np.mean(right)) / 2

def extract_sillon_naso(img):
    nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    roi = img[50:150, 60:140]
    bboxes = nose_cascade.detectMultiScale(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), 1.1, 5)
    if len(bboxes) == 0:
        return np.nan
    b = boite_in_im(bboxes)
    y = b[1] + 50
    x = b[0] + 60
    w = b[2] + 40
    h = b[3]
    grad = cv2.Sobel(lab[y:y+h, x:x+w, 0], cv2.CV_64F, 1, 0, ksize=5)
    left = grad[:, :21]
    right = grad[:, -21:]
    return (np.mean(left) + np.mean(right)) / 2

# ==== 2. CHARGEMENT DES IMAGES ET EXTRACTION ====

main_input_folder = 'face images'
subfolders = ['0_10', '10-30', '30-60', '60-90']
class_map = {'0_10': 1, '10-30': 2, '30-60': 3, '60-90': 4}
class_labels = ['0-10', '10-30', '30-60', '60-90']

feature_matrix = []
labels = []

for folder in subfolders:
    folder_path = os.path.join(main_input_folder, folder)
    for fname in os.listdir(folder_path):
        if fname.endswith('.png'):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            features = [
                extract_ride(img_rgb),
                extract_levres(img_rgb),
                extract_sillon_naso(img_rgb),
                extract_eye_face_ratio(img_rgb),
                extract_pocket_diff(img_rgb)
            ]
            if not any(np.isnan(features)):
                feature_matrix.append(features)
                labels.append(class_map[folder])

X = np.array(feature_matrix)
y = np.array(labels)

# ==== 3. CLASSIFICATION KNN ====

kf = KFold(n_splits=3, shuffle=True, random_state=42)
conf_mat_total = np.zeros((4, 4), dtype=int)
total_accuracy = 0

for train_idx, test_idx in kf.split(X):
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])
    conf = confusion_matrix(y[test_idx], y_pred, labels=[1, 2, 3, 4])
    conf_mat_total += conf
    total_accuracy += np.mean(y[test_idx] == y_pred)

final_accuracy = total_accuracy / 3
print(f"Précision moyenne du modèle : {final_accuracy * 100:.2f}%")

disp = ConfusionMatrixDisplay(conf_mat_total, display_labels=class_labels)
disp.plot()
plt.title("Matrice de confusion - KNN")
plt.show()

# ==== 4. MÉTRIQUES PAR CLASSE ====

for i, label in enumerate(class_labels):
    tp = conf_mat_total[i, i]
    fp = conf_mat_total[:, i].sum() - tp
    fn = conf_mat_total[i, :].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    err = (fp + fn) / conf_mat_total.sum()
    print(f"{label} -> Précision: {prec:.2f}, Rappel: {rec:.2f}, F1: {f1:.2f}, Erreur: {err:.2f}")

