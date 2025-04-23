% Charger l'image
img = imread('personne.jpg'); % Remplacez par le chemin de votre image

% Vérifier si l'image est en niveaux de gris, sinon convertir
if size(img, 3) == 3
    grayImg = rgb2gray(img); % Convertir en niveaux de gris
else
    grayImg = img;
end

% Détecteurs de composants faciaux basés sur Viola-Jones
faceDetector = vision.CascadeObjectDetector(); % Détecteur de visage
eyeDetector = vision.CascadeObjectDetector('EyePairBig'); % Détecteur des yeux
mouthDetector = vision.CascadeObjectDetector('Mouth'); % Détecteur de bouche
noseDetector = vision.CascadeObjectDetector('Nose'); % Détecteur de nez

% Étape 1 : Détection du visage
faceBBox = step(faceDetector, grayImg); % Détecter les visages
if isempty(faceBBox)
    disp('Aucun visage détecté !');
    return;
end

% Annoter les visages détectés
annotatedImg = insertObjectAnnotation(img, 'rectangle', faceBBox, 'Visage', 'Color', 'yellow');

% Étape 2 : Détection des yeux
for i = 1:size(faceBBox, 1)
    % Définir la région supérieure du visage pour chercher les yeux
    upperFace = [faceBBox(i, 1), faceBBox(i, 2), faceBBox(i, 3), round(faceBBox(i, 4) * 0.5)];
    roiUpperFace = imcrop(grayImg, upperFace);

    % Détecter les yeux
    eyesBBox = step(eyeDetector, roiUpperFace);
    if ~isempty(eyesBBox)
        % Ajuster les coordonnées pour l'image d'origine
        eyesBBox(:, 1) = eyesBBox(:, 1) + upperFace(1);
        eyesBBox(:, 2) = eyesBBox(:, 2) + upperFace(2);

        % Annoter les yeux
        annotatedImg = insertObjectAnnotation(annotatedImg, 'rectangle', eyesBBox, 'Yeux', 'Color', 'blue');
    end
end

% Étape 3 : Détection du nez
for i = 1:size(faceBBox, 1)
    % Définir la région centrale pour chercher le nez
    midFace = [faceBBox(i, 1) + faceBBox(i, 3) * 0.25, faceBBox(i, 2) + faceBBox(i, 4) * 0.3, ...
               faceBBox(i, 3) * 0.5, faceBBox(i, 4) * 0.4];
    roiMidFace = imcrop(grayImg, midFace);

    % Détecter le nez
    noseBBox = step(noseDetector, roiMidFace);
    if ~isempty(noseBBox)
        % Ajuster les coordonnées pour l'image d'origine
        noseBBox(:, 1) = noseBBox(:, 1) + midFace(1);
        noseBBox(:, 2) = noseBBox(:, 2) + midFace(2);

        % Annoter le nez
        annotatedImg = insertObjectAnnotation(annotatedImg, 'rectangle', noseBBox, 'Nez', 'Color', 'green');
    end
end

% Étape 4 : Détection de la bouche
for i = 1:size(faceBBox, 1)
    % Définir la région inférieure pour chercher la bouche
    lowerFace = [faceBBox(i, 1), faceBBox(i, 2) + faceBBox(i, 4) * 0.5, faceBBox(i, 3), faceBBox(i, 4) * 0.5];
    roiLowerFace = imcrop(grayImg, lowerFace);

    % Détecter la bouche
    mouthBBox = step(mouthDetector, roiLowerFace);
    if ~isempty(mouthBBox)
        % Ajuster les coordonnées pour l'image d'origine
        mouthBBox(:, 1) = mouthBBox(:, 1) + lowerFace(1);
        mouthBBox(:, 2) = mouthBBox(:, 2) + lowerFace(2);

        % Annoter la bouche
        annotatedImg = insertObjectAnnotation(annotatedImg, 'rectangle', mouthBBox, 'Bouche', 'Color', 'red');
    end
end

% Afficher l'image annotée
imshow(annotatedImg);
title('Détection des yeux, nez, bouche et visage');
