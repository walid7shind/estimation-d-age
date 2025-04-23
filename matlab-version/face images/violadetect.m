clc; clear; close all;

inputFolder = '60-90/'; 
outputFolder = 'output/'; 
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end


imageFiles = dir(fullfile(inputFolder, '*.png'));

% détecteurs basés sur Viola-Jones (AdaBoost)
faceDetector = vision.CascadeObjectDetector(); 
faceDetector.MergeThreshold = 1; % Ajustable

eyeDetector = vision.CascadeObjectDetector('EyePairBig'); 
mouthDetector = vision.CascadeObjectDetector('Mouth'); 


for idx = 1:length(imageFiles)
    
    imagePath = fullfile(inputFolder, imageFiles(idx).name);
    img = imread(imagePath);
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    faceBBox = step(faceDetector, grayImg);
    if isempty(faceBBox)
        disp(['Aucun visage détecté dans : ', imageFiles(idx).name]);
        continue;
    end
    annotatedImg = img;

    for i = 1:size(faceBBox, 1)
        % zone sup pour recherche des yeux
        upperFace = [faceBBox(i, 1), faceBBox(i, 2), faceBBox(i, 3), round(faceBBox(i, 4) * 0.5)];
        roiUpperFace = imcrop(grayImg, upperFace);


        eyesBBox = step(eyeDetector, roiUpperFace);
        if ~isempty(eyesBBox)
            eyesBBox(:, 1) = eyesBBox(:, 1) + upperFace(1);
            eyesBBox(:, 2) = eyesBBox(:, 2) + upperFace(2);
            annotatedImg = insertObjectAnnotation(annotatedImg, 'rectangle', eyesBBox, 'Yeux', 'Color', 'blue');
        end
    end

    
    for i = 1:size(faceBBox, 1)
        % zone inf pour chercher la bouche
        lowerFace = [faceBBox(i, 1), faceBBox(i, 2) + faceBBox(i, 4) * 0.5, faceBBox(i, 3), faceBBox(i, 4) * 0.5];
        roiLowerFace = imcrop(grayImg, lowerFace);

        mouthBBox = step(mouthDetector, roiLowerFace);
        if ~isempty(mouthBBox)
            mouthBBox(:, 1) = mouthBBox(:, 1) + lowerFace(1);
            mouthBBox(:, 2) = mouthBBox(:, 2) + lowerFace(2);

            % Garder la bouche située le plus en bas
            [~, lowestIndex] = max(mouthBBox(:, 2));
            mouthBBox = mouthBBox(lowestIndex, :);
            annotatedImg = insertObjectAnnotation(annotatedImg, 'rectangle', mouthBBox, 'Bouche', 'Color', 'red');
        end
    end
    outputImagePath = fullfile(outputFolder, imageFiles(idx).name);
    imwrite(annotatedImg, outputImagePath);

    figure, imshow(annotatedImg);
    title(['Détection des yeux et bouche : ', imageFiles(idx).name]);
end

disp('Traitement terminé ! Résultats enregistrés dans le dossier output/');
