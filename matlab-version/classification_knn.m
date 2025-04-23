clc; clear; close all;

mainInputFolder = 'face images/';
subfolders = {'0_10', '10-30', '30-60', '60-90'};
classLabels = {'0-10', '10-30', '30-60', '60-90'};

% pour stocker les caractéristiques et labels
featureMatrix = [];
labels = [];

% ID pour chaque classe
classMap = containers.Map({'0_10', '10-30', '30-60', '60-90'}, [1, 2, 3, 4]);

for f = 1:length(subfolders)
    inputFolder = fullfile(mainInputFolder, subfolders{f});
    imageFiles = dir(fullfile(inputFolder, '*.png')); 
    
    for idx = 1:length(imageFiles)
        imagePath = fullfile(inputFolder, imageFiles(idx).name);
        img = imread(imagePath);

        % extraction des 5 descripteurs
        rideValue = ride(img);
        levresValue = levres(img);
        sillonValue = sillon_naso(img);
        eyeFaceRatio = extract_eye_face_ratio(img);
        pocketDiff = extract_pocket_diff(img);

        % on stocke les résultats si les valeurs sont valides
        if ~isnan(rideValue) && ~isnan(levresValue) && ~isnan(sillonValue) && ...
           ~isnan(eyeFaceRatio) && ~isnan(pocketDiff)
       
            featureVector = [rideValue, levresValue, sillonValue, eyeFaceRatio, pocketDiff];
            featureMatrix = [featureMatrix; featureVector];
            labels = [labels; classMap(subfolders{f})]; % Assigner l'ID de la classe
        end
    end
end

% Visualisation des descripteurs:
featureNames = {'Rides', 'Lèvres', 'Sillon Nasogénien', 'Ratio Yeux/Visage', 'Poches sous les Yeux'};

figure;
for i = 1:size(featureMatrix, 2)
    subplot(2,3,i);
    boxplot(featureMatrix(:,i), labels, 'Labels', classLabels);
    title(featureNames{i});
    xlabel('Classes d age');
    ylabel('Valeur du descripteur');
    grid on;
end
sgtitle('Distribution des Descripteurs par Classe d age');

% KNN et Validation K-Fold (K=3)
K = 3;
cv = cvpartition(size(featureMatrix, 1), 'KFold', 3);
totalAccuracy = 0;
bestConfMat = zeros(4,4); % Matrice de confusion

for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    trainFeatures = featureMatrix(trainIdx, :);
    trainLabels = labels(trainIdx);
    testFeatures = featureMatrix(testIdx, :);
    testLabels = labels(testIdx);
    
    % knn training
    Mdl = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', K);
    predictedLabels = predict(Mdl, testFeatures);
    
    % Matrice de confusion pour ce fold
    confMat = confusionmat(testLabels, predictedLabels);
    bestConfMat = bestConfMat + confMat; 
    % precision pour ce fold
    accuracy = sum(diag(confMat)) / sum(confMat(:));
    totalAccuracy = totalAccuracy + accuracy;
end

% Moyenne des precisions sur tous les folds
finalAccuracy = totalAccuracy / cv.NumTestSets;


fprintf('Precision moyenne du modèle : %.2f%%\n', finalAccuracy * 100);

% Matrice de Confusion
figure;
confusionchart(bestConfMat);
title(sprintf('Matrice de Confusion - KNN (K=%d)', K));

% evaluation du model
numClasses = size(bestConfMat, 1);
precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
fscore = zeros(numClasses, 1);
errorRate = zeros(numClasses, 1);

for i = 1:numClasses
    TP = bestConfMat(i, i);
    FP = sum(bestConfMat(:, i)) - TP;
    FN = sum(bestConfMat(i, :)) - TP;
    TN = sum(bestConfMat(:)) - (TP + FP + FN);

    if (TP + FP) > 0
        precision(i) = TP / (TP + FP);
    else
        precision(i) = 0;
    end

    if (TP + FN) > 0
        recall(i) = TP / (TP + FN);
    else
        recall(i) = 0;
    end

    if (precision(i) + recall(i)) > 0
        fscore(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    else
        fscore(i) = 0;
    end

    errorRate(i) = (FP + FN) / sum(bestConfMat(:));
end

%Affichage resultat
fprintf('\nRésultats par classe :\n');
fprintf('Classe\tPrécision\tRappel\t\tF-Score\t\tTaux dErreur\n');
for i = 1:numClasses
    fprintf('%d\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n', i, precision(i), recall(i), fscore(i), errorRate(i));
end
