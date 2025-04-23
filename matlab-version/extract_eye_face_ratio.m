function eyeFaceRatio = extract_eye_face_ratio(img)
    faceDetector = vision.CascadeObjectDetector(); 
    faceDetector.MergeThreshold = 1; 
    grayImg = rgb2gray(img);
    faceBBox = step(faceDetector, grayImg);
    
    if isempty(faceBBox)
        %disp('⚠️ Visage non détecté');
        eyeFaceRatio = NaN;
        return;
    end
    eyeDetector = vision.CascadeObjectDetector('EyePairBig');
    upperFace = [faceBBox(1,1), faceBBox(1,2), faceBBox(1,3), round(faceBBox(1,4) * 0.5)];
    roiUpperFace = imcrop(img, upperFace);
    eyesBBox = step(eyeDetector, rgb2gray(roiUpperFace));
    
    if isempty(eyesBBox)
        %disp('⚠️ Yeux non détectés');
        eyeFaceRatio = NaN;
        return;
    end
    eyesBBox(:, 1) = eyesBBox(:, 1) + upperFace(1);
    eyesBBox(:, 2) = eyesBBox(:, 2) + upperFace(2);
    
    % Calcul du ratio Surface des Yeux / Surface du Visage
    eyeArea = eyesBBox(1,3) * eyesBBox(1,4);
    faceArea = faceBBox(1,3) * faceBBox(1,4);
    eyeFaceRatio = eyeArea / faceArea;

    % Régions d'Intérêt
    % figure;
    % imshow(img);
    % hold on;
    % rectangle('Position', eyesBBox, 'EdgeColor', 'g', 'LineWidth', 2);
    % text(eyesBBox(1,1), eyesBBox(1,2) - 10, 'Yeux', 'Color', 'g', 'FontSize', 12, 'FontWeight', 'bold');
    % rectangle('Position', faceBBox, 'EdgeColor', 'b', 'LineWidth', 2);
    %  text(faceBBox(1,1), faceBBox(1,2) - 10, 'Visage', 'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold');

    % title(sprintf('Ratio Yeux/Visage : %.4f', eyeFaceRatio));
   %  hold off;
end
