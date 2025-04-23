function pocketDiff = extract_pocket_diff(img)
    imgLab = rgb2lab(img);
    % detection du visage:
    faceDetector = vision.CascadeObjectDetector(); 
    faceDetector.MergeThreshold = 1; 
    grayImg = rgb2gray(img);
    faceBBox = step(faceDetector, grayImg);
    
    if isempty(faceBBox)
        %disp('Visage non détecté');
        pocketDiff = NaN; 
        return;
    end
    % detection des yeux dans la partie superieur du visage
    eyeDetector = vision.CascadeObjectDetector('EyePairBig');
    upperFace = [faceBBox(1,1), faceBBox(1,2), faceBBox(1,3), round(faceBBox(1,4) * 0.5)];
    roiUpperFace = imcrop(img, upperFace);
    eyesBBox = step(eyeDetector, rgb2gray(roiUpperFace));
    
    if isempty(eyesBBox)
        % disp('Yeux non détectés');
        pocketDiff = NaN;
        return;
    end
    eyesBBox(:, 1) = eyesBBox(:, 1) + upperFace(1);
    eyesBBox(:, 2) = eyesBBox(:, 2) + upperFace(2);
    
    % la région sous l'oeil GAUCHE
    eyeX = eyesBBox(1,1);
    eyeY = eyesBBox(1,2);
    eyeWidth = eyesBBox(1,3);
    eyeHeight = eyesBBox(1,4);

    shiftUp = round(eyeHeight * 0.25); 
    spacing = round(eyeWidth * 0.2); 
    underEyeHeight = round(eyeHeight * 0.5); 

    underLeftEye = [eyeX, eyeY + eyeHeight - shiftUp, (eyeWidth/2) - spacing/2, underEyeHeight];

    % région de la Joue GAUCHE (juste sous la poche)
    cheekLeft = [underLeftEye(1), underLeftEye(2) + underEyeHeight, underLeftEye(3), underEyeHeight];

    % images sous l'œil gauche et de la joue
    underLeftEyeImg_a = imcrop(imgLab(:,:,2), underLeftEye);
    underLeftEyeImg_b = imcrop(imgLab(:,:,3), underLeftEye);
    
    cheekLeftImg_a = imcrop(imgLab(:,:,2), cheekLeft);
    cheekLeftImg_b = imcrop(imgLab(:,:,3), cheekLeft);

    % descripteur (Moyenne des axes a et b)
    mean_pocket = mean(underLeftEyeImg_a(:)) + mean(underLeftEyeImg_b(:));
    mean_cheek = mean(cheekLeftImg_a(:)) + mean(cheekLeftImg_b(:));
    
    pocketDiff = mean_pocket - mean_cheek;

   % Visualisation des Régions d'Intérêt
   % figure;
   % imshow(img);
   % hold on;
   % rectangle('Position', underLeftEye, 'EdgeColor', 'r', 'LineWidth', 2);
   % text(underLeftEye(1), underLeftEye(2) - 10, 'Poche gauche', 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold');
   % rectangle('Position', cheekLeft, 'EdgeColor', 'b', 'LineWidth', 2);
   % text(cheekLeft(1), cheekLeft(2) - 10, 'Joue gauche', 'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold');
   % title(sprintf('Différence de Poches (a+b) : %.2f', pocketDiff));
   % hold off;
end
