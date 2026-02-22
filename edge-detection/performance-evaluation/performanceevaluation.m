% evaluate_standalone.m
clear; clc; close all;

% Configuration
imageName = 'Deer'; % Change this to 'Deer' to run the other image
gtFile = sprintf('%s_GT.mat', imageName);

% Load Ground Truth data
gtData = load(gtFile);
GTs = gtData.groundTruth;
numGTs = length(GTs);

% Define detectors and their output map types
detectors = {'Sobel', 'Canny', 'SE'};
fileTypes = {'_prob.png', '_binary.jpg', '_prob.png'};
colors = {'g', 'k', 'b'}; % Green = Sobel, Black = Canny, Blue = SE

% Prepare the plot
figure('Name', sprintf('F-Measure Comparison: %s', imageName));
hold on;
fprintf('=== Standalone Evaluation for %s ===\n\n', imageName);

% Loop through each of the 3 detectors
for d = 1:length(detectors)
    detName = detectors{d};
    fileName = sprintf('%s_%s%s', imageName, detName, fileTypes{d});
    
    
    % Load image and scale to [0, 1]
    E = double(imread(fileName)) / 255.0;
    
    % INVERT COLORS: Ensure edges are 1 (white) and background is 0 (black)
    E = 1.0 - E; 
    
    % Determine if we are testing a curve (continuous) or a point (binary)
    isBinary = strcmp(detName, 'Canny');
    if isBinary
        thresholds = 0.5; % Dummy threshold since Canny is already binarized
    else
        thresholds = 0.01:0.02:0.99; % Sweep 50 thresholds for Sobel and SE
    end
    
    numThresh = length(thresholds);
    meanP_all = zeros(1, numThresh);
    meanR_all = zeros(1, numThresh);
    F_all = zeros(1, numThresh);
    
    bestF = 0; bestT = 0;
    best_P_GT = zeros(1, numGTs);
    best_R_GT = zeros(1, numGTs);
    
    % Iterate through thresholds
    for tIdx = 1:numThresh
        t = thresholds(tIdx);
        
        % Binarize the map at the current threshold
        if isBinary
            E_bin = E > 0.5; 
        else
            E_bin = E >= t;
        end
        
        P_GT = zeros(1, numGTs);
        R_GT = zeros(1, numGTs);
        
        % Evaluate against each of the 5 Ground Truths
        for g = 1:numGTs
            gtBnd = GTs{g}.Boundaries;
            
            % STRICT PIXEL-WISE MATCHING (TP, FP, FN)
            TP = sum(E_bin(:) & gtBnd(:));
            FP = sum(E_bin(:) & ~gtBnd(:));
            FN = sum(~E_bin(:) & gtBnd(:));
            
            P_GT(g) = TP / max(eps, TP + FP);
            R_GT(g) = TP / max(eps, TP + FN);
        end
        
        % Calculate Overall Means and F-Measure for this threshold
        mP = mean(P_GT);
        mR = mean(R_GT);
        currentF = 2 * (mP * mR) / max(eps, mP + mR);
        
        meanP_all(tIdx) = mP;
        meanR_all(tIdx) = mR;
        F_all(tIdx) = currentF;
        
        % Keep track of the best performing threshold for the table
        if currentF >= bestF
            bestF = currentF;
            bestT = t;
            best_P_GT = P_GT;
            best_R_GT = R_GT;
        end
    end
    
    % Plot the results onto the chart
    if isBinary
        plot(bestT, bestF, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', sprintf('Canny (Max F=%.3f)', bestF));
    else
        plot(thresholds, F_all, 'Color', colors{d}, 'LineWidth', 2, 'DisplayName', sprintf('%s (Max F=%.3f)', detName, bestF));
        plot(bestT, bestF, 'ro', 'MarkerSize', 6, 'HandleVisibility', 'off'); % Mark the peak
    end
    
    % Print the Table Data for the Command Window (NOW INCLUDES INDIVIDUAL F-MEASURES)
    fprintf('--- %s Detector (Optimal Threshold t = %.2f) ---\n', detName, bestT);
    fprintf('%-8s | %-12s | %-12s | %-12s\n', 'GT Index', 'Precision', 'Recall', 'F-Measure');
    
    for g = 1:numGTs
        p_val = best_P_GT(g);
        r_val = best_R_GT(g);
        
        % Calculate F-Measure for this specific GT row
        f_val = 2 * (p_val * r_val) / max(eps, p_val + r_val);
        
        fprintf('GT %-5d | %-12.4f | %-12.4f | %-12.4f\n', g, p_val, r_val, f_val);
    end
    
    fprintf('Overall Mean P: %.4f, Mean R: %.4f, Final F: %.4f\n\n', mean(best_P_GT), mean(best_R_GT), bestF);
    
end

% Finalize the plot formatting
title(sprintf('F-Measure vs. Threshold - %s', imageName));
xlabel('Threshold Value');
ylabel('Overall F-Measure');
legend('Location', 'southoutside');
grid on;
hold off;