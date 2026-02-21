% Clear workspace and command window
clear; clc; close all;

% 1. Setup paths and load data
imageName = 'Bird'; % Change to 'Deer' when testing the other image
gtFile = sprintf('%s_GT.mat', imageName);
probMapFile = sprintf('%s_SE_prob.png', imageName);

% Load Ground Truth .mat file
% BSDS500 GTs are stored in a cell array named 'groundTruth'
gtData = load(gtFile);
GTs = gtData.groundTruth; 
numGTs = length(GTs);

% Read the generated probability edge map and normalize to [0, 1]
probEdgeMap = double(imread(probMapFile)) / 255.0;

% 2. Define Thresholds for Evaluation (Part 2)
thresholds = 0.05:0.05:0.95; % Testing thresholds from 0.05 to 0.95
numThresh = length(thresholds);

% Preallocate arrays to store overall means for plotting
overallMeanP = zeros(1, numThresh);
overallMeanR = zeros(1, numThresh);
overallFMeasure = zeros(1, numThresh);

fprintf('Evaluating %s...\n', imageName);
fprintf('--------------------------------------------------\n');

% 3. Iterate through each threshold
for tIdx = 1:numThresh
    t = thresholds(tIdx);
    binaryMap = probEdgeMap >= t; % Binarize the probability map
    
    P_per_GT = zeros(1, numGTs);
    R_per_GT = zeros(1, numGTs);
    
    % Evaluate against each of the 5 Ground Truths (Part 1 logic)
    for gIdx = 1:numGTs
        % Extract boundary coordinates from the GT cell array
        gtBoundary = GTs{gIdx}.Boundaries;
        
        % NOTE: The official SE toolbox uses a distance tolerance (bipartite 
        % graph matching) via edgesEvalImg(). If you don't have the compiled 
        % mex files, this exact pixel-wise matching is the standard alternative:
        TP = sum(binaryMap(:) & gtBoundary(:));
        FP = sum(binaryMap(:) & ~gtBoundary(:));
        FN = sum(~binaryMap(:) & gtBoundary(:));
        
        % Calculate Precision and Recall, avoiding division by zero
        if (TP + FP) > 0
            P_per_GT(gIdx) = TP / (TP + FP);
        else
            P_per_GT(gIdx) = 0;
        end
        
        if (TP + FN) > 0
            R_per_GT(gIdx) = TP / (TP + FN);
        else
            R_per_GT(gIdx) = 0;
        end
    end
    
    % Compute mean Precision and mean Recall across all 5 GTs for this threshold
    overallMeanP(tIdx) = mean(P_per_GT);
    overallMeanR(tIdx) = mean(R_per_GT);
    
    % Calculate overall F-measure
    if (overallMeanP(tIdx) + overallMeanR(tIdx)) > 0
        overallFMeasure(tIdx) = 2 * (overallMeanP(tIdx) * overallMeanR(tIdx)) / ...
                                    (overallMeanP(tIdx) + overallMeanR(tIdx));
    else
        overallFMeasure(tIdx) = 0;
    end
    
    % Print results for a specific threshold (e.g., t = 0.8) to answer Part (1)
    if abs(t - 0.8) < 0.01
        fprintf('\n--- Results at Threshold p = 0.80 ---\n');
        for gIdx = 1:numGTs
            fprintf('GT %d -> Precision: %.4f, Recall: %.4f\n', ...
                    gIdx, P_per_GT(gIdx), R_per_GT(gIdx));
        end
        fprintf('Overall Mean P: %.4f, Overall Mean R: %.4f, Final F: %.4f\n\n', ...
                overallMeanP(tIdx), overallMeanR(tIdx), overallFMeasure(tIdx));
    end
end

% 4. Find the Best F-Measure (Part 2)
[bestF, bestIdx] = max(overallFMeasure);
bestThresh = thresholds(bestIdx);

fprintf('Optimal Threshold (Best F-Measure): %.2f with F = %.4f\n', bestThresh, bestF);

% 5. Plot F-Measure vs. Threshold (Part 2)
figure('Name', sprintf('F-Measure Curve: %s', imageName));
plot(thresholds, overallFMeasure, '-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(bestThresh, bestF, 'r*', 'MarkerSize', 10, 'LineWidth', 2); % Highlight peak
title(sprintf('F-Measure vs. Threshold (%s)', imageName));
xlabel('Threshold Value');
ylabel('Overall F-Measure');
grid on;
legend('F-Measure Curve', sprintf('Max F = %.3f at t = %.2f', bestF, bestThresh), ...
'Location', 'southoutside');
