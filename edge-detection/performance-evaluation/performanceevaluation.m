clear; clc; close all;

imageName = 'Deer'; % ENUM: Bird or Deer

gtFile = sprintf('%s_GT.mat', imageName);

gtData = load(gtFile);
GTs = gtData.groundTruth;
numGTs = length(GTs);

detectors = {'Sobel', 'Canny', 'SE'};
fileTypes = {'_prob.png', '_binary.jpg', '_prob.png'};
colors = {'g', 'k', 'b'};

figure('Name', sprintf('F-Measure: %s', imageName));
hold on;
fprintf('=== Evaluation for %s ===\n\n', imageName);

for d = 1:length(detectors)
    detName = detectors{d};
    fileName = sprintf('%s_%s%s', imageName, detName, fileTypes{d});
    
    E = double(imread(fileName)) / 255.0;
    E = 1.0 - E; % white background and black line
    
    isBinary = strcmp(detName, 'Canny');
    if isBinary
        thresholds = 0.5;
    else
        thresholds = 0.01:0.02:0.99;
    end
    
    numThresh = length(thresholds);
    meanP_all = zeros(1, numThresh);
    meanR_all = zeros(1, numThresh);
    F_all = zeros(1, numThresh);
    
    bestF = 0; bestT = 0;
    best_P_GT = zeros(1, numGTs);
    best_R_GT = zeros(1, numGTs);
    
    for tIdx = 1:numThresh
        t = thresholds(tIdx);
        if isBinary
            E_bin = E > 0.5; 
        else
            E_bin = E >= t;
        end
        P_GT = zeros(1, numGTs);
        R_GT = zeros(1, numGTs);
        
        for g = 1:numGTs
            gtBnd = GTs{g}.Boundaries;
            
            TP = sum(E_bin(:) & gtBnd(:));
            FP = sum(E_bin(:) & ~gtBnd(:));
            FN = sum(~E_bin(:) & gtBnd(:));
            
            P_GT(g) = TP / max(eps, TP + FP);
            R_GT(g) = TP / max(eps, TP + FN);
        end
        
        mP = mean(P_GT);
        mR = mean(R_GT);
        currentF = 2 * (mP * mR) / max(eps, mP + mR);
        meanP_all(tIdx) = mP;
        meanR_all(tIdx) = mR;
        F_all(tIdx) = currentF;
        
        if currentF >= bestF
            bestF = currentF;
            bestT = t;
            best_P_GT = P_GT;
            best_R_GT = R_GT;
        end
    end
    
    if isBinary
        plot(bestT, bestF, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k', 'DisplayName', sprintf('Canny (Max F=%.3f)', bestF));
    else
        plot(thresholds, F_all, 'Color', colors{d}, 'LineWidth', 2, 'DisplayName', sprintf('%s (Max F=%.3f)', detName, bestF));
        plot(bestT, bestF, 'ro', 'MarkerSize', 6, 'HandleVisibility', 'off'); % makr peak
    end
    fprintf('--- %s Detector ---\n', detName, bestT);
    fprintf('%-8s | %-12s | %-12s | %-12s\n', 'GT Index', 'Precision', 'Recall', 'F-Measure');
    
    for g = 1:numGTs
        p_val = best_P_GT(g);
        r_val = best_R_GT(g);
        
        f_val = 2 * (p_val * r_val) / max(eps, p_val + r_val);
        
        fprintf('GT %-5d | %-12.4f | %-12.4f | %-12.4f\n', g, p_val, r_val, f_val);
    end
    fprintf('Overall Mean P: %.4f, Mean R: %.4f, F: %.4f\n\n', mean(best_P_GT), mean(best_R_GT), bestF);
    
end

title(sprintf('F-Measure vs. Threshold - %s', imageName));
xlabel('Threshold Value');
ylabel('Overall F-Measure');
legend('Location', 'southoutside');
grid on;
hold off;