%% Computational MRI: Fall 2021
% Lab 1: Sequences and image contrast
% Part 2: T2 mapping
% florian.knoll@fau.de

clear all; close all; clc;

%% McGill numerical phantom: Plot CSF, T1 and T2 regions, and display T1, T2, SD
% Aubert-Broche Neuroimage 2006
load digital_brain_phantom.mat;
[nR,nC,nSl] = size(ph.label);

% Plot regions
figure, imshow(ph.label == 1,[]); title('brain phantom: label 1, CSF'); 
figure, imshow(ph.label == 2,[]); title('brain phantom: label 2, GM'); drawnow;
figure, imshow(ph.label == 3,[]); title('brain phantom: label 3, WM'); drawnow;

% Display T1, T2, SD
T1_csf = mean(ph.t1(ph.label == 1)); % Just use mean to select one element
T2_csf = mean(ph.t2(ph.label == 1));
SD_csf = mean(ph.sd(ph.label == 1));

disp(['T1 CSF = ', num2str(T1_csf), 'ms']);
disp(['T2 CSF = ', num2str(T2_csf), 'ms']);
disp(['SD CSF = ', num2str(SD_csf)]);

T1_gm = mean(ph.t1(ph.label == 2));
T2_gm = mean(ph.t2(ph.label == 2));
SD_gm = mean(ph.sd(ph.label == 2));

disp(['T1 GM = ', num2str(T1_gm), 'ms']);
disp(['T2 GM = ', num2str(T2_gm), 'ms']);
disp(['SD GM = ', num2str(SD_gm)]);

T1_wm = mean(ph.t1(ph.label == 3));
T2_wm = mean(ph.t2(ph.label == 3));
SD_wm = mean(ph.sd(ph.label == 3));

disp(['T1 WM = ', num2str(T1_wm), 'ms']);
disp(['T2 WM = ', num2str(T2_wm), 'ms']);
disp(['SD WM = ', num2str(SD_wm)]);