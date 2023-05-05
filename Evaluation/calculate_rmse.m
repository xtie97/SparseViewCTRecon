clear; clc; close all;
%%
df_test = readtable('/media/xintie/Elements/DeepEnChroma/Data_rcn/test_index.csv');
root_dir = '/media/xintie/Elements/DeepEnChroma/Data_rcn/dense_view';
output_dir1 =  '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_sep_sino/output';
output_dir2 =  '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_sep_grad/output';

CT_filenames = df_test.high;
examID_list = [61, 62, 63, 64, 65, 66, 68, 69];
% examID_list = [69]
MAE1 = [];
rRMSE1 = [];
SSIM1 = [];
MAE2 = [];
rRMSE2 = [];
SSIM2 = [];
for examID = examID_list
    for ii = 1: length(CT_filenames)
        fprintf('Exam: %d, Progress: %d %% \r', examID, round(ii/length(CT_filenames)*100));
        filename = replace(CT_filenames{ii}, '.raw', '_rcn.raw');
        if contains(filename, ['exam', int2str(examID)])
            GT = read_raw(fullfile(root_dir, filename), 'float32', [512, 512]) - 1024;
            
            output1 = read_raw(fullfile(output_dir1, filename), 'float32', [512, 512]) - 1024;
            output1 = output1(GT>-200 & GT<2000);
            output2 = read_raw(fullfile(output_dir2, filename), 'float32', [512, 512]) - 1024;
            output2 = output2(GT>-200 & GT<2000);
            GT = GT(GT>-200 & GT<2000);
            
            MAE_ind = mean(abs(GT-output1));
            rRMSE_ind =  sqrt(mean((GT-output1).^2)) / (max(GT(:)) - min(GT(:)));
            ssim_ind = ssim(output1, GT, "radius", 3, "DynamicRange", 2000);
            MAE1 = [MAE1, MAE_ind];
            rRMSE1 = [rRMSE1, rRMSE_ind*100];
            SSIM1 = [SSIM1, ssim_ind];
            
            MAE_ind = mean(abs(GT-output2));
            rRMSE_ind =  sqrt(mean((GT-output2).^2)) / (max(GT(:)) - min(GT(:)));
            ssim_ind = ssim(output2, GT, "radius", 3, "DynamicRange", 2000);
            
            MAE2 = [MAE2, MAE_ind];
            rRMSE2 = [rRMSE2, rRMSE_ind*100];
            SSIM2 = [SSIM2, ssim_ind];
        end
    end
    fclose('all');
    
end

fprintf('MAE\n')
fprintf('%.1f [%.1f, %.1f]\n', median(MAE1), prctile(MAE1, 25), prctile(MAE1, 75));
fprintf('%.1f [%.1f, %.1f]\n', median(MAE2), prctile(MAE2, 25), prctile(MAE2, 75));
fprintf('rRMSE\n')
fprintf('%.2f [%.2f, %.2f]\n', median(rRMSE1), prctile(rRMSE1, 25), prctile(rRMSE1, 75));
fprintf('%.2f [%.2f, %.2f]\n', median(rRMSE2), prctile(rRMSE2, 25), prctile(rRMSE2, 75));
fprintf('SSIM\n')
fprintf('%.3f [%.3f, %.3f]\n', median(SSIM1), prctile(SSIM1, 25), prctile(SSIM1, 75));
fprintf('%.3f [%.3f, %.3f]\n', median(SSIM2), prctile(SSIM2, 25), prctile(SSIM2, 75));
