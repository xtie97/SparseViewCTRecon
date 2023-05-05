clear; clc; close all;
%%
df_test = readtable('/media/xintie/Elements/DeepEnChroma/Data_rcn/test_index.csv');
output_dir_1 =  '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_sino_246/output_rcn';
output_dir_2 =  '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_grad_246/output';
target_dir =  '/media/xintie/Elements/DeepEnChroma/Data_rcn/results_sino_246/output';

CT_filenames = df_test.high;

rng(716); 
for ii = 1: length(CT_filenames)
    disp(ii/length(CT_filenames)*100); 
    filename = replace(CT_filenames{ii}, '.raw', '_rcn.raw');
    output1 = read_raw(fullfile(output_dir_1, filename), 'float32', [512, 512]);
    output2 = read_raw(fullfile(output_dir_2, filename), 'float32', [512, 512]);
    alpha = rand(); 
    output = output1*alpha + output2 * (1-alpha);
    write_raw(fullfile(target_dir, filename), output, 'float32'); 
end



% counts: 117   116   139   109   116   103   123   115
% Simple MSE:
% MAE: 26.4957   28.5892   31.3567   21.8290   26.8839   20.6150   25.6949   21.2181
% rRMSE: 2.0012    2.2552    2.1717    1.8841    2.0094    1.7185    2.0604    1.8318
% SSIM: 0.8467    0.8225    0.7984    0.8726    0.8523    0.8939    0.8326    0.8814

% MSE + Image gradient loss
% MAE: 25.3372   27.7816   29.9593   20.9380   26.6238   20.1762   25.2854   20.4996
% rRMSE: 1.8683    2.1340    2.0299    1.7339    1.8128    1.6612    2.0072    1.7321
% SSIM: 0.8552    0.8269    0.8108    0.8802    0.8427    0.8966    0.8245    0.8876

% Sinogram MSE + gradient loss
% MAE: 10.8176    9.8447   10.9039   10.7476   11.1231    9.4520   11.7605   11.1054
% rRMSE: 0.7514    0.7373    0.6761    0.8525    0.6775    0.7356    0.9183    0.8881
% SSIM: 0.9715    0.9759    0.9728    0.9683    0.9706    0.9764    0.9616    0.9660

