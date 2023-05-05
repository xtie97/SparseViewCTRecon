clear; clc; close all;

refw = 12; refw_1 = 14; refw_2 = 16;
refi = 0.73; refi_1 = 0.64; refi_2 = 0.80; 

%%
x = [10, 20, 30, 40]; 
y1 = [36.58, 29.44, 24.54, 24.14];
y2 = [1.195, 1.081, 0.950, 0.929];

y1_low = (y1 - [29.54, 25.21, 20.03, 19.615]) / 24.54 * refw; 
y1_high = ([50.950, 36.174, 29.863, 28.597] - y1) / 24.54 * refw; 

y2_low = (y2 - [0.936, 0.829, 0.794, 0.765]) / 0.950 * refi; 
y2_high = ([1.548, 1.305, 1.224, 1.189] - y2) / 0.950 * refi; 

y1 = y1 / 24.54 * refw; 
y2 = y2 / 0.950 * refi; 

figure(1); 
bar(x, y1);  
xticks([10 20 30 40])
xticklabels({'10 (2)','20 (4)','30 (6)', '40 (8)'}); 
xlabel(sprintf('Number of patients used in training \n(number of patients used in validation)'));
ylabel('RMSE (mg/ml)'); title('Water basis image'); 
hold on; 
er = errorbar(x, y1, y1_low, y1_high);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off;

figure(2); 
bar(x, y2);  
xticks([10 20 30 40])
xticklabels({'10 (2)','20 (4)','30 (6)', '40 (8)'}); 
xlabel(sprintf('Number of patients used in training \n(number of patients used in validation)'));
ylabel('RMSE (mg/ml)'); title('Iodine basis image'); 
hold on; 
er = errorbar(x, y2, y2_low, y2_high);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off;