clear; clc; close all;
%%
x = [10, 20, 30, 40]; 
y1 = [7.5, 5.6, 4.5, 4.4]; 
y1_low = y1 - [5.6, 4.2, 3.3, 3.3];
y1_high = [9.1, 6.7, 5.3, 5.3] - y1;

y2 = [6.4, 5.4, 5.1, 5.0]; 
y2_low = y2 - [5.2, 4.4, 4.1, 4.0];
y2_high = [7.6, 6.3, 5.9, 5.8] - y2;

figure(1); 
bar(x, y1);  
xticks([10 20 30 40])
xticklabels({'10 (2)','20 (4)','30 (6)', '40 (8)'}); 
xlabel(sprintf('Number of patients used in training \n(number of patients used in validation)'));
ylabel('rRMSE (%)'); title('Water basis image'); 
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
ylabel('rRMSE (%)'); title('Iodine basis image'); 
hold on; 
er = errorbar(x, y2, y2_low, y2_high);    
er.Color = [0 0 0];                            
er.LineStyle = 'none';  

hold off;