clear; close all;
%%
exam = 65;
slice_id = 62;
root_dir = '/media/xintie/Elements/DeepEnChroma/Data_rcn'; 

%%
filename_high = sprintf('exam%d_high_%d_rcn.raw', exam, slice_id);
CT_sparse_61 = read_raw(fullfile(root_dir, 'sparse_view_61', filename_high), 'single', [512,512]) - 1024;
CT_sparse_123 = read_raw(fullfile(root_dir, 'sparse_view_123', filename_high), 'single', [512,512]) - 1024;
CT_sparse_246 = read_raw(fullfile(root_dir, 'sparse_view_246', filename_high), 'single', [512,512]) - 1024;

CT_dense_61 = read_raw(fullfile(root_dir, 'results_grad_61/output', filename_high), 'single', [512,512]) - 1024;
CT_dense_123 = read_raw(fullfile(root_dir, 'results_grad_123/output', filename_high), 'single', [512,512]) - 1024;
CT_dense_246 = read_raw(fullfile(root_dir, 'results_grad_246/output', filename_high), 'single', [512,512]) - 1024;

CT_dense_sino_61 = read_raw(fullfile(root_dir, 'results_sino_61/output', filename_high), 'single', [512,512]) - 1024;
CT_dense_sino_123 = read_raw(fullfile(root_dir, 'results_sino_123/output', filename_high), 'single', [512,512]) - 1024;
CT_dense_sino_246 = read_raw(fullfile(root_dir, 'results_sino_246/output', filename_high), 'single', [512,512]) - 1024;

CT_dense = read_raw(fullfile(root_dir, 'dense_view', filename_high), 'single', [512,512]) - 1024;

%%
display_window = [-200, 200]; 
MgSetFigureTheme("dark");
f = figure; f.Position = [0, 0, 1500, 1000];
t = tiledlayout(3, 4,'TileSpacing','Compact','Padding','Compact');
ax1=nexttile; imshow(CT_sparse_61, display_window); colormap(ax1, 'gray'); 
ax2=nexttile; imshow(CT_sparse_123, display_window); colormap(ax2, 'gray'); 
ax3=nexttile; imshow(CT_sparse_246, display_window); colormap(ax3, 'gray'); 
ax4=nexttile; imshow(CT_dense, display_window); colormap(ax4, 'gray'); 

ax5=nexttile; imshow(CT_dense_61, display_window); colormap(ax5, 'gray'); 
ax6=nexttile; imshow(CT_dense_123, display_window); colormap(ax6, 'gray'); 
ax7=nexttile; imshow(CT_dense_246, display_window); colormap(ax7, 'gray'); 
ax8=nexttile; imshow(zeros(512), []); 

ax9=nexttile; imshow(CT_dense_sino_61, display_window); colormap(ax9, 'gray'); 
ax10=nexttile; imshow(CT_dense_sino_123, display_window); colormap(ax10, 'gray'); 
ax11=nexttile; imshow(CT_dense_sino_246, display_window); colormap(ax11, 'gray'); 
ax12=nexttile; imshow(zeros(512), []); 

exportgraphics(t, sprintf('exam%d_%d.jpg', exam, slice_id), 'BackgroundColor', [0 0 0])
