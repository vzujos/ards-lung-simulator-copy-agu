%% root

root = 'D:\ARAOS-PIGS\CORNELLU-PIGS-GROUPED\PIG2\BaselineInjury\';
root = 'C:\Users\angus\Downloads\CORNELL-NEWGEO\PIG5\ARDSnet\';

root_nii = strcat(root,'NIFTI\');
elem_size = 8;
size_str = int2str(elem_size);
root_mesh =  strcat(root,'MESH-',size_str,"/");


seg_name = "NEW_Mask_Exp.nii.gz"; % lung seg name
coarse_mesh_name = "coarse_lung.mat";
smooth_mesh_name = "smooth_lung.off";
out_tetmesh = "tet_lung.mat";


% Generate a surface mesh 

impath = strcat(root_nii, seg_name)

img = niftiread(impath);
% element size
opt = elem_size;
% execute algorithm
ix = 1:size(img,1);
iy = 1:size(img,2);
iz = 1:size(img,3);
[node,elem,face] = vol2surf(img,ix,iy,iz,opt,1,'cgalsurf',1);

% repair mesh
% remove duplicated nodes and elements 
[node, elem] = meshcheckrepair(node,elem,'dup'); 
% remove isolated nodes
[node, elem] = meshcheckrepair(node,elem,'isolated');
% test a surface for self-intersecting elements
[node, elem] = meshcheckrepair(node,elem,'intersect');
% call external jmeshlib to remove non-manifold vertices
[node, elem] = meshcheckrepair(node,elem,'deep');
% abort when open surface is found
[node, elem] = meshcheckrepair(node,elem,'open');

% export coarse surface mesh
%mkdir(root_mesh)
matname = strcat(root_mesh,coarse_mesh_name);
save(matname,"node","elem","face");

%% quality check
quality=meshquality(node,elem);

histogram(quality) % 1 is good and 0 is bad

%% read off

matname = strcat(root_mesh, smooth_mesh_name);
[node,elem] = readoff(matname);

v=node; 
f=elem;
d_ = 0.5;
d = [d_,d_,d_];
p0 = max(node)+d;
p1 = min(node)-d;
keepratio = 0.90; % closer to 1 generates better surface but more elements
maxvol=5000.0;
regions=[];
holes=[];
forcebox=0;
method='tetgen'
%[node,elem,face]=surf2mesh(v,f,p0,p1,keepratio,maxvol,regions,holes,forcebox,method,cmdopt)
[tetnode,tetelem,tetface]=surf2mesh(v,f,p0,p1,keepratio,maxvol,regions,holes,forcebox,method);

[nelem,dim]=size(tetelem);
[nnodes,dim]=size(tetnode);



fprintf("%s: %i\n","number of elements",nelem);
fprintf("%s: %i\n","number of nodes",nnodes);

matname = strcat(root_mesh, out_tetmesh);
save(matname,"tetnode","tetelem","tetface");

outquality=meshquality(tetnode,tetelem);
histogram(outquality) % 1 is good and 0 is bad