%% IMT-2020 Channel Model Software
%% Copyright:Zhang Jianhua Lab, Beijing University of Posts and Telecommunications (BUPT)
%% Editor:Zhang Jianhua (ZJH), Tian Lei (TL)
%% Version: 1.0   Date: Dec.15, 2017

clc;  
clear ;

N_r = 32
N_t = 16
N_sc = 64
J = 100

path = sprintf('%s',fileparts(mfilename('fullpath')));
cd(path)

%% Channel coefficient generation for 1 UT-BS link with default settings. 
%Create folder to store data
cd ./SSP;
delete *.mat;
cd ../;
cd ./H;
delete *.mat;
cd ../;
cd ./LSP;
delete *.mat;
cd ../
cd ./LayoutParameters;
delete *.mat;
cd ../
cd ./ScenarioParameters;
delete *.mat;
cd ../
Input=struct('Sce','RMa_A',... %Set the scenario (InH_x, UMi_x, UMa_x, RMa_x)
    'C',19,...           %Set the number of Bs
    'N_user',19,...    %Set the number of subscribers per Bs
    'fc',6,...          %Set the center frequency (GHz)
    'AA',[1,1,10,1,1,2.5,2.5,0.5,0.5,102],... %AA=(Mg,Ng,M,N,P,dgH,dgV,dH,dV,downtilt)  BS antenna panel configuration,unit of d and dg is wave length.
    'sim',1,...         %Set the number of simulations
    'BW',200,...        %Set the bandwidth of the simulation(MHz)
     'T',10 );            %Set the number of sampling points of CIR in time domain

layoutpar=Layout(Input.Sce,Input.C,Input.N_user,Input.fc,Input.AA);
[Pathloss,SF_sigma]=GeneratePathloss(layoutpar);%Generate path loss and shadow fading.
fixpar=Scenario(Input.fc,layoutpar);%Generate scenario information. 
sigmas= GenerateLSP(layoutpar,fixpar);
GenerateSSP(layoutpar,fixpar,Input.sim,sigmas);%Generate small-scale parameters.
GenerateCIR(fixpar,layoutpar,Input.sim,Input.BW,Input.T);%Generate the channel coefficient.

