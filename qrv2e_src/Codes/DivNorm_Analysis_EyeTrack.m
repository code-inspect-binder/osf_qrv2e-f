function DivNorm_Analysis_EyeTrack
% Divisive Normalization Project
% Experiment: Eye Tracking
% 
% This script runs the following analyses:
% - frequentist statistics (t-tests, regression analyses)
% - Hierarchical Bayesian analyses (with JAGS)
% - Model comparison between probit and divisive normalization
% - eye-tracking analyses (only to produce figures; stats are run in R)
%
% The following additional files are required to run this script:
% - "behavdata_for_Matlab" (behavioral data prepared in R)
% - "fixdata_for_Matlab" (eye-tracking data prepared in R)
% - LogReg_V3_JAGS.txt (for the hierarchical Bayesian analyses)
% - gs_probit_fast.m (for the model comparison)
% - fs_probit_fast.m (for the model comparison)
% - gs_norm_fast.m (for the model comparison)
% - fs_norm_fast.m (for the model comparison)
% - fminsearchcon.m (for the model comparison)
%
% The following tools/libraries are required to run this script
% - matjags (for the hierarchical Bayesian analyses)
% - parallel computing (for the hierarchical Bayesian analyses - but could
%   also be switched off)
%
% Note that the required input files "behavdata_for_Matlab" and
% "fixdata_for_Matlab" are created in the R script "DivNorm_R_EyeTrack.R",
% which needs to be run first (unless the input files exist already)
%
% Note that some of these analyses are very time consuming (esp. the
% Bayesian analysis and the model comparison). I recommend going through it
% step by step.
%
% sebastian.gluth@unibas.ch


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load and prepare data %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

datapath = pwd; %assume that data is in same folder (adapt if necessary)

%behavior
behavdata = csvread([datapath,'/behavdata_for_Matlab'],1,0);
behavheader = {'item1','item2','item3','distractor','difficulty','rating1','rating2','rating3','position1','position2','position3',...
               'choice','RT','rating1c','rating2c','rating3c','position1c','position2c','position3c','choice123','trialnr','ID'};
%fixations
fixdata = csvread([datapath,'/fixdata_for_Matlab'],1,0);
fixheader = {'fixdur','intrialonset','AOI','trialnr','ID'};
isfirstfix = [1;fixdata(2:end,ismember(fixheader,'trialnr'))~=fixdata(1:end-1,ismember(fixheader,'trialnr'))]==1;
% islastfix = [fixdata(2:end,ismember(fixheader,'trialnr'))~=fixdata(1:end-1,ismember(fixheader,'trialnr'));1]==1;

IDs = unique(behavdata(:,ismember(behavheader,'ID')));
nsubj = length(IDs);

fxdata.V3 = []; %will be dataset for fixed effects analysis
fxdata.normV3 = []; %will be dataset for fixed effects analysis
%within-subject analyses
for s = 1:nsubj
    
    %preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %descriptive statistics
    descriptives.best_chosen(s) = mean(cdata(:,ismember(behavheader,'choice123'))==1);
    descriptives.second_best_chosen(s) = mean(cdata(:,ismember(behavheader,'choice123'))==2);
    descriptives.distractor_chosen(s) = mean(cdata(:,ismember(behavheader,'choice123'))==3);
    descriptives.RT(s) = mean(cdata(:,ismember(behavheader,'RT')));
    
    %clean up data: exclude any trials in which targets were not chosen
    targetchosen = cdata(:,ismember(behavheader,'choice123'))<=2;
    V1c = V1(targetchosen);
    V2c = V2(targetchosen);
    V3c = V3(targetchosen);
    normV3c = normV3(targetchosen);
    Choice = cdata(targetchosen,ismember(behavheader,'choice123'))==1; %1 = best chosen, 0 = 2nd best chosen
    
    %analysis 1a: choice efficiency as a function of high vs. low V3
    V3chigh = V3c>median(V3c);
    analysis.a1(s,:) = [mean(Choice(V3chigh==1)),mean(Choice(V3chigh==0))];
    %analysis 1b: choice efficiency as a function of high vs. low normV3
    normV3chigh =  normV3c>median(normV3c);
    analysis.b1(s,:) = [mean(Choice(normV3chigh==1)),mean(Choice(normV3chigh==0))];
    
    %analysis 2a: multiple regression with V3
    X = [V1c,V2c,V3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.a2.betas(s,:) = glmfit(Xz,Choice,'binomial');
    fxdata.V3 = [fxdata.V3;[X,Choice,zeros(length(X),1)+s]]; %prepare analysis 3a (fixed effects multiple regression)
    %analysis 2b: multiple regression with normV3
    X = [V1c,V2c,normV3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.b2.betas(s,:) = glmfit(Xz,Choice,'binomial');
    fxdata.normV3 = [fxdata.normV3;[X,Choice,zeros(length(X),1)+s]]; %prepare analysis 3b (fixed effects multiple regression)
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Frequentist statistics %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%exclude no (additional) subjects (done in R or even prior to R)
inclsubj = 1:nsubj; %for random effects analyses
incltrials = fxdata.V3(:,5)>0; %for fixed effects analyses

%analysis 1a and 1b on the group level
%(choice efficiency as a function of high vs. low V3/normV3)
[~,p,~,t] = ttest(analysis.a1(inclsubj,1),analysis.a1(inclsubj,2),'tail','left');
[~,~,ci,~] = ttest(analysis.a1(inclsubj,:));
d = mean(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2))./std(analysis.a1(inclsubj,1)-analysis.a1(inclsubj,2));
disp(['t-test of high vs. low V3: t(',int2str(t.df),') = ',num2str(t.tstat,3),'; p = ',num2str(p,3),'; d = ',num2str(d,3)])

figure;hold on;title('Eye-tracking experiment');ylabel('Relative choice accuracy');
xlim([0.5,2.5]);ylim([0.25,0.95])
plot([1,1],ci(:,1),'k-','LineWidth',10);plot([2,2],ci(:,2),'k-','LineWidth',10)
plot(ones(length(inclsubj),1),analysis.a1(inclsubj,1),'.','Color',[.5,.5,.5]);
plot(ones(length(inclsubj),1)+1,analysis.a1(inclsubj,2),'.','Color',[.5,.5,.5])
plot([0.8,1.2],zeros(1,2)+mean(analysis.a1(inclsubj,1)),'-','LineWidth',2,'Color',[.25,1,.25])
plot([1.8,2.2],zeros(1,2)+mean(analysis.a1(inclsubj,2)),'m-','LineWidth',2,'Color',[0,.5,0])
set(gca,'XTick',1:2,'XTickLabel',{'high V3','low V3'},'YTick',.3:.2:.9,'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,8,8]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_LowHigh_V3_R1')

[~,p,~,t] = ttest(analysis.b1(inclsubj,1),analysis.b1(inclsubj,2),'tail','left');
% [~,~,ci,~] = ttest(analysis.b1(inclsubj,:));
d = mean(analysis.b1(inclsubj,1)-analysis.b1(inclsubj,2))./std(analysis.b1(inclsubj,1)-analysis.b1(inclsubj,2));
disp(['t-test of high vs. low normV3: t(',int2str(t.df),') = ',num2str(t.tstat,3),'; p = ',num2str(p,3),'; d = ',num2str(d,3)])

%analysis 2a and 2b on the group level
%(multiple logistic random-effects regression with V3/normV3)
[~,p(2)] = ttest(analysis.a2.betas(inclsubj,2),zeros(size(analysis.a2.betas(inclsubj,2))),'tail','right');
[~,p(3:4)] = ttest(analysis.a2.betas(inclsubj,3:4),zeros(size(analysis.a2.betas(inclsubj,3:4))),'tail','left');
[~,~,ci,t] = ttest(analysis.a2.betas(inclsubj,:));
d = mean(analysis.a2.betas(inclsubj,:))./std(analysis.a2.betas(inclsubj,:));
disp(['t-test of beta(V3): t(',int2str(t.df(4)),') = ',num2str(t.tstat(4),3),'; p = ',num2str(p(4),3),'; d = ',num2str(d(4),3)])

figure;hold on;title('Eye-tracking experiment');ylabel('Effect on relative choice accuracy')
xlim([0.5,3.5]);ylim([-1.5,1.5]);
plot([0.5,3.5],[0,0],'k-')
plot([1,1],ci(:,2),'k-','LineWidth',10);plot([2,2],ci(:,3),'k-','LineWidth',10);plot([3,3],ci(:,4),'k-','LineWidth',10);
% plot(zeros(length(inclsubj),1)+1,analysis.a2.betas(inclsubj,2),'.','Color',[.5,.5,.5]);
% plot(zeros(length(inclsubj),1)+2,analysis.a2.betas(inclsubj,3),'.','Color',[.5,.5,.5]);
plot(zeros(length(inclsubj),1)+3,analysis.a2.betas(inclsubj,4),'.','Color',[.5,.5,.5]);
plot([0.8,1.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,2)),'b-','LineWidth',2)
plot([1.8,2.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,3)),'r-','LineWidth',2)
plot([2.8,3.2],zeros(1,2)+mean(analysis.a2.betas(inclsubj,4)),'g-','LineWidth',2)
set(gca,'XTick',1:3,'XTickLabel',{'V1','V2','V3'},'FontSize',12)
% set(gcf,'PaperPosition',[1/4,1/4,8,8]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_LogReg_V3_R1')

[~,p,~,t] = ttest(analysis.b2.betas(inclsubj,:));
d = mean(analysis.b2.betas(inclsubj,:))./std(analysis.b2.betas(inclsubj,:));
disp(['t-test of beta(normV3): t(',int2str(t.df(4)),') = ',num2str(t.tstat(4),3),'; p = ',num2str(p(4),3),'; d = ',num2str(d(4),3)])

%analysis 3a and 3b
%(multiple logistic fixed-effects regression with V3/normV3)
[analysis.a3.B,analysis.a3.DEV,analysis.a3.STATS] = glmfit(fxdata.V3(incltrials,1:3),fxdata.V3(incltrials,4),'binomial');
[analysis.b3.B,analysis.b3.DEV,analysis.b3.STATS] = glmfit(fxdata.normV3(incltrials,1:3),fxdata.normV3(incltrials,4),'binomial');

%pre-analysis for response times: identify cut-off for RT (> mean + 4*SD)
allRT = behavdata(:,ismember(behavheader,'RT'));
RTcut = mean(allRT)+4*std(allRT); %note: min RT is 349 ms, so that's fine

%perform analysis of RT
for s = 1:nsubj %note: the excluded subject is still analyzed, so make sure to exclude her when running the statistics
    
	%preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    %clean up data: exclude any trials in which targets were not chosen
    include_trials = (cdata(:,ismember(behavheader,'choice123'))<=2)&(cdata(:,ismember(behavheader,'RT'))<=RTcut);
    V1c = V1(include_trials);
    V2c = V2(include_trials);
    V3c = V3(include_trials);
    normV3c = normV3(include_trials);
    RT = cdata(include_trials,ismember(behavheader,'RT'));
        
    %multiple regression of RT with V3
    X = [V1c-V2c,V1c+V2c,V3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.RTa.betas(s,:) = glmfit(Xz,RT); %analysis of RT
    %analysis 2b: multiple regression with normV3
    X = [V1c-V2c,V1c+V2c,normV3c]; %independent variables
    Xz = (X-repmat(mean(X),length(X),1))./repmat(std(X),length(X),1); %standardize independent variables
    analysis.RTb.betas(s,:) = glmfit(Xz,RT); %analysis of RT
    
    %plot RT as a function of V3 (similar to Fig. 5A in Louie et al., 2013)
    ncatRT = 5;
    V3cat = sortrows([sortrows([(1:length(V3))',V3],2),sort(repmat((1:ncatRT)',length(V3)/(ncatRT),1))],1)*[0;0;1];
    V3ccat = V3cat(include_trials);
    for c = 1:ncatRT
        ct = V3ccat==c; %trials belonging to current category
        analysis.RTb.visualization_V3(s,c) = mean(RT(ct));
    end
    
end

%figure for RT effects
[~,~,ci,~] = ttest(analysis.RTa.betas(inclsubj,2:end));
figure('OuterPosition',[440 378 600 800]);hold on;title('Eye-tracking experiment');ylabel('Effect on response time')
xlim([0.5,3.5]);ylim([-1000,500])
plot([0.5,3.5],[0,0],'k-')
for x = 1:size(ci,2)
    plot([x,x],ci(:,x),'k-','LineWidth',10);
    plot(zeros(length(inclsubj),1)+x,analysis.RTa.betas(inclsubj,x+1),'.','Color',[.5,.5,.5])
end
plot([0.8,1.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,2)),'b-','LineWidth',2)
plot([1.8,2.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,3)),'r-','LineWidth',2)
plot([2.8,3.2],zeros(1,2)+nanmean(analysis.RTa.betas(inclsubj,4)),'g-','LineWidth',2)
set(gca,'XTick',1:3,'XTickLabel',{'V1 - V2','V1 + V2','V3'},'FontSize',16)
% set(gcf,'PaperPosition',[1/4,1/4,12,12]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_LinReg_RT_R1')

%plot RT similar Figure 5A
[~,~,ci,~] = ttest(analysis.RTb.visualization_V3(inclsubj,:));
figure;hold on;xlim([0.5,5.5]);ylim([1550,2300])
for x = 1:ncatRT
    plot([x,x],ci(:,x),'k-','LineWidth',6);
    plot([0.8,1.2]+(x-1),zeros(1,2)+nanmean(analysis.RTb.visualization_V3(inclsubj,x)),'g-','LineWidth',3)
end
plot(nanmean(analysis.RTb.visualization_V3(inclsubj,:)),'g-')
set(gca,'XTick',[1,5],'XTickLabel',{'lowest','highest'},'YTick',1600:200:2200,'FontSize',16)
xlabel('V3');ylabel('RT in ms')
% set(gcf,'PaperPosition',[1/4,1/4,12,9]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_RTbyV3')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Hierarchical Bayesian analyses %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

inclsubj = 1:nsubj;
close all
addpath /Applications/MATLAB_ADD/matjags

%specify data inputs for JAGS
jagsdata.V1 = [];
jagsdata.V2 = [];
jagsdata.V3 = [];
jagsdata.normV3 = [];
jagsdata.C = [];
jagsdata.S = [];
for s = 1:length(inclsubj)
    
    %preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    normV3 = 2*V3./(V1+V2); %value of distractor "normalized" to value of targets
    
    C = cdata(:,ismember(behavheader,'choice123')); %1 = best chosen, 2 = 2nd best chosen, 3 = worst chosen
    
    %clean up data: exclude any trials in which targets were not chosen
    jagsdata.V1 = [jagsdata.V1;V1(C<=2)];
    jagsdata.V2 = [jagsdata.V2;V2(C<=2)];
    jagsdata.V3 = [jagsdata.V3;V3(C<=2)];
    jagsdata.normV3 = [jagsdata.normV3;normV3(C<=2)];
    jagsdata.C = [jagsdata.C;C(C<=2)==1]; %1 = best chosen, 0 = 2nd best chosen
    jagsdata.S = [jagsdata.S;zeros(sum(C<=2),1)+s];
end

% Specify a reasonable prior SD of group-level coefficients
priorSD = 5;

% Assign Matlab Variables to the Observed Nodes
N = size(jagsdata.C,1);
% subj = inclsubj;
subj = 1:length(inclsubj);
jags_data = struct('priorSD',priorSD,'N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.V3);
               
% JAGS settings
doparallel = 1;
nchains = 4;
nburnin = 2000;
nsamples = 10000;
thin = 10;

% Initial Values
clear init0
for iv = 1:nchains
    I.muintercept = (rand-0.5)*priorSD;
    I.muwV1 = (rand-0.5)*priorSD;
    I.muwV2 = (rand-0.5)*priorSD;
    I.muwV3 = (rand-0.5)*priorSD;
    I.sigmaintercept = (rand+0.2)*priorSD;
    I.sigmawV1 = (rand+0.2)*priorSD;
    I.sigmawV2 = (rand+0.2)*priorSD;
    I.sigmawV3 = (rand+0.2)*priorSD;
    init0(iv) = I;
end

%start parallel pool
p = gcp;
if isempty(p)
    parpool(4);
end

% Use JAGS to Sample
if size(dir('tmpjags'),1)>0
    rmdir('tmpjags','s')
end
tic
fprintf( 'Running JAGS ...\n' );
[samples_with_V3, stats_with_V3] =  matjags(jags_data,fullfile(pwd,'LogReg_V3_JAGS.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'muintercept','muwV1','muwV2','muwV3',...
                     'sigmaintercept','sigmawV1','sigmawV2','sigmawV3',...
                     'intercept','wV1','wV2','wV3'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_with_V3.Rhat.muintercept;stats_with_V3.Rhat.muwV1;stats_with_V3.Rhat.muwV2;stats_with_V3.Rhat.muwV3;...
            stats_with_V3.Rhat.sigmaintercept;stats_with_V3.Rhat.sigmawV1;stats_with_V3.Rhat.sigmawV2;stats_with_V3.Rhat.sigmawV3;...
            stats_with_V3.Rhat.intercept;stats_with_V3.Rhat.wV1;stats_with_V3.Rhat.wV2;stats_with_V3.Rhat.wV3;stats_with_V3.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%get Bayes Factors (Savage-Dickey method)
prior = normpdf(0,0,priorSD);
posterior_V1 = normpdf(0,mean(samples_with_V3.muwV1(1:nsamples*nchains)),std(samples_with_V3.muwV1(1:nsamples*nchains)));
BF10.V1 = prior./posterior_V1;
posterior_V2 = normpdf(0,mean(samples_with_V3.muwV2(1:nsamples*nchains)),std(samples_with_V3.muwV2(1:nsamples*nchains)));
BF10.V2 = prior./posterior_V2;
posterior_V3 = normpdf(0,mean(samples_with_V3.muwV3(1:nsamples*nchains)),std(samples_with_V3.muwV3(1:nsamples*nchains)));
BF10.V3 = prior./posterior_V3;

figure;hold on;xlabel('Regression coefficient of V3');ylabel('Frequency');xlim([-0.1,0.1])
histogram(samples_with_V3.muwV3(1:nsamples*nchains),x,'FaceColor','g','EdgeColor','g');
p3 = plot(x,y*nsamples*nchains*normpdf(x,mean(samples_with_V3.muwV3(1:nsamples*nchains)),std(samples_with_V3.muwV3(1:nsamples*nchains))),'LineWidth',3,'Color','g');
histogram(randn(nsamples*nchains,1)*priorSD,x,'FaceColor','k','EdgeColor','k');
p0 = plot(x,y*nsamples*nchains*normpdf(x,0,priorSD),'LineWidth',3,'Color','k');
legend([p0,p3],{'Prior','Posterior'});legend boxoff
set(gca,'XTick',-.1:.05:.1,'YTick',0:1000:3000)
% set(gcf,'PaperPosition',[1/4,1/4,16,8]);print(gcf,'-dtiff','-r300','../Figures/EyeStudy_BayesianRegression')

% THE SAME BUT WITH normV3 INSTEAD OF V3
jags_data = struct('priorSD',priorSD,'N',N,'subj',subj,'nsubj',length(subj),...
                   'S',jagsdata.S,'C',jagsdata.C,'V1',jagsdata.V1,'V2',jagsdata.V2,'V3',jagsdata.normV3);
               
% Initial Values
clear init0
for iv = 1:nchains
    I.muintercept = (rand-0.5)*priorSD;
    I.muwV1 = (rand-0.5)*priorSD;
    I.muwV2 = (rand-0.5)*priorSD;
    I.muwV3 = (rand-0.5)*priorSD;
    I.sigmaintercept = (rand+0.2)*priorSD;
    I.sigmawV1 = (rand+0.2)*priorSD;
    I.sigmawV2 = (rand+0.2)*priorSD;
    I.sigmawV3 = (rand+0.2)*priorSD;
    init0(iv) = I;
end

% Use JAGS to Sample
if size(dir('tmpjags'),1)>0
    rmdir('tmpjags','s')
end
tic
fprintf( 'Running JAGS ...\n' );
[samples_with_normV3, stats_with_normV3] =  matjags(jags_data,fullfile(pwd,'LogReg_V3_JAGS.txt'),init0, ...
	'doparallel' , doparallel,'nchains', nchains,'nburnin', nburnin,'nsamples', nsamples,'thin',thin,...
	'monitorparams',{'muintercept','muwV1','muwV2','muwV3',...
                     'sigmaintercept','sigmawV1','sigmawV2','sigmawV3',...
                     'intercept','wV1','wV2','wV3'},...
	'savejagsoutput',1,'verbosity',1,'cleanup',0,'workingdir','tmpjags');
toc

%check that all R-hats are below 1.01
allRhats = [stats_with_normV3.Rhat.muintercept;stats_with_normV3.Rhat.muwV1;stats_with_normV3.Rhat.muwV2;stats_with_normV3.Rhat.muwV3;...
            stats_with_normV3.Rhat.sigmaintercept;stats_with_normV3.Rhat.sigmawV1;stats_with_normV3.Rhat.sigmawV2;stats_with_normV3.Rhat.sigmawV3;...
            stats_with_normV3.Rhat.intercept;stats_with_normV3.Rhat.wV1;stats_with_normV3.Rhat.wV2;stats_with_normV3.Rhat.wV3;stats_with_normV3.Rhat.deviance];
if max(allRhats) < 1.01
    disp('all R-hats are below 1.01')
else
    disp('WARNING: not all R-hats are below 1.01!')
end

%get Bayes Factors (Savage-Dickey method)
prior = normpdf(0,0,priorSD);
posterior_V1 = normpdf(0,mean(samples_with_normV3.muwV1(1:nsamples*nchains)),std(samples_with_normV3.muwV1(1:nsamples*nchains)));
BF10.V1_with_normV3 = prior./posterior_V1;
posterior_V2 = normpdf(0,mean(samples_with_normV3.muwV2(1:nsamples*nchains)),std(samples_with_normV3.muwV2(1:nsamples*nchains)));
BF10.V2_with_normV3 = prior./posterior_V2;
posterior_V3 = normpdf(0,mean(samples_with_normV3.muwV3(1:nsamples*nchains)),std(samples_with_normV3.muwV3(1:nsamples*nchains)));
BF10.normV3 = prior./posterior_V3;

%first figure is to check whether normal distribution approximations hold
% figure;hold on;
subplot(2,1,2);hold on;
histogram(randn(nsamples*nchains,1)*priorSD,x,'FaceColor','k','EdgeColor','k');
p0 = plot(x,y*nsamples*nchains*normpdf(x,0,priorSD),'LineWidth',3,'Color','k');
histogram(samples_with_normV3.muwV1(1:nsamples*nchains),x,'FaceColor','b','EdgeColor','b');
p1 = plot(x,y*nsamples*nchains*normpdf(x,mean(samples_with_normV3.muwV1(1:nsamples*nchains)),std(samples_with_normV3.muwV1(1:nsamples*nchains))),'LineWidth',3,'Color','b');
histogram(samples_with_normV3.muwV2(1:nsamples*nchains),x,'FaceColor','r','EdgeColor','r');
p2 = plot(x,y*nsamples*nchains*normpdf(x,mean(samples_with_normV3.muwV2(1:nsamples*nchains)),std(samples_with_normV3.muwV2(1:nsamples*nchains))),'LineWidth',3,'Color','r');
histogram(samples_with_normV3.muwV3(1:nsamples*nchains),x,'FaceColor','g','EdgeColor','g');
p3 = plot(x,y*nsamples*nchains*normpdf(x,mean(samples_with_normV3.muwV3(1:nsamples*nchains)),std(samples_with_normV3.muwV3(1:nsamples*nchains))),'LineWidth',3,'Color','g');
legend([p0,p1,p2,p3],{'prior','posterior of V1','posterior of V2','posterior of normV3'});legend boxoff
% set(gcf,'PaperPosition',[1/4,1/4,36,18]);print(gcf,'-dtiff','-r300','Figures/ReplStudy_Posterior_V123_ID3included')


%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%% Model comparison %%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

close all
%general settings
fei = [2000,1000]; %max number of function evaluations and iterations
fs = [1,3,5,1]; %number of fminsearches with best parameter estimates from grid search; random pick of best 30%/1%; random pick; nested (or random again)

%probit model settings
probit.gs = 10; %number of gridsearches per parameter for probit model
probit.params_minmax = [eps;30]; %the free parameter of this model is the standard deviation of the normally distributed "choice error"
probit.params_range = linspace(0.1,18.1,probit.gs); %grid search parameter range for probit model

%normalization model settings
norm.gs = probit.gs+zeros(1,5); %number of gridsearches per parameter for normalization model
norm.params_minmax = [0,eps,0,eps,0;200,200,10,30,10];%gain, semisaturation, weight term, fixed noise, scaled noise
norm.prepr = [1,1,0,0.1,0;181,181,1.8,18.1,1.8]; %preparation of grid search parameter range for normalization model
for x = 1:length(norm.prepr)
    norm.params_range(x,:) = linspace(norm.prepr(1,x),norm.prepr(2,x),norm.gs(x));
end

for s = 1:nsubj
%     tic
    %preparations
    cdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    
    V3 = cdata(:,ismember(behavheader,'rating3c')); %value of worst option (distractor)
    V2 = cdata(:,ismember(behavheader,'rating2c')); %value of 2nd best option (target 2)
    V1 = cdata(:,ismember(behavheader,'rating1c')); %value of best option (target 1)
    
    C = cdata(:,ismember(behavheader,'choice123')); %1 = best chosen, 2 = 2nd best chosen, 3 = worst chosen
    Vcu = [V3.*(C==3)+V2.*(C==2)+V1.*(C==1),V2.*(C==1)+V1.*(C==2)+V1.*(C==3),V3.*(C==1)+V3.*(C==2)+V2.*(C==3)]; %value of chosen, better unchosen and worse unchosen option
    ntrials = length(C);
    
    %probit model, grid search
    deviance = gs_probit_fast(Vcu,ntrials,probit.params_range,probit.gs);
    probit.dev_gs(s,:) = deviance;
    %fminsearch
    preparams = zeros(sum(fs),1);
	predev = zeros(sum(fs),1);
	best_gs = sortrows([deviance,(1:probit.gs)'])*[0;1];
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            params_init = probit.params_range(best_gs(1));
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good
            params_init = probit.params_range(best_gs(randi(round(probit.gs/3))+1));
        else %start with random
            params_init = probit.params_range(randi(probit.gs));
        end
        [preparams(x),predev(x)] = fs_probit_fast(Vcu,params_init,probit.params_minmax,fei);
    end
    probit.params(s) = preparams(find(predev==min(predev),1));
    probit.dev_fs(s) = predev(find(predev==min(predev),1));
    %simulation
    for t = 1:100
        M = [V1,V2,V3]*100+randn(size(Vcu))*probit.params(s);
        probit.prediction(s,:,t) = mean(repmat(max(M,[],2),1,3)==M);
    end
    
	%normalization model, grid search
    deviance = gs_norm_fast(Vcu,ntrials,norm.params_range,norm.gs(1));
    norm.dev_gs{s} = deviance;
	%fminsearch
    preparams = zeros(sum(fs),length(norm.params_minmax));
	predev = zeros(sum(fs),1);
	preexitflag = zeros(sum(fs),1);
    thresh_deviance = max(((1:prod(norm.gs))==round(prod(norm.gs)*.01)).*sort(reshape(deviance,prod(norm.gs),1))');
	[D1,D2,D3,D4,D5] = ind2sub(size(deviance),find(deviance<=thresh_deviance)); %best 1%
	[D1b,D2b,D3b,D4b,D5b] = ind2sub(size(deviance),find(deviance==min(min(min(min(min(deviance))))))); %best
    for x = 1:sum(fs)
        if x <= fs(1) %start with best
            r = randi(length(D1b)); %in case of multiple best values
            params_init = [norm.params_range(1,D1b(r)),norm.params_range(2,D2b(r)),norm.params_range(3,D3b(r)),norm.params_range(4,D4b(r)),norm.params_range(5,D5b(r))];
        elseif (x > fs(1)) && (x <= sum(fs(1:2))) %start with good
            r = randi(length(D1));
            params_init = [norm.params_range(1,D1(r)),norm.params_range(2,D2(r)),norm.params_range(3,D3(r)),norm.params_range(4,D4(r)),norm.params_range(5,D5(r))];
        elseif (x > sum(fs(1:2))) && (x <= sum(fs(1:3))) %start with random
            params_init = [norm.params_range(1,randi(norm.gs(1))),norm.params_range(2,randi(norm.gs(2))),norm.params_range(3,randi(norm.gs(3))),norm.params_range(4,randi(norm.gs(4))),norm.params_range(5,randi(norm.gs(5)))];
        else %start with best of probit
            params_init = [100,50,0,probit.params(s),0];
        end
        [preparams(x,:),predev(x),preexitflag(x)] = fs_norm_fast(Vcu,params_init,norm.params_minmax,fei);
    end
	norm.params(s,:) = preparams(find(predev==min(predev),1),:);
    norm.dev_fs(s) = predev(find(predev==min(predev),1));
	norm.exitflag(s) = preexitflag(find(predev==min(predev),1));
	norm.allexitflags(s,:) = preexitflag;
    
    save(['ws_DivNorm_Analsis_EyeTrack_with_Modeling_gs',int2str(probit.gs)]) %commented here, but it makes sense to save after each subject in case something happens after a long time
    disp(['done model fitting of subject #',int2str(s),' ...'])
%     toc
end

%figure showing the fit improvement from probit to norm in each subject
figure('OuterPosition',[440 378 1920/2 1080/2]);hold on;
bar(sort(probit.dev_fs(inclsubj)-norm.dev_fs(inclsubj),'descend'),'FaceColor','k');
plot([0,length(inclsubj)+1],zeros(1,2)+chi2inv(.95,4),'k--')
text(length(inclsubj)*.9,chi2inv(.95,4)*.9,'{\it p} < .05')
xlabel('Participants');ylabel('Improvement of deviance')
% set(gcf,'PaperPosition',[1/4,1/4,32,8]);print(gcf,'-dtiff','-r300','../Figures/EyeStudy_ProbitNormalization')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Eye-tracking analyses %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%get the relevant data
behavdata(:,size(behavdata,2)+1) = behavdata(:,ismember(behavheader,'ID'))*1000+behavdata(:,ismember(behavheader,'trialnr'));
fixdata(:,size(fixdata,2)+1) = fixdata(:,ismember(fixheader,'ID'))*1000+fixdata(:,ismember(fixheader,'trialnr'));
behavheader{length(behavheader)+1} = 'trialID';
fixheader{length(fixheader)+1} = 'trialID';

%Bottom-up question: Does the probability of first fixation depend on 
%whether it's the best, 2nd-best or worst option?

%Top-down question: How do fixation patterns develop within trials? Do 
%fixations on worst options decrease and does this depend on value?

%run loop with within-subject and within-trial analyses
%pre-allocate variables for QUESTION 1
first_fixation_probabilities = zeros(nsubj,3);
middle_fixation_probabilities = zeros(nsubj,3);
%pre-allocate variables for QUESTION 2
fixation_in_trials_all = zeros(nsubj,58); %(there were never more than 58 fixations in a single trial)
fixation_in_trials_best = zeros(nsubj,58);
fixation_in_trials_2nd = zeros(nsubj,58);
fixation_in_trials_worst = zeros(nsubj,58);
fixation_in_trials_worst_high = zeros(nsubj,58);
fixation_in_trials_worst_low = zeros(nsubj,58);
%pre-allocate variables for other questions
fixdur.all.first_best = [];
fixdur.all.first_2nd = [];
fixdur.all.first_D = [];
fixdur.all.middle_best = [];
fixdur.all.middle_2nd = [];
fixdur.all.middle_D = [];
for s = 1:nsubj
    s_behavdata = behavdata(behavdata(:,ismember(behavheader,'ID'))==IDs(s),:);
    s_fixdata = fixdata(fixdata(:,ismember(fixheader,'ID'))==IDs(s),:);
    s_validtrials = s_behavdata(s_behavdata(:,ismember(behavheader,'RT'))<RTcut,ismember(behavheader,'trialID'));
    
    %Bottom-up question
    fixdur.subj(s).first_best = [];
    fixdur.subj(s).first_2nd = [];
    fixdur.subj(s).first_D = [];
    fixdur.subj(s).middle_best = [];
    fixdur.subj(s).middle_2nd = [];
    fixdur.subj(s).middle_D = [];
    for t = 1:length(s_validtrials)
        ct = s_validtrials(t);
        t_behavdata = s_behavdata(s_behavdata(:,ismember(behavheader,'trialID'))==ct,:);
        t_fixdata = s_fixdata(s_fixdata(:,ismember(fixheader,'trialID'))==ct,:);
        if size(t_fixdata,1)>1 
            fixposition = (t_fixdata(:,ismember(fixheader,'AOI'))==2)*1+(t_fixdata(:,ismember(fixheader,'AOI'))==4)*2+(t_fixdata(:,ismember(fixheader,'AOI'))==3)*3;
            optposition = [t_behavdata(:,ismember(behavheader,'position1c')),t_behavdata(:,ismember(behavheader,'position2c')),t_behavdata(:,ismember(behavheader,'position3c'))];
            fixdurs = t_fixdata(:,ismember(fixheader,'fixdur'));
            if fixposition(1) == optposition(1)
                fixdur.subj(s).first_best = [fixdur.subj(s).first_best;fixdurs(1)];
            elseif fixposition(1) == optposition(2)
                fixdur.subj(s).first_2nd = [fixdur.subj(s).first_2nd;fixdurs(1)];
            elseif fixposition(1) == optposition(3)
                fixdur.subj(s).first_D = [fixdur.subj(s).first_D;fixdurs(1)];
            end
        end
        if size(t_fixdata,1)>2 %only analyze data with at least 3 fixations (to have at least 1 middle fixation)
            fixposition = (t_fixdata(:,ismember(fixheader,'AOI'))==2)*1+(t_fixdata(:,ismember(fixheader,'AOI'))==4)*2+(t_fixdata(:,ismember(fixheader,'AOI'))==3)*3;
            optposition = [t_behavdata(:,ismember(behavheader,'position1c')),t_behavdata(:,ismember(behavheader,'position2c')),t_behavdata(:,ismember(behavheader,'position3c'))];
            fixdurs = t_fixdata(:,ismember(fixheader,'fixdur'));
            for u = 2:size(t_fixdata,1)-1
                if fixposition(u) == optposition(1)
                    fixdur.subj(s).middle_best = [fixdur.subj(s).middle_best;fixdurs(u)];
                elseif fixposition(u) == optposition(2)
                    fixdur.subj(s).middle_2nd = [fixdur.subj(s).middle_2nd;fixdurs(u)];
                elseif fixposition(u) == optposition(3)
                    fixdur.subj(s).middle_D = [fixdur.subj(s).middle_D;fixdurs(u)];
                end
            end 
        end
    end
    nfirst = [length(fixdur.subj(s).first_best),length(fixdur.subj(s).first_2nd),length(fixdur.subj(s).first_D)];
    nmiddle = [length(fixdur.subj(s).middle_best),length(fixdur.subj(s).middle_2nd),length(fixdur.subj(s).middle_D)];
    first_fixation_probabilities(s,:) = nfirst./sum(nfirst);
    middle_fixation_probabilities(s,:) = nmiddle./sum(nmiddle);

	%Top-down question
    fit_best = zeros(length(s_validtrials),58);
    fit_2nd = zeros(length(s_validtrials),58);
    fit_worst = zeros(length(s_validtrials),58);
    for t = 1:length(s_validtrials)
        ct = s_validtrials(t);
        t_behavdata = s_behavdata(s_behavdata(:,ismember(behavheader,'trialID'))==ct,:);
        t_fixdata = s_fixdata(s_fixdata(:,ismember(fixheader,'trialID'))==ct,:);
        
        if size(t_fixdata,1)>1 %only analyze data with at least 2 fixations (because the last fixation is not analyzed)
            fixposition = (t_fixdata(:,ismember(fixheader,'AOI'))==2)*1+(t_fixdata(:,ismember(fixheader,'AOI'))==4)*2+(t_fixdata(:,ismember(fixheader,'AOI'))==3)*3;
            optposition = [t_behavdata(:,ismember(behavheader,'position1c')),t_behavdata(:,ismember(behavheader,'position2c')),t_behavdata(:,ismember(behavheader,'position3c'))];
            fixations_in_trial_t = sum((repmat(fixposition,1,length(optposition))==repmat(optposition,length(fixposition),1)).*repmat(1:3,length(fixposition),1),2);
            fit_best(t,1:length(fixations_in_trial_t)-1) = fixations_in_trial_t(1:end-1)==1;
            fit_2nd(t,1:length(fixations_in_trial_t)-1) = fixations_in_trial_t(1:end-1)==2;
            fit_worst(t,1:length(fixations_in_trial_t)-1) = fixations_in_trial_t(1:end-1)==3;
        end
    end
    fixation_in_trials_all(s,:) = sum(fit_best+fit_2nd+fit_worst);
    fixation_in_trials_best(s,:) = sum(fit_best)./fixation_in_trials_all(s,:);
    fixation_in_trials_2nd(s,:) = sum(fit_2nd)./fixation_in_trials_all(s,:);
    fixation_in_trials_worst(s,:) = sum(fit_worst)./fixation_in_trials_all(s,:);
    %split into high vs. low worst option
    value_worst = s_behavdata(s_behavdata(:,ismember(behavheader,'RT'))<RTcut,ismember(behavheader,'rating3c'));
    highD = value_worst>median(value_worst);
	fixation_in_trials_worst_high(s,:) = sum(fit_worst(highD==1,:))./sum(fit_best(highD==1,:)+fit_2nd(highD==1,:)+fit_worst(highD==1,:));
    fixation_in_trials_worst_low(s,:) = sum(fit_worst(highD==0,:))./sum(fit_best(highD==0,:)+fit_2nd(highD==0,:)+fit_worst(highD==0,:));
end

%%%Figure for Bottom-up question

%figure
[~,~,c1,~] = ttest(first_fixation_probabilities);
% [~,~,c2,~] = ttest(middle_fixation_probabilities);
nsubj = length(unique(behavdata(:,ismember(behavheader,'ID'))));
figure;hold on;xlim([0.5,3.5]);ylim([.08,.64]);xlabel('Option');ylabel('First fixation probability')
% p1 = plot((1:3)-0,mean(first_fixation_probabilities),'c-','LineWidth',2);
% p2 = plot((1:3)+.1,mean(middle_fixation_probabilities),'m-','LineWidth',2);
option_colors = {'b','r','g'};
for x = 1:size(c1,2)
    plot([x,x]-0,c1(:,x),'k-','LineWidth',10)
%     plot([x,x]+.1,c2(:,x),'k-','LineWidth',10)
    plot(zeros(nsubj,1)+x-0,first_fixation_probabilities(:,x),'.','Color',[.5,.5,.5])
%     plot(zeros(nsubj,1)+x+.1,middle_fixation_probabilities(:,x),'.','Color',[.5,.5,.5])
    plot((x-1)+[0.8,1.2]-0,zeros(1,2)+mean(first_fixation_probabilities(:,x)),[option_colors{x},'-'],'LineWidth',2);
%     plot((x-1)+[0.8,1.2]+.1,zeros(1,2)+mean(middle_fixation_probabilities(:,x)),'m-','LineWidth',2);
end
% plot((1:3)-0,mean(first_fixation_probabilities),'c-','LineWidth',2);
% plot((1:3)+.1,mean(middle_fixation_probabilities),'m-','LineWidth',2);
% legend([p1,p2],{'first fixation','middle fixations'});legend boxoff
plot([1,3],zeros(1,2)+max(max(first_fixation_probabilities))*1.05,'k-')
% text(2,max(max(first_fixation_probabilities))*1.07,'***')
set(gca,'XTick',1:3,'XTickLabel',{'Best','Second-best','Distractor'},'YTick',.1:.1:1,'FontSize',16)
% set(gcf,'PaperPosition',[1/4,1/4,14,12]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_FirstFixProb_R1')

%%%Figure for Top-down question

nshow = 11;
[~,~,c1,~] = ttest(fixation_in_trials_best(:,1:nshow));
[~,~,c2,~] = ttest(fixation_in_trials_2nd(:,1:nshow));
[~,~,c3,~] = ttest(fixation_in_trials_worst(:,1:nshow));
figure;hold on;xlabel('Fixation number (within trial)');ylabel('Fixation probability');ylim([0.05,0.53])
p1 = plot((1:nshow)-.1,nanmean(fixation_in_trials_best(:,1:nshow)),'b-','LineWidth',1);
p2 = plot((1:nshow),nanmean(fixation_in_trials_2nd(:,1:nshow)),'r-','LineWidth',1);
p3 = plot((1:nshow)+.1,nanmean(fixation_in_trials_worst(:,1:nshow)),'g-','LineWidth',1);
for x = 1:nshow
    plot([x,x]-.1,c1(:,x),'k-','LineWidth',3);
    plot([x,x],c2(:,x),'k-','LineWidth',3);
    plot([x,x]+.1,c3(:,x),'k-','LineWidth',3);
%     plot(zeros(nsubj,1)+x-.1,fixation_in_trials_best(:,x),'.','Color',[.5,.5,.5])
%     plot(zeros(nsubj,1)+x,fixation_in_trials_2nd(:,x),'.','Color',[.5,.5,.5])
%     plot(zeros(nsubj,1)+x+.1,fixation_in_trials_worst(:,x),'.','Color',[.5,.5,.5])
    plot((x-1)+[0.8,1.2]-.1,zeros(1,2)+nanmean(fixation_in_trials_best(:,x)),'b-','LineWidth',3);
    plot((x-1)+[0.8,1.2],zeros(1,2)+nanmean(fixation_in_trials_2nd(:,x)),'r-','LineWidth',3);
    plot((x-1)+[0.8,1.2]+.1,zeros(1,2)+nanmean(fixation_in_trials_worst(:,x)),'g-','LineWidth',3);
end
plot((1:nshow)-.1,nanmean(fixation_in_trials_best(:,1:nshow)),'b-','LineWidth',1);
plot((1:nshow),nanmean(fixation_in_trials_2nd(:,1:nshow)),'r-','LineWidth',1);
plot((1:nshow)+.1,nanmean(fixation_in_trials_worst(:,1:nshow)),'g-','LineWidth',1);
legend([p1,p2,p3],{'Best','Second-best','Distractor'},'Location','NorthWest');legend boxoff
set(gca,'YTick',.1:.1:.5,'FontSize',16)
% set(gcf,'PaperPosition',[1/4,1/4,16,12]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_FixDevelop_R1')

[~,~,c3a,~] = ttest(fixation_in_trials_worst_high(:,1:nshow));
[~,~,c3b,~] = ttest(fixation_in_trials_worst_low(:,1:nshow));
figure;hold on;xlabel('Fixation number (within trial)');ylabel('Fixation probability');ylim([0.05,0.53])
p3a = plot((1:nshow)-.1,nanmean(fixation_in_trials_worst_high(:,1:nshow)),'-','Color',[.25,1,.25],'LineWidth',1);
p3b = plot((1:nshow)+.1,nanmean(fixation_in_trials_worst_low(:,1:nshow)),'-','Color',[0,.5,0],'LineWidth',1);
for x = 1:nshow
    plot([x,x]-.1,c3a(:,x),'k-','LineWidth',3);
    plot([x,x]+.1,c3b(:,x),'k-','LineWidth',3);
%     plot(zeros(nsubj,1)+x-.1,fixation_in_trials_worst_high(:,x),'.','Color',[.5,.5,.5])
%     plot(zeros(nsubj,1)+x+.1,fixation_in_trials_worst_low(:,x),'.','Color',[.5,.5,.5])
    plot((x-1)+[0.8,1.2]-.1,zeros(1,2)+nanmean(fixation_in_trials_worst_high(:,x)),'-','Color',[.25,1,.25],'LineWidth',3);
    plot((x-1)+[0.8,1.2]+.1,zeros(1,2)+nanmean(fixation_in_trials_worst_low(:,x)),'-','Color',[0,.5,0],'LineWidth',3);    
end
plot((1:nshow)-.1,nanmean(fixation_in_trials_worst_high(:,1:nshow)),'-','Color',[.25,1,.25],'LineWidth',1);
plot((1:nshow)+.1,nanmean(fixation_in_trials_worst_low(:,1:nshow)),'-','Color',[0,.5,0],'LineWidth',1);
legend([p3a,p3b],{'High-value distractor','Low-value distractor'},'Location','NorthWest');legend boxoff
set(gca,'YTick',.1:.1:.5,'FontSize',16)
% set(gcf,'PaperPosition',[1/4,1/4,16,12]);print(gcf,'-dtiff','-r300','../Figures/R1/EyeStudy_FixDevelop_HighLowV3_R1')

end