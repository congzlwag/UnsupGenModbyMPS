classdef MPS < handle
    % Matrix product state for unsupervised learning
    %   Pan Zhang 08.26.2017 panzhang@itp.ac.cn
    properties
        n=20; % number of spins
        current_bond=1; % current bond
        m=1; % number of training samples
        
        data; % training data, size n x m, where n is the system size and m is number of training samples. This is because Matlab uses column-major storage of multi-arrays.
        batched_data; % indexs
        batch_idx;
        n_batches=1; % number of batches
        batch_size=1;
        max_bondim=80; % maximum bond dimension
        min_bondim=2; % minimum bond dimension
        bond_dims=[];
        tensors={}; % each of its element is a three-way tensor, with size Dl,2,Dr.
        going_right=true;
        merged_tensor=[];  %This array will be set only by merge_bond(), and set to [] only by rebuild_bond().
        cutoff=0.000001;
        learning_rate=0.001;
        cumulants={};
        psi=[]; % wave function for each sample (state)
        log_file=''; % if ~isempty(log_file), MPS will store the status of current run.
        converge_crit=0.002; % Stop training when difference of nll between two loops is smaller than converge_crit.
        nll_history=[];
    end
    
    methods
        function self=MPS(n,data,n_batches)
            self.n=n;
            self.data=data;
            self.n_batches=n_batches;
            %% initialize tensors randomly
            self.tensors{1}=randn(1,2,self.min_bondim);
            %self.tensors{1}=ones(1,2,self.min_bondim); % or from all-one tensors
            for i=2:n-1
                self.tensors{i}=randn(self.min_bondim,2,self.min_bondim);
                %self.tensors{i}=ones(self.min_bondim,2,self.min_bondim);
                self.bond_dims(i-1)=self.min_bondim;
            end
            self.bond_dims(self.n)=1;
            %self.tensors{n}=randn(self.min_bondim,2,1);
            self.tensors{n}=ones(self.min_bondim,2,1);
            self.bond_dims(n-1)=self.min_bondim;
            self.make_batches();
            self.current_bond=1;
            self.left_canonical();
            
        end
        function make_batches(self)
            assert(size(self.data,1)==self.n);
            self.m=size(self.data,2);
            assert(mod(self.m,self.n_batches)==0);
            self.batch_size=self.m/self.n_batches;
            self.batch_idx = reshape(1:self.m,self.batch_size,self.n_batches); % batch_size * n_batch
        end 
        function left_canonical(self)
            self.going_right=true;
            for bond=self.current_bond:self.n-1
                self.current_bond=bond;
                self.merge_bond();
                self.rebuild_bond(true);
            end
        end
        function merge_bond(self)
            % This function sets self.merged_tensor
            assert(self.current_bond>0 && self.current_bond<self.n);
            self.merged_tensor = tensor_product(self.tensors{self.current_bond},'ijk',self.tensors{self.current_bond+1},'kmn');
            return
        end
        function rebuild_bond(self,fix_bondim)
            %rebuild merged_tensor into two self.tensors{bl} and
            % self.merged_tensor will be set to [] at the end of this funciton
            dl=size(self.tensors{self.current_bond},1);
            dr=size(self.tensors{self.current_bond+1},3);
            D=size(self.tensors{self.current_bond},3);
            self.merged_tensor=reshape(self.merged_tensor,2*dl,2*dr);
            [U,S,V]=svd(self.merged_tensor);
            if(~fix_bondim)
                s=diag(S)';
                [~,idx]=find(s>=self.cutoff*s(1));
                D=max(idx);
            end
            D=min(D,self.max_bondim);
            D=min(D,dl*2);
            D=min(D,dr*2);
            U=U(:,1:D);
            S=S(1:D,1:D);
            V=V(:,1:D)';
            if(self.going_right)
                V=S*V;
                V=V./norm(V,'fro');
            else
                U=U*S;
                U=U./norm(U,'fro');
            end
            self.tensors{self.current_bond}=reshape(U,[dl,2,D]);
            self.bond_dims(self.current_bond)=D;
            self.tensors{self.current_bond+1}=reshape(V,[D,2,dr]);
            self.merged_tensor=[];
        end
        
        function train(self,n_loops)
            self.init_cumulants();
            assert(self.current_bond==self.n-1); % this should be set properly during the left_canonical();
            %fprintf('#0 nll=%.6f\n',self.compute_nll());
            nll_new=Inf;
            for loop=1:n_loops
                tic;
                for batch=1:self.n_batches
                    %% left sweep
                    self.going_right=false; % going left
                    for bond=self.n-1:-1:2
                        self.current_bond=bond;
                        self.merge_bond();
                        self.gradient_descent(batch);
                        self.rebuild_bond(false);
                        self.update_cumulants();
                    end
                                    
                    self.going_right=true; % going right
                    for bond=1:self.n-2
                        self.current_bond=bond;
                        self.merge_bond();
                        self.gradient_descent(batch);
                        self.rebuild_bond(false);
                        self.update_cumulants();
                    end
                end
                nll_old=nll_new;
                nll_new=self.compute_nll();
                fprintf('#%d nll=%.3f, max_bondim=%d, <bondim>=%.4f\n',loop,nll_new,max(self.bond_dims),mean(self.bond_dims));
                toc;
                self.nll_history(loop)=nll_new;
                if(~isempty(self.log_file))
                    fprintf('saving to %s\n',self.log_file);
                    mpsf=sprintf('%s.mps.mat',self.log_file);
                    mps=self;
                    save(mpsf,'mps');
                    s=self.generate_sample(100)-1;
                    matf=sprintf('%s.sample.mat',self.log_file);
                    save(matf,'s');
                end
                if(nll_new > nll_old || abs(nll_new-nll_old)<self.converge_crit)
                    break;
                end
            end
        end
        
        function init_cumulants(self)
           self.cumulants = {};
           self.cumulants{1} = ones(self.m,1);
           for i=2:self.n-1
               self.cumulants{i} = ones(self.m,self.bond_dims(i-1));
               for a=1:self.m
                   self.cumulants{i}(a,:)=self.cumulants{i-1}(a,:)*reshape(self.tensors{i-1}(:,self.data(i-1,a),:),size(self.tensors{i-1},1),size(self.tensors{i-1},3));
               end
           end
           i=self.n; % for computing psi
           for a=1:self.m
               self.psi(a)=self.cumulants{i-1}(a,:)*squeeze(self.tensors{i-1}(:,self.data(i-1,a),:))*squeeze(self.tensors{i}(:,self.data(i,a),:));
           end
           assert(self.current_bond==self.n-1); % this should be set properly during the left_canonical();
           self.cumulants{self.n}=ones(self.m,1);
        end
                
        function gradient_descent(self,batch)
            if(self.current_bond==1)
                Dl=1;
                Dr=self.bond_dims(self.current_bond+1);
            elseif(self.current_bond==self.n-1)
                Dr=1;
                Dl=self.bond_dims(self.current_bond-1);
            else
                Dl=self.bond_dims(self.current_bond-1);
                Dr=self.bond_dims(self.current_bond+1);
            end
                
            nominator = zeros(Dl,Dr,self.batch_size);
            for i=1:self.batch_size
                idx=self.batch_idx(i,batch); % index of sample in self.data
                sample=self.data(:,idx);
                lvec=self.cumulants{self.current_bond}(idx,:);  % 1*Dl
                rvec=self.cumulants{self.current_bond+1}(idx,:); % 1*Dr
                nominator(:,:,i) = lvec'*rvec;
                tmp=reshape(self.merged_tensor(:,sample(self.current_bond),sample(self.current_bond+1),:),Dl,Dr);
                self.psi(idx) = lvec*tmp*rvec';% 1* Dl * Dl*Dr * Dr*1  -> psi value
            end
            data_idx = self.batch_idx(:,batch); % all indices (of self.data) in the batch
            samples=self.data(:,data_idx);
            for si=1:2
                for sj=1:2
                    idx = logical( (samples(self.current_bond,:) == si ) .* (samples(self.current_bond+1,:)==sj) ); % index in the current batch !
                    a=nominator(:,:,idx);
                    b=self.psi(data_idx(idx));
                    gradient=zeros(Dl,Dr);
                    if(size(a,3)==0) 
                        gradient=-2*self.merged_tensor(:,si,sj,:);
                    else
                        for i=1:size(a,3)                        
                            gradient=gradient+a(:,:,i)./b(i)*2;
                        end
                        gradient =reshape(gradient./self.batch_size,size(self.merged_tensor(:,si,sj,:)))- 2*self.merged_tensor(:,si,sj,:);
                    end
                    self.merged_tensor(:,si,sj,:) = self.merged_tensor(:,si,sj,:) + self.learning_rate .* gradient;
                end
            end
            self.merged_tensor = self.merged_tensor/norm(reshape(self.merged_tensor,Dl*2,Dr*2),'fro');
        end
        
        function update_cumulants(self)
            % this function is to be called after calling rebuild_bond()
            if((self.current_bond==1 && (~self.going_right)) || (self.current_bond==self.n-1 && self.going_right))
                return
            end
            if(self.going_right)
                self.cumulants{self.current_bond+1} = ones(self.m,self.bond_dims(self.current_bond));
                for a=1:self.m
                    self.cumulants{self.current_bond+1}(a,:)=self.cumulants{self.current_bond}(a,:)*squeeze(self.tensors{self.current_bond}(:,self.data(self.current_bond,a),:));
                end
            else % going left
                self.cumulants{self.current_bond} = ones(self.m,self.bond_dims(self.current_bond));
                for a=1:self.m
                    if(self.current_bond==self.n-1)
                        tmp=reshape(self.tensors{self.current_bond+1}(:,self.data(self.current_bond+1,a),:),self.bond_dims(self.current_bond),1);
                    else
                        tmp=reshape(self.tensors{self.current_bond+1}(:,self.data(self.current_bond+1,a),:),self.bond_dims(self.current_bond),self.bond_dims(self.current_bond+1));
                    end
                    self.cumulants{self.current_bond}(a,:)=tmp* (self.cumulants{self.current_bond+1}(a,:))' ; % Dl*Dr * Dr*1 ->  Dl*1
                end
            end
        end
        
        function nll=compute_nll(self)
            nll=-mean(log(abs(self.recompute_psi()).^2));
            return
        end
            
        function [samples]=generate_sample(self,n_samples)
            self.left_canonical();
            samples=ones(self.n,n_samples);
            for a=1:n_samples
                rvec=1;
                for i=self.n:-1:1
                    vec1=reshape(self.tensors{i}(:,1,:),size(self.tensors{i},1),size(self.tensors{i},3))*rvec; %D*1
                    vec2=reshape(self.tensors{i}(:,2,:),size(self.tensors{i},1),size(self.tensors{i},3))*rvec; %D*1
                    n1=vec1'*vec1;
                    n2=vec2'*vec2;
                    if(rand()< (n2/(n1+n2)))
                        samples(i,a)=2;
                        rvec=vec2;
                    else
                        rvec=vec1;
                    end
                end
            end
            return;
        end
        
        function [psi]=recompute_psi(self) % compute psi from scratch
            if(isempty(self.merged_tensor))
                for idata=1:self.m
                    sample=self.data(:,idata);
                    psi=1;
                    for i=1:self.n
                        psi = psi* reshape(self.tensors{i}(:,sample(i),:),size(self.tensors{i},1),size(self.tensors{i},3));%1*Dl
                    end
                    self.psi(idata)=psi;
                end
            else
               for idata=1:self.m
                   sample=self.data(:,idata);
                    lvec=1;
                    for i=1:self.current_bond-1
                        lvec = lvec* reshape(self.tensors{i}(:,sample(i),:),size(self.tensors{i},1),size(self.tensors{i},3));%1*Dl
                    end
                    rvec=1;
                    for i=self.n:-1:self.current_bond+2
                        rvec = reshape(self.tensors{i}(:,sample(i),:),size(self.tensors{i},1),size(self.tensors{i},3))*rvec; %Dr*1
                    end
                    psi=squeeze(lvec*reshape( self.merged_tensor(:,sample(self.current_bond),sample(self.current_bond+1),:),size(lvec,2),size(rvec,1) )*rvec);
                    self.psi(idata)=psi;
               end
            end
            psi=self.psi;
            return
        end
    end
end

